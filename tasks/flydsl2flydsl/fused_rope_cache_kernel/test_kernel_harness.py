#!/usr/bin/env python3
"""Test harness for FlyDSL fused_rope_cache_kernel (flydsl2flydsl).

Tests the bf16, flash_layout=True, apply_scale=False path (most common
in vLLM-style inference). Validates Q_out and K_out RoPE correctness
against a PyTorch reference.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

# ============================================================================
# GEAK bootstrap
# ============================================================================

KERNEL_FILE = "kernel.py"


def _find_baseline_kernel_dir():
    work = os.environ.get("GEAK_WORK_DIR", "").strip()
    if not work:
        return None
    d = Path(work).resolve()
    for _ in range(10):
        if d is None or not d.exists():
            break
        if (d / "benchmark_baseline.txt").is_file():
            return str(d)
        d = d.parent
    return None


def _resolve_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    original = os.path.dirname(os.path.abspath(__file__))
    candidates.append(original)
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return original


def _load_kernel(kernel_dir, alias="flydsl_kernel"):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# ============================================================================
# Test configurations
# ============================================================================

ALL_CONFIGS = [
    {"num_tokens": 16, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64, "block_size": 16},
    {"num_tokens": 32, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64, "block_size": 16},
    {"num_tokens": 32, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16},
    {"num_tokens": 64, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "block_size": 16},
    {"num_tokens": 64, "num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128, "block_size": 16},
    {"num_tokens": 128, "num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128, "block_size": 16},
]

_n_all = len(ALL_CONFIGS)
HARNESS_CONFIGS = ALL_CONFIGS
_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(min(5, _n_all))]
PROFILE_CONFIGS = [ALL_CONFIGS[i] for i in _pidx]

RTOL, ATOL = 1e-2, 1e-2
MAX_POS = 4096
DTYPE_STR = "bf16"

# ============================================================================
# Reference
# ============================================================================


def reference_rope_neox(x, cos, sin, positions):
    import torch

    T_len, H, D = x.shape
    half = D // 2
    pos_cos = cos[positions].unsqueeze(1).expand(-1, H, -1)
    pos_sin = sin[positions].unsqueeze(1).expand(-1, H, -1)

    x_f32 = x.float()
    x_first = x_f32[..., :half]
    x_second = x_f32[..., half:]

    rotated_first = x_first * pos_cos - x_second * pos_sin
    rotated_second = x_second * pos_cos + x_first * pos_sin
    return torch.cat([rotated_first, rotated_second], dim=-1).to(x.dtype)


def _make_inputs(cfg, seed=42):
    import torch

    T_len = cfg["num_tokens"]
    QH, KH, D, BS = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["block_size"]
    half_d = D // 2

    torch.manual_seed(seed)
    Q = torch.randn(T_len, QH, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(T_len, KH, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(T_len, KH, D, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (T_len,), device="cuda", dtype=torch.int32)
    freqs = torch.randn(MAX_POS, half_d, device="cuda", dtype=torch.bfloat16)
    cos_cache = torch.cos(freqs.float()).to(torch.bfloat16)
    sin_cache = torch.sin(freqs.float()).to(torch.bfloat16)

    num_blocks = (T_len + BS - 1) // BS + 4
    slot_mapping = torch.arange(T_len, device="cuda", dtype=torch.int32)
    key_cache = torch.zeros(num_blocks, BS, KH, D, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.zeros(num_blocks, BS, KH, D, device="cuda", dtype=torch.bfloat16)
    Q_out = torch.empty_like(Q)
    K_out = torch.empty_like(K)
    k_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    v_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    return {
        "Q": Q, "K": K, "V": V, "positions": positions,
        "cos_cache": cos_cache, "sin_cache": sin_cache,
        "slot_mapping": slot_mapping, "key_cache": key_cache,
        "value_cache": value_cache, "Q_out": Q_out, "K_out": K_out,
        "k_scale": k_scale, "v_scale": v_scale,
        "T_len": T_len, "QH": QH, "KH": KH, "D": D, "BS": BS,
    }


# ============================================================================
# Modes
# ============================================================================


def run_correctness(configs=None, verbose=True):
    import torch

    if configs is None:
        configs = HARNESS_CONFIGS
    if verbose:
        print(f"Running correctness on {len(configs)} configs...")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"correct": False, "num_correct": 0, "num_failed": len(configs), "failures": []}

    results, failures = [], []
    for i, cfg in enumerate(configs):
        try:
            inp = _make_inputs(cfg, seed=42 + i)
            launch_fn = mod.build_fused_rope_cache_module(
                head_dim=inp["D"], num_q_heads=inp["QH"], num_kv_heads=inp["KH"],
                block_size=inp["BS"], is_neox=True, flash_layout=True,
                dtype_str=DTYPE_STR, apply_scale=False,
                reuse_freqs_front_part=True, pos_dtype="i32",
            )
            launch_fn(
                inp["Q"], inp["K"], inp["V"], inp["positions"],
                inp["cos_cache"], inp["sin_cache"], inp["slot_mapping"],
                inp["key_cache"], inp["value_cache"], inp["Q_out"], inp["K_out"],
                inp["T_len"], inp["k_scale"], inp["v_scale"],
            )
            torch.cuda.synchronize()

            q_ref = reference_rope_neox(inp["Q"], inp["cos_cache"], inp["sin_cache"], inp["positions"])
            k_ref = reference_rope_neox(inp["K"], inp["cos_cache"], inp["sin_cache"], inp["positions"])

            torch.testing.assert_close(inp["Q_out"], q_ref, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(inp["K_out"], k_ref, atol=ATOL, rtol=RTOL)

            expected_key_cache = torch.zeros_like(inp["key_cache"])
            expected_value_cache = torch.zeros_like(inp["value_cache"])
            slots = inp["slot_mapping"].to(torch.long)
            valid = slots >= 0
            block_ids = slots[valid] // inp["BS"]
            block_offsets = slots[valid] % inp["BS"]
            expected_key_cache[block_ids, block_offsets, :, :] = k_ref[valid]
            expected_value_cache[block_ids, block_offsets, :, :] = inp["V"][valid]
            torch.testing.assert_close(inp["key_cache"], expected_key_cache, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(inp["value_cache"], expected_value_cache, atol=ATOL, rtol=RTOL)

            label = f"T={cfg['num_tokens']},QH={cfg['num_q_heads']},D={cfg['head_dim']}"
            results.append({"config": label, "correct": True})
            if verbose:
                print(f"  PASS: {label}")
        except Exception as e:
            label = f"T={cfg['num_tokens']},QH={cfg['num_q_heads']},D={cfg['head_dim']}"
            failures.append({"config": label, "error": str(e)})
            if verbose:
                print(f"  FAIL: {label} - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(configs=None, warmup=50, iters=200, verbose=True):
    import torch

    if configs is None:
        configs = PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        return

    for cfg in configs:
        inp = _make_inputs(cfg)
        launch_fn = mod.build_fused_rope_cache_module(
            head_dim=inp["D"], num_q_heads=inp["QH"], num_kv_heads=inp["KH"],
            block_size=inp["BS"], is_neox=True, flash_layout=True,
            dtype_str=DTYPE_STR, apply_scale=False,
            reuse_freqs_front_part=True, pos_dtype="i32",
        )

        def _run():
            launch_fn(
                inp["Q"], inp["K"], inp["V"], inp["positions"],
                inp["cos_cache"], inp["sin_cache"], inp["slot_mapping"],
                inp["key_cache"], inp["value_cache"], inp["Q_out"], inp["K_out"],
                inp["T_len"], inp["k_scale"], inp["v_scale"],
            )

        for _ in range(warmup):
            _run()
        torch.cuda.synchronize()
        for _ in range(iters):
            _run()
        torch.cuda.synchronize()
        if verbose:
            print(f"  T={cfg['num_tokens']},QH={cfg['num_q_heads']},D={cfg['head_dim']} done")


def run_benchmark(configs=None, warmup=50, iters=200, verbose=True):
    import torch

    if configs is None:
        configs = HARNESS_CONFIGS

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(configs)} configs, {warmup} warmup, {iters} iterations...")
    print(f"  Comparing kernel vs PyTorch reference RoPE")
    print(f"{'Config':<36} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 72)

    for idx, cfg in enumerate(configs):
        inp = _make_inputs(cfg)
        launch_fn = mod.build_fused_rope_cache_module(
            head_dim=inp["D"], num_q_heads=inp["QH"], num_kv_heads=inp["KH"],
            block_size=inp["BS"], is_neox=True, flash_layout=True,
            dtype_str=DTYPE_STR, apply_scale=False,
            reuse_freqs_front_part=True, pos_dtype="i32",
        )

        def _run_kernel():
            launch_fn(
                inp["Q"], inp["K"], inp["V"], inp["positions"],
                inp["cos_cache"], inp["sin_cache"], inp["slot_mapping"],
                inp["key_cache"], inp["value_cache"], inp["Q_out"], inp["K_out"],
                inp["T_len"], inp["k_scale"], inp["v_scale"],
            )

        for _ in range(warmup):
            _run_kernel()
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _run_kernel()
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = reference_rope_neox(inp["Q"], inp["cos_cache"], inp["sin_cache"], inp["positions"])
            _ = reference_rope_neox(inp["K"], inp["cos_cache"], inp["sin_cache"], inp["positions"])
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)
        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "params": {
                "num_tokens": cfg["num_tokens"],
                "num_q_heads": cfg["num_q_heads"],
                "num_kv_heads": cfg["num_kv_heads"],
                "head_dim": cfg["head_dim"],
            },
        })

        label = f"T={cfg['num_tokens']:>3},QH={cfg['num_q_heads']:>2},KH={cfg['num_kv_heads']},D={cfg['head_dim']:>3}"
        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"{label:<36} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 72)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL Fused RoPE+Cache Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Fused RoPE + KV Cache Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_CONFIGS)
        sys.exit(0 if result.get("correct", False) else 1)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_CONFIGS, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_CONFIGS, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_CONFIGS, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
