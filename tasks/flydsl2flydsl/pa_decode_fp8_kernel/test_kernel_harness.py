#!/usr/bin/env python3
"""Test harness for FlyDSL pa_decode_fp8_kernel (flydsl2flydsl)."""
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
# Constants and test shapes
# ============================================================================

HEAD_SIZE = 128
QUERY_GROUP_SIZE = 16
KV_BLOCK_SIZE = 16
FP8_DTYPE = None  # set at runtime

# (num_seqs, seq_len, num_kv_heads)
ALL_SHAPES = [
    (1, 128, 1),
    (1, 256, 1),
    (1, 512, 1),
    (2, 128, 2),
    (2, 256, 2),
    (4, 128, 4),
    (4, 256, 4),
    (8, 128, 8),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]


def _get_fp8_dtype():
    import torch
    global FP8_DTYPE
    if FP8_DTYPE is None:
        FP8_DTYPE = torch.float8_e4m3fnuz
    return FP8_DTYPE


# ============================================================================
# Helpers
# ============================================================================


def _simple_fp8_quantize(tensor):
    """Quantize a BF16/FP32 tensor to FP8 with per-tensor scaling."""
    import torch
    fp8_dt = _get_fp8_dtype()
    FP8_MAX = 240.0
    amax = tensor.abs().max().item()
    scale = FP8_MAX / max(amax, 1e-12)
    q = (tensor.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(fp8_dt)
    return q, torch.tensor(1.0 / scale, dtype=torch.float32, device=tensor.device)


def _create_test_data(num_seqs, seq_len, num_kv_heads):
    """Create paged KV cache test data."""
    import torch
    device = "cuda"
    num_query_heads = num_kv_heads * QUERY_GROUP_SIZE
    num_blocks_per_seq = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_blocks = num_seqs * num_blocks_per_seq + 10

    query = torch.randn(num_seqs, num_query_heads, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    x = 16
    key_cache_bf16 = torch.randn(
        total_blocks, num_kv_heads, HEAD_SIZE // x, KV_BLOCK_SIZE, x,
        dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    value_cache_bf16 = torch.randn(
        total_blocks, num_kv_heads, HEAD_SIZE, KV_BLOCK_SIZE,
        dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    block_tables = torch.zeros(num_seqs, num_blocks_per_seq,
                                dtype=torch.int32, device=device)
    for b in range(num_seqs):
        for i in range(num_blocks_per_seq):
            block_tables[b, i] = b * num_blocks_per_seq + i

    context_lengths = torch.full((num_seqs,), seq_len,
                                  dtype=torch.int32, device=device)

    kc_perm = key_cache_bf16.permute(0, 1, 3, 2, 4).reshape(
        total_blocks, num_kv_heads, KV_BLOCK_SIZE, -1).contiguous()
    kc_perm = kc_perm.view(total_blocks, num_kv_heads, KV_BLOCK_SIZE,
                            HEAD_SIZE // x, x).permute(0, 1, 3, 2, 4).contiguous()
    q_keys, key_scale = _simple_fp8_quantize(kc_perm)
    q_vals, val_scale = _simple_fp8_quantize(value_cache_bf16)

    return {
        "query": query,
        "key_cache_fp8": q_keys,
        "key_scale": key_scale,
        "value_cache_fp8": q_vals,
        "value_scale": val_scale,
        "key_cache_bf16": key_cache_bf16,
        "value_cache_bf16": value_cache_bf16,
        "block_tables": block_tables,
        "context_lengths": context_lengths,
        "num_blocks_per_seq": num_blocks_per_seq,
        "total_blocks": total_blocks,
    }


def _torch_ref_attention(data, num_kv_heads):
    """PyTorch reference paged attention."""
    import torch
    query = data["query"]
    key_cache = data["key_cache_bf16"]
    value_cache = data["value_cache_bf16"]
    block_tables = data["block_tables"]
    context_lengths = data["context_lengths"]

    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    softmax_scale = 1.0 / math.sqrt(HEAD_SIZE)

    kc_flat = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, HEAD_SIZE)
    vc_flat = value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_kv_heads, HEAD_SIZE)

    outputs = []
    for b in range(num_seqs):
        bt = block_tables[b]
        ctx_len = context_lengths[b].item()
        tok_idx = (bt.repeat_interleave(KV_BLOCK_SIZE)[:ctx_len] * KV_BLOCK_SIZE
                   + torch.arange(ctx_len, device="cuda") % KV_BLOCK_SIZE)

        keys = kc_flat[tok_idx]
        vals = vc_flat[tok_idx]
        q = query[b].float()

        group_size = num_query_heads // num_kv_heads
        head_outs = []
        for h in range(num_query_heads):
            kv_h = h // group_size
            qh = q[h]
            kh = keys[:, kv_h, :].float()
            vh = vals[:, kv_h, :].float()
            scores = (qh @ kh.T) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            head_outs.append(probs @ vh)
        outputs.append(torch.stack(head_outs))

    return torch.stack(outputs).to(torch.bfloat16)


# ============================================================================
# Modes
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    results, failures = [], []
    for i, (num_seqs, seq_len, num_kv_heads) in enumerate(shapes):
        try:
            torch.manual_seed(42 + i)
            data = _create_test_data(num_seqs, seq_len, num_kv_heads)
            num_query_heads = num_kv_heads * QUERY_GROUP_SIZE
            num_partitions = 1

            exe = mod.build_pa_decode_module(
                num_seqs=num_seqs,
                num_kv_heads=num_kv_heads,
                num_partitions=num_partitions,
                max_blocks_per_seq=data["num_blocks_per_seq"] + 10,
                query_scale=1.0,
                key_scale=data["key_scale"].item(),
                value_scale=data["value_scale"].item(),
                kv_block_size=KV_BLOCK_SIZE,
                one_shot=True,
            )

            output = torch.zeros(num_seqs, num_query_heads, HEAD_SIZE,
                                  dtype=torch.bfloat16, device="cuda")
            exe(data["query"], data["key_cache_fp8"], data["value_cache_fp8"],
                data["block_tables"], data["context_lengths"], output)
            torch.cuda.synchronize()

            ref = _torch_ref_attention(data, num_kv_heads)
            max_err = (output.float() - ref.float()).abs().max().item()
            passed = max_err < 0.15

            if not passed:
                raise AssertionError(f"max_err={max_err:.4e} > 0.15")

            results.append({"config": (num_seqs, seq_len, num_kv_heads), "correct": True})
            if verbose:
                print(f"  PASS: (seqs={num_seqs}, len={seq_len}, kv_heads={num_kv_heads}) max_err={max_err:.4e}")
        except Exception as e:
            failures.append({"config": (num_seqs, seq_len, num_kv_heads), "error": str(e)})
            if verbose:
                print(f"  FAIL: (seqs={num_seqs}, len={seq_len}, kv_heads={num_kv_heads}) - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_benchmark(shapes=None, warmup=10, iters=50, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config':<35} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 70)

    for idx, (num_seqs, seq_len, num_kv_heads) in enumerate(shapes):
        torch.manual_seed(42)
        num_query_heads = num_kv_heads * QUERY_GROUP_SIZE
        data = _create_test_data(num_seqs, seq_len, num_kv_heads)
        num_partitions = 1

        exe = mod.build_pa_decode_module(
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            num_partitions=num_partitions,
            max_blocks_per_seq=data["num_blocks_per_seq"] + 10,
            query_scale=1.0,
            key_scale=data["key_scale"].item(),
            value_scale=data["value_scale"].item(),
            kv_block_size=KV_BLOCK_SIZE,
            one_shot=True,
        )

        output = torch.zeros(num_seqs, num_query_heads, HEAD_SIZE,
                              dtype=torch.bfloat16, device="cuda")

        for _ in range(warmup):
            exe(data["query"], data["key_cache_fp8"], data["value_cache_fp8"],
                data["block_tables"], data["context_lengths"], output)
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            exe(data["query"], data["key_cache_fp8"], data["value_cache_fp8"],
                data["block_tables"], data["context_lengths"], output)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = _torch_ref_attention(data, num_kv_heads)
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
            "shape": [num_seqs, seq_len, num_kv_heads],
            "params": {"num_seqs": num_seqs, "seq_len": seq_len, "num_kv_heads": num_kv_heads},
        })

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"(seqs={num_seqs:>2}, len={seq_len:>4}, kv_h={num_kv_heads:>2})"
                f"       {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 70)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL PA Decode FP8 Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Paged Attention Decode FP8 Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result.get("correct", False) else 1)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
