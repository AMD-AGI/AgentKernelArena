#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl qk_norm_rope_quant task.

`model.py` is the pure-torch reference (bf16 RMSNorm + GPT-J RoPE). The FlyDSL
`kernel.py` runs its `quant=False` bf16 path, which computes the same math.

Correctness gate (element-wise): the normalized max error
``max|ref - out| / max|ref|`` (computed for Q and for KV, take the worse) must
be <= REL_TOL. The check asserts and exits non-zero on failure.

Modes:
  --correctness     assert the kernel matches the torch Model reference
  --full-benchmark  time FlyDSL vs the torch reference, write perf report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_tensor_contract, require_unchanged

from task_runtime import candidate_relative_path
KERNEL_FILE = candidate_relative_path()
ARENA_PROVIDED_BASELINE = False
MODEL_FILE = "model.py"
# Keep earlier correctness imports alive when performance reloads the alias.
# Old FlyDSL module finalizers may call hipModuleUnload during graph capture.
_LOADED_MODULES = []


def _resolve_kernel_dir():
    return str(__import__("pathlib").Path(__file__).resolve().parent)


def _load_module(kernel_dir, filename, alias):
    if filename == KERNEL_FILE and ARENA_PROVIDED_BASELINE:
        return None
    entry = os.path.join(kernel_dir, filename)
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
    _LOADED_MODULES.append(mod)
    if filename == KERNEL_FILE:
        _require_candidate_outputs(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# Shapes: D=512 / VEC=8 is the only kernel-supported head_dim (RD=64). Sweep T
# (decode batch / seq len) and H (Q head count); group_size in {32,64,128} does
# not change the bf16 math.
SHAPES = [
    {"name": "T1_H16", "T": 1, "H": 16, "D": 512, "RD": 64, "group_size": 64},
    {"name": "T16_H16", "T": 16, "H": 16, "D": 512, "RD": 64, "group_size": 64},
    {"name": "T64_H16", "T": 64, "H": 16, "D": 512, "RD": 64, "group_size": 32},
    {"name": "T256_H16", "T": 256, "H": 16, "D": 512, "RD": 64, "group_size": 128},
    {"name": "T64_H128", "T": 64, "H": 128, "D": 512, "RD": 64, "group_size": 64},
    {"name": "T512_H128", "T": 512, "H": 128, "D": 512, "RD": 64, "group_size": 64},
]

# Element-wise gate: normalized worst-element error <= REL_TOL.
REL_TOL = 1e-2
SEED = 0
_QLORA = 1536  # KV is a strided slice of a [T, QLORA+D] tensor


def _retry(fn, *, tries=5, what="op"):
    """Retry on transient OOM/contention (a 2nd worker may share the GPU)."""
    import torch

    delay = 0.5
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:  # noqa: PERF203
            msg = str(e).lower()
            transient = (
                "out of memory" in msg or "hip" in msg or "ran out" in msg
            )
            if not transient or attempt == tries - 1:
                raise
            print(
                f"  [retry] transient GPU error on {what} "
                f"(attempt {attempt + 1}/{tries}): {str(e)[:80]} — backing off {delay:.1f}s"
            )
            torch.cuda.empty_cache()
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _make_inputs(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    T, H, D, RD = shape["T"], shape["H"], shape["D"], shape["RD"]
    max_pos = max(T, 64)
    cos, sin = mmod._build_cos_sin(max_pos, RD, device=device)

    q = torch.randn(T, H * D, dtype=torch.bfloat16, device=device) * 0.1
    qkv_a = torch.randn(T, _QLORA + D, dtype=torch.bfloat16, device=device) * 0.1
    _, kv = torch.split(qkv_a, [_QLORA, D], dim=-1)  # strided view
    kv_weight = torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5
    positions = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64, device=device)
    return q, kv, kv_weight, cos, sin, positions


def _norm_max_err(ref, out):
    pass

    ref_f, out_f = ref.float(), out.float()
    max_abs = (ref_f - out_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-9
    return max_abs / denom, max_abs, denom


def _checked_qk_outputs(actual, expected):
    import torch
    if not isinstance(actual, (tuple, list)) or len(actual) != 4:
        raise AssertionError("quant=False requires (BF16 Q, BF16 KV, None, None)")
    if actual[2] is not None or actual[3] is not None:
        raise AssertionError("quant=False must not return quantization scales")
    for out, ref in zip(actual[:2], expected):
        require_tensor_contract(out, ref)
        if not bool(torch.isfinite(out).all() and torch.isfinite(ref).all()):
            raise AssertionError("Non-finite Q/KV output/reference")
        if _norm_max_err(ref, out)[0] > REL_TOL:
            raise AssertionError("Numerical mismatch: Q/KV normalized max error exceeds original gate")


def _qk_replay_validator(model, inputs):
    originals = tuple(value.clone() for value in inputs)
    expected = model(*inputs)
    def validate(timed):
        if not timed.bound:
            raise RuntimeError("Benchmark did not expose its measured invocation")
        require_unchanged(inputs, originals)
        _checked_qk_outputs(timed.outputs, expected)
        try:
            # RMSNorm/RoPE preserve this sign change without changing the
            # declared BF16 domain or the original strided KV allocation.
            inputs[0].neg_()
            inputs[1].neg_()
            changed = tuple(value.clone() for value in inputs)
            replay_expected = model(*inputs)
            for output in timed.outputs[:2]:
                output.fill_(float("nan"))
            replay_output = timed.rerun()
            require_unchanged(inputs, changed)
            _checked_qk_outputs(replay_output, replay_expected)
        finally:
            for value, original in zip(inputs, originals):
                value.copy_(original)
        return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
                "replay_inputs_perturbed": True, "replay_output_poisoned": True}
    return validate


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    # End-to-end smoke: Model(*get_init_inputs()) + get_inputs() must run.
    init = mmod.get_init_inputs()
    smoke_model = mmod.Model(*init).to("cuda").eval()
    with torch.no_grad():
        # get_inputs() returns CPU tensors (KernelBench convention); relocate.
        smoke_args = [a.to("cuda") for a in mmod.get_inputs()]
        _sq, _skv = smoke_model(*smoke_args)
    assert _sq.shape[0] == smoke_args[0].shape[0], "smoke Model forward shape mismatch"
    if verbose:
        print(f"  smoke: Model(*get_init_inputs())+get_inputs() OK "
              f"(init={init}, q_out={tuple(_sq.shape)}, kv_out={tuple(_skv.shape)})")

    failures = []
    worst = 0.0
    for shape in SHAPES:
        T, H, D, RD, G = (
            shape["T"], shape["H"], shape["D"], shape["RD"], shape["group_size"]
        )
        try:
            model = mmod.Model(H, D, RD, G).to("cuda").eval()
            q, kv, kv_weight, cos, sin, positions = _make_inputs(mmod, shape)
            protected_inputs = (q, kv, kv_weight, cos, sin, positions)
            originals = tuple(v.clone() for v in protected_inputs)

            with torch.no_grad():
                ref_q, ref_kv = model(q, kv, kv_weight, cos, sin, positions)

            def _run():
                return kmod.flydsl_qk_norm_rope_quant(
                    q, kv, kv_weight, cos, sin, positions,
                    num_q_heads=H, head_dim=D, rope_head_dim=RD,
                    quant=False,
                )

            out_q, out_kv, qs, ks = _retry(_run, what=shape["name"])
            torch.cuda.synchronize()

            require_unchanged(protected_inputs, originals)
            _checked_qk_outputs((out_q, out_kv, qs, ks), (ref_q, ref_kv))
            err_q, ma_q, _ = _norm_max_err(ref_q, out_q)
            err_kv, ma_kv, _ = _norm_max_err(ref_kv, out_kv)
            err = max(err_q, err_kv)
            worst = max(worst, err)
            pctq = torch.isclose(ref_q.float(), out_q.float(), atol=1e-2, rtol=1e-2).float().mean().item() * 100
            pctkv = torch.isclose(ref_kv.float(), out_kv.float(), atol=1e-2, rtol=1e-2).float().mean().item() * 100
            ok = err <= REL_TOL and qs is None and ks is None
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(T{T}/H{H}/D{D}/RD{RD}/g{G}) "
                    f"norm_max_err={err:.6f} (tol={REL_TOL}) "
                    f"[q={err_q:.6f} max_abs={ma_q:.5f}, kv={err_kv:.6f} max_abs={ma_kv:.5f}] "
                    f"close%@1e-2 q={pctq:.2f} kv={pctkv:.2f}"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"worst normalized max error across all shapes: {worst:.6f} (tol={REL_TOL})")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    latencies, speedups, report = [], [], []
    print(f"{'Config':<24} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        T, H, D, RD, G = (
            shape["T"], shape["H"], shape["D"], shape["RD"], shape["group_size"]
        )
        model = mmod.Model(H, D, RD, G).to("cuda").eval()
        q, kv, kv_weight, cos, sin, positions = _make_inputs(mmod, shape)

        replay_validate = _qk_replay_validator(model, (q, kv, kv_weight, cos, sin, positions))

        def run_kernel():
            return kmod.flydsl_qk_norm_rope_quant(
                q, kv, kv_weight, cos, sin, positions,
                num_q_heads=H, head_dim=D, rope_head_dim=RD, quant=False,
            )

        _retry(run_kernel, what=shape["name"])
        torch.cuda.synchronize()
        for _ in range(warmup):
            run_kernel()
        torch.cuda.synchronize()

        timed = TimedRun()
        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            run_kernel, warmup=0, repetition=iters, timed_run=timed
        )

        kernel_bench_meta.update(replay_validate(timed))

        with torch.no_grad():
            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                lambda: model(q, kv, kv_weight, cos, sin, positions),
                warmup=warmup,
                repetition=iters,
            )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        # bytes moved: Q in/out + KV in/out + kv_weight (bf16).
        bytes_total = (T * H * D * 2 * 2) + (T * D * 2 * 2) + (D * 2)
        gbps = bytes_total / (kernel_ms * 1e-3) / 1e9
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [T, H, D, RD],
            "params": {"T": T, "H": H, "D": D, "RD": RD, "group_size": G, "dtype": "bf16"},
            "gbps": gbps,
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del model, q, kv, kv_weight, cos, sin, positions
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 60)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl qk_norm_rope_quant harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl QK-RMSNorm + GPT-J RoPE (bf16)")
    print("=" * 60)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)


def _require_candidate_outputs(mod):
    import functools
    for name in tuple(vars(mod)):
        target = getattr(mod, name)
        if name.startswith("flydsl_") and callable(target):
            @functools.wraps(target)
            def checked(*args, __target=target, **kwargs):
                try:
                    result = __target(*args, **kwargs)
                except NotImplementedError as exc:
                    raise RuntimeError("Executed candidate is unimplemented; no baseline fallback") from exc
                if result is None:
                    raise RuntimeError("Candidate operator returned None; output is required")
                return result
            setattr(mod, name, checked)


# V2 direct timing evidence from this invocation.
def arena_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    latencies, speedups, report = [], [], []
    print(f"{'Config':<24} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        T, H, D, RD, G = (
            shape["T"], shape["H"], shape["D"], shape["RD"], shape["group_size"]
        )
        model = mmod.Model(H, D, RD, G).to("cuda").eval()
        q, kv, kv_weight, cos, sin, positions = _make_inputs(mmod, shape)

        replay_validate = _qk_replay_validator(model, (q, kv, kv_weight, cos, sin, positions))

        def run_kernel():
            return kmod.flydsl_qk_norm_rope_quant(
                q, kv, kv_weight, cos, sin, positions,
                num_q_heads=H, head_dim=D, rope_head_dim=RD, quant=False,
            )

        _retry(run_kernel, what=shape["name"])
        torch.cuda.synchronize()
        for _ in range(warmup):
            run_kernel()
        torch.cuda.synchronize()

        timed = TimedRun()
        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            run_kernel, warmup=0, repetition=iters, timed_run=timed
        )

        kernel_bench_meta.update(replay_validate(timed))

        with torch.no_grad():
            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                lambda: model(q, kv, kv_weight, cos, sin, positions),
                warmup=warmup,
                repetition=iters,
            )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        # bytes moved: Q in/out + KV in/out + kv_weight (bf16).
        bytes_total = (T * H * D * 2 * 2) + (T * D * 2 * 2) + (D * 2)
        gbps = bytes_total / (kernel_ms * 1e-3) / 1e9
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [T, H, D, RD],
            "params": {"T": T, "H": H, "D": D, "RD": RD, "group_size": G, "dtype": "bf16"},
            "gbps": gbps,
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del model, q, kv, kv_weight, cos, sin, positions
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 60)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return report
