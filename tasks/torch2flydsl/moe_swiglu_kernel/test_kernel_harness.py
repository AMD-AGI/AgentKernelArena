#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and performance harness for the a4w4 SWIGLU fused MoE task.

The pure-torch reference in ``model.py`` and the FlyDSL kernel share the same
top-k routing (``model.route_topk``) so expert selection is identical. The
correctness gate is the normalized max error ``max|ref - out| / max|ref|``, which
must stay <= ``REL_TOL``; element-wise close% at 1e-2 and 1e-1 is also reported.
The check asserts and exits non-zero on failure.

Modes:
  --correctness     compare the kernel against the reference
  --full-benchmark  time the kernel vs the reference and write a perf report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, KERNEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, KERNEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
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
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# Real a4w4 fp4 SWIGLU fused-MoE shapes (q_dtype_a/w = float4_e2m1fn_x2,
# per_1x32, act_type = ActivationType.Swiglu) from gptoss_fp4_untuned_fmoe.csv
# (GPT-OSS): D=3072, E=128, topk=4, inter_dim in {512, 1536}.
SHAPES = [
    {"name": "gptoss_t16_i512_e128_k4", "tokens": 16, "model_dim": 3072, "inter_dim": 512, "experts": 128, "topk": 4},
    {"name": "gptoss_t32_i1536_e128_k4", "tokens": 32, "model_dim": 3072, "inter_dim": 1536, "experts": 128, "topk": 4},
]

# Tight element-wise gate: normalized max error <= REL_TOL.
REL_TOL = 1e-2
SEED = 20260401
BLOCK_M, TILE_N, TILE_K, MODE = 32, 256, 256, "atomic"
# Correctness uses the deterministic "reduce" combine. The "atomic" stage-2
# combine sums per-expert partials with order-dependent fp32 atomic-adds, so its
# result is nondeterministic run-to-run; "reduce" computes the identical math
# with a deterministic reduction (same numerics, reproducible comparison).
CORRECTNESS_MODE = "reduce"


def _build_model(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = mmod.Model(
        model_dim=shape["model_dim"], inter_dim=shape["inter_dim"],
        experts=shape["experts"], topk=shape["topk"],
    ).to(device).eval()
    hidden = torch.randn(
        shape["tokens"], shape["model_dim"], dtype=torch.bfloat16, device=device
    )
    return model, hidden


def _kernel_out(kmod, mmod, model, hidden, topk):
    # Recompute the SAME routing the reference used and run the FlyDSL kernel.
    logits = model.gate(hidden)
    topk_weights, topk_ids = mmod.route_topk(logits, topk)
    return kmod.flydsl_moe_swiglu(
        hidden, model.w1.detach(), model.w2.detach(), topk_weights, topk_ids,
        block_m=BLOCK_M, tile_n=TILE_N, tile_k=TILE_K, mode=CORRECTNESS_MODE,
    )


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    failures = []
    for shape in SHAPES:
        model, hidden = _build_model(mmod, shape)
        with torch.no_grad():
            ref = model(hidden).float()
            out = _kernel_out(kmod, mmod, model, hidden, shape["topk"]).float()
        torch.cuda.synchronize()

        max_abs = (ref - out).abs().max().item()
        ref_scale = ref.abs().max().item() + 1e-9
        rel_err = max_abs / ref_scale
        max_rel = ((ref - out).abs() / (ref.abs() + 1e-9)).max().item()
        pct1e2 = torch.isclose(ref, out, atol=1e-2, rtol=1e-2).float().mean().item() * 100
        pct1e1 = torch.isclose(ref, out, atol=1e-1, rtol=1e-1).float().mean().item() * 100
        ok = rel_err <= REL_TOL
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"(D{shape['model_dim']}/I{shape['inter_dim']}/E{shape['experts']}/k{shape['topk']}) "
                f"norm_max_err={rel_err:.5f} (tol={REL_TOL}) "
                f"max_abs={max_abs:.4f} max_rel={max_rel:.3f} "
                f"close%@1e-2={pct1e2:.2f} @1e-1={pct1e1:.2f}"
            )
        if not ok:
            failures.append(shape["name"])

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
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
        model, hidden = _build_model(mmod, shape)
        topk = shape["topk"]
        with torch.no_grad():
            logits = model.gate(hidden)
            topk_weights, topk_ids = mmod.route_topk(logits, topk)
            w1, w2 = model.w1.detach(), model.w2.detach()

            def run_kernel():
                return kmod.flydsl_moe_swiglu(
                    hidden, w1, w2, topk_weights, topk_ids,
                    block_m=BLOCK_M, tile_n=TILE_N, tile_k=TILE_K, mode=MODE,
                )

            run_kernel()
            torch.cuda.synchronize()
            for _ in range(warmup):
                run_kernel()
            torch.cuda.synchronize()
            ktimes = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); run_kernel(); e.record(); torch.cuda.synchronize()
                ktimes.append(s.elapsed_time(e))
            kernel_ms = sum(ktimes) / len(ktimes)

            rtimes = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); model(hidden); e.record(); torch.cuda.synchronize()
                rtimes.append(s.elapsed_time(e))
            ref_ms = sum(rtimes) / len(rtimes)

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms); speedups.append(speedup)
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [shape["tokens"], shape["model_dim"], shape["inter_dim"]],
            "params": {k: shape[k] for k in ("tokens", "model_dim", "inter_dim", "experts", "topk")},
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x")
        del model, hidden
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 60)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup:.2f}x")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    try:
        import torch as _t
        _arch = _t.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        _arch = ""
    if _arch != "gfx950":
        print(f"SKIPPED: gfx950-only task on arch={_arch or 'unknown'} (FP4/MX scaled-MFMA requires CDNA4/gfx950)")
        print("correctness: skip")
        sys.exit(0)
    parser = argparse.ArgumentParser(description="torch2flydsl moe harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl MoE (a4w4 swiglu, quantized reference)")
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
