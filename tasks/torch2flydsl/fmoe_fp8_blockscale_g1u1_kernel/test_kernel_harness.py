#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl fmoe_fp8_blockscale_g1u1 task.

The op is an FP8 (float8_e4m3fn) two-stage g1u1 fused MoE with 128x128 block
scales: per-token per-1x128 activation scales and per-128x128 weight scales,
silu-gated stage 1, grouped down GEMM + top-k combine, accumulated in fp32 and
returned in bf16.

Correctness compares the (b)-faithful PyTorch reference in model.py against the
real AMD runtime op (``aiter.fmoe_fp8_blockscale_g1u1``) over byte-identical FP8
operands produced by ``model.quantize_blockscale_moe`` (the harness only adds the
host-side weight shuffle + moe_sorting dispatch the kernel needs). The gate is
the normalized worst-element error ``max|ref-gt| / max|ref| <= TOL``. When the
FlyDSL kernel.py exists it is additionally validated against the reference.

Modes:
  --correctness     compare the model.py reference to the aiter ground truth
  --full-benchmark  time the FlyDSL kernel (or the aiter op when no kernel.py),
                    write build/performance_report.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"
KERNEL_ENTRY = "flydsl_fmoe_fp8_blockscale_g1u1"


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

# Real FP8 block-scaled fused-MoE shapes (q_dtype_a/w = float8_e4m3fn,
# QuantType per-128x128, g1u1, silu) from
# configs/model_configs/a8w8_blockscale_untuned_fmoe_{ds_v3,glm5}.csv. model_dim
# and inter_dim*2 are multiples of 128 so the 128x128 blocks tile exactly.
SHAPES = [
    {"name": "dsv3_t16_e257_k9", "tokens": 16, "model_dim": 7168, "inter_dim": 256, "experts": 257, "topk": 9},
    {"name": "dsv3_t32_e257_k9", "tokens": 32, "model_dim": 7168, "inter_dim": 256, "experts": 257, "topk": 9},
    {"name": "glm5_t32_e257_k9", "tokens": 32, "model_dim": 6144, "inter_dim": 256, "experts": 257, "topk": 9},
]

# Quantized fused-MoE gate: normalized worst-element error vs the aiter op.
TOL = 1e-2
SEED = 20260401
BLOCK_N, BLOCK_K = 128, 128


def _retry(fn, tries=5, what="kernel call"):
    """Retry on transient out-of-memory / HIP errors (shared-GPU friendly)."""
    import torch

    last = None
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as exc:  # noqa: PERF203
            msg = str(exc).lower()
            if "out of memory" not in msg and "hip" not in msg:
                raise
            last = exc
            torch.cuda.empty_cache()
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{what} failed after {tries} retries: {last}")


def _build_model(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = (
        mmod.Model(
            model_dim=shape["model_dim"], inter_dim=shape["inter_dim"],
            experts=shape["experts"], topk=shape["topk"],
        )
        .to(device)
        .eval()
    )
    hidden = torch.randn(
        shape["tokens"], shape["model_dim"], dtype=torch.bfloat16, device=device
    )
    return model, hidden


def _make_prepared_aiter_op(
    mmod, model, hidden, topk_weights, topk_ids, topk
):
    """Prepare routing/quantization/sorting and return the AITER kernel call."""
    import torch
    import aiter
    from aiter.fused_moe import moe_sorting
    from aiter.ops.shuffle import shuffle_weight

    E = model.w1.shape[0]
    model_dim = hidden.shape[-1]
    a1_q, a1_scale, w1_q, w1_scale, w2_q, w2_scale = mmod.quantize_blockscale_moe(
        hidden, model.w1.detach(), model.w2.detach()
    )
    a1_scale = a1_scale.t().contiguous()
    w1_scale = w1_scale.view(E, -1).contiguous()
    w2_scale = w2_scale.view(E, -1).contiguous()
    w1_shuffled = shuffle_weight(w1_q, (16, 16))
    w2_shuffled = shuffle_weight(w2_q, (16, 16))

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out_asm = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, torch.bfloat16)
    )

    def device_op():
        aiter.fmoe_fp8_blockscale_g1u1(
            out_asm,
            a1_q,
            w1_shuffled,
            w2_shuffled,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            "",
            BLOCK_N,
            BLOCK_K,
            None,
        )
        return out_asm

    return device_op


def _aiter_op(mmod, model, hidden, topk):
    """Drive the real aiter fmoe_fp8_blockscale_g1u1 over the model's FP8 operands."""
    logits = model.gate(hidden)
    topk_weights, topk_ids = mmod.route_topk(logits, topk)
    return _make_prepared_aiter_op(
        mmod, model, hidden, topk_weights, topk_ids, topk
    )()


def _norm_worst(ref, out):
    rf, of = ref.float(), out.float()
    worst = (rf - of).abs().max().item()
    denom = rf.abs().max().item()
    denom = denom if denom > 0 else 1.0
    return worst, worst / denom


def run_correctness(verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if mmod is None:
        print("FAIL: cannot load model.py")
        assert False, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    failures = []
    for shape in SHAPES:
        try:
            model, hidden = _build_model(mmod, shape)
            with torch.no_grad():
                ref = model(hidden)
                gt = _retry(
                    lambda: _aiter_op(mmod, model, hidden, shape["topk"]),
                    what="aiter fmoe_fp8_blockscale_g1u1",
                )
            torch.cuda.synchronize()

            worst, norm = _norm_worst(ref, gt)
            ok = norm <= TOL
            note = ""
            if has_kernel:
                logits = model.gate(hidden)
                topk_weights, topk_ids = mmod.route_topk(logits, shape["topk"])
                try:
                    out = _retry(
                        lambda: kmod.flydsl_fmoe_fp8_blockscale_g1u1(
                            hidden, model.w1.detach(), model.w2.detach(),
                            topk_weights, topk_ids,
                        ),
                        what="flydsl kernel",
                    )
                except NotImplementedError:
                    has_kernel = False
                    print(
                        "  SKIP: kernel.py FlyDSL target not implemented yet "
                        "(reference validated against the aiter op above)"
                    )
                else:
                    torch.cuda.synchronize()
                    _, knorm = _norm_worst(ref, out)
                    kok = knorm <= TOL
                    ok = ok and kok
                    note = f" | kernel norm={knorm:.4g} {'ok' if kok else 'BAD'}"

            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(D{shape['model_dim']}/I{shape['inter_dim']}/"
                    f"E{shape['experts']}/k{shape['topk']}) "
                    f"worst={worst:.4g} norm={norm:.4g} tol={TOL}{note}"
                )
            if not ok:
                failures.append(shape["name"])
            del model, hidden, gt
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:200]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    if has_kernel:
        s0 = SHAPES[0]
        model0, hidden0 = _build_model(mmod, s0)
        with torch.no_grad():
            logits0 = model0.gate(hidden0)
            topk_weights0, topk_ids0 = mmod.route_topk(logits0, s0["topk"])
            try:
                kmod.flydsl_fmoe_fp8_blockscale_g1u1(
                    hidden0, model0.w1.detach(), model0.w2.detach(),
                    topk_weights0, topk_ids0,
                )
            except NotImplementedError:
                has_kernel = False
                print(
                    "SKIP: kernel.py FlyDSL target not implemented yet "
                    "(benchmarking aiter op instead)"
                )
        del model0, hidden0
        torch.cuda.empty_cache()

    label = "FlyDSL" if has_kernel else "aiter"
    latencies, speedups, report = [], [], []
    print(f"{'Config':<24} {'Ref':>10} {label:>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        model, hidden = _build_model(mmod, shape)
        topk = shape["topk"]
        with torch.no_grad():
            logits = model.gate(hidden)
            topk_weights, topk_ids = mmod.route_topk(logits, topk)
            expert_plan = mmod.prepare_expert_plan(topk_ids, model.experts)

            if has_kernel:
                def device_op():
                    return kmod.flydsl_fmoe_fp8_blockscale_g1u1(
                        hidden, model.w1.detach(), model.w2.detach(),
                        topk_weights, topk_ids,
                    )
            else:
                device_op = _make_prepared_aiter_op(
                    mmod, model, hidden, topk_weights, topk_ids, topk
                )

            _retry(device_op, what="benchmark warmup")
            torch.cuda.synchronize()
            for _ in range(warmup):
                device_op()
            torch.cuda.synchronize()
            kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
                device_op, warmup=0, repetition=iters
            )

            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                lambda: model.forward_with_routing(
                    hidden, topk_weights, topk_ids, expert_plan
                ),
                warmup=0,
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
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [shape["tokens"], shape["model_dim"], shape["inter_dim"]],
            "params": {k: shape[k] for k in ("tokens", "model_dim", "inter_dim", "experts", "topk")},
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del model, hidden
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
    parser = argparse.ArgumentParser(description="torch2flydsl fmoe_fp8_blockscale_g1u1 harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl fused MoE (fp8 a8w8 128x128 blockscale, g1u1)")
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
