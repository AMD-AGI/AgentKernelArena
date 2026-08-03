#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/moe_routing_sigmoid_top1 (FLAT layout).

The kernel under test is AITER's fused sigmoid top-1 MoE router
(`routing_sigmoid_top1` / `_routing_sigmoid_top1_kernel`). It computes
scores = sigmoid(x @ w) over N experts and emits the top-1 expert id + weight
per token.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the triton kernel on TEST_SHAPES, assert finite output
  --full-benchmark  graph-first GPU timing, write build/performance_report.json

The flydsl-vs-triton comparison will be added when the FlyDSL target lands.
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

SOURCE_FILE = "moe_routing_sigmoid_top1.py"
ENTRY = "routing_sigmoid_top1"
KERNEL = "_routing_sigmoid_top1_kernel"

# (M, K, N_experts, fused_shared_experts). N must be a power of two.
TEST_SHAPES = [
    {"name": "m256_k2048_n16", "M": 256, "K": 2048, "N": 16, "shared": False},
    {"name": "m1024_k4096_n32", "M": 1024, "K": 4096, "N": 32, "shared": False},
    {"name": "m2048_k5120_n128", "M": 2048, "K": 5120, "N": 128, "shared": False},
    {"name": "m512_k4096_n64_shared", "M": 512, "K": 4096, "N": 64, "shared": True},
    {"name": "m4096_k1024_n128", "M": 4096, "K": 1024, "N": 128, "shared": False},
]

SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("moe_routing_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, K, N, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((M, K), generator=gen, device=device, dtype=torch.float16)
    w = torch.randn((K, N), generator=gen, device=device, dtype=torch.float16) * 0.1
    return x, w


def _torch_routing_ref(x, w, N, fused_shared):
    # Reference for routing_sigmoid_top1: scores = sigmoid(x @ w) over N experts,
    # top-1 = argmax (tie_break_left), weight = the max sigmoid score. With a
    # fused shared expert an extra column (id=N, weight=1.0) is appended.
    import torch

    scores = torch.sigmoid(x.float() @ w.float())  # [M, N], fp32
    top_w, top_id = scores.max(dim=1)  # first max index for ties
    top_id = top_id.to(torch.int32)
    if fused_shared:
        ref_ids = torch.stack([top_id, torch.full_like(top_id, N)], dim=1)
        ref_w = torch.stack([top_w, torch.ones_like(top_w)], dim=1)
    else:
        ref_ids = top_id[:, None]
        ref_w = top_w[:, None]
    return ref_ids, ref_w, scores


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    import torch

    mod = _load_source()
    failures = []
    for shape in TEST_SHAPES:
        try:
            x, w = _make_inputs(shape["M"], shape["K"], shape["N"])
            ids, weights = mod.routing_sigmoid_top1(
                x, w, topk=1, fused_shared_experts=shape["shared"]
            )
            torch.cuda.synchronize()

            ref_ids, ref_w, ref_scores = _torch_routing_ref(
                x, w, shape["N"], shape["shared"]
            )
            finite = torch.isfinite(weights.float()).all().item()
            # Numerical gate: the top-1 sigmoid weights must match the fp32
            # reference within the fp16 tolerance.
            w_close = torch.allclose(
                weights.float(), ref_w, atol=1e-2, rtol=1e-2
            )
            # Expert-id gate: accept the kernel's pick when it lands on a
            # (near-)argmax expert — i.e. the reference score at the kernel's
            # chosen id is within tol of the reference max — so a benign fp16
            # tie-break flip is not a false failure.
            col0_ids = ids[:, 0].long().clamp_max(shape["N"] - 1)
            chosen_scores = ref_scores.gather(1, col0_ids[:, None]).squeeze(1)
            ref_max = ref_scores.max(dim=1).values
            id_ok = bool((ref_max - chosen_scores <= 1e-2).all().item())
            if shape["shared"]:
                # Shared-expert column must be exactly (id=N, weight=1.0).
                id_ok = id_ok and bool((ids[:, 1] == shape["N"]).all().item())
                w_close = w_close and torch.allclose(
                    weights[:, 1].float(),
                    torch.ones_like(ref_w[:, 1]),
                    atol=1e-3,
                    rtol=1e-3,
                )
            ok = finite and w_close and id_ok
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(M={shape['M']},K={shape['K']},N={shape['N']},shared={shape['shared']}) "
                    f"finite={finite} weights_close={w_close} id_ok={id_ok}"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(TEST_SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w = _make_inputs(shape["M"], shape["K"], shape["N"])
        fn = lambda: mod.routing_sigmoid_top1(  # noqa: E731
            x, w, topk=1, fused_shared_experts=shape["shared"]
        )
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "K", "N", "shared")},
            }
        )
        if verbose:
            print(f"  {shape['name']}: {ms:.4f} ms")

    build_dir = Path(_HERE) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    geomean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    print(f"Geometric mean latency: {geomean:.4f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="moe_routing_sigmoid_top1 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl MoE sigmoid top-1 routing")
    print("=" * 62)

    if args.compile:
        try:
            run_compile()
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"Compilation: FAIL\nError: {e}")
            sys.exit(1)
    elif args.correctness:
        sys.exit(0 if run_correctness() else 1)
    else:
        run_benchmark()
        sys.exit(0)
