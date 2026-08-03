#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/moe_fused_gemm (FLAT layout).

The kernel under test is AITER's (vLLM-derived) fused Mixture-of-Experts GEMM
(`fused_moe` / `_fused_moe_kernel`). Each token is routed to `top_k` experts;
the kernel multiplies the token's activations by each selected expert's weight
matrix and (optionally) scales by the routing weight, writing C[M, top_k, N].

The token->expert sort + block padding is produced by the in-source
`moe_align_block_size` (AITER's 4-stage Triton counting-sort).

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output
                    (flydsl-vs-triton comparison added when the FlyDSL target lands)
  --full-benchmark  graph-first GPU timing, write build/performance_report.json
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

SOURCE_FILE = "moe_fused_gemm.py"
ENTRY = "fused_moe"
KERNEL = "_fused_moe_kernel"

# (num_tokens M, hidden K, expert-out N, num_experts E, top_k)
TEST_SHAPES = [
    {"name": "m64_k1024_n512_e8_top2", "M": 64, "K": 1024, "N": 512, "E": 8, "top_k": 2},
    {"name": "m128_k2048_n1024_e8_top2", "M": 128, "K": 2048, "N": 1024, "E": 8, "top_k": 2},
    {"name": "m256_k4096_n512_e16_top1", "M": 256, "K": 4096, "N": 512, "E": 16, "top_k": 1},
    {"name": "m512_k1024_n768_e16_top2", "M": 512, "K": 1024, "N": 768, "E": 16, "top_k": 2},
    {"name": "m1024_k2048_n256_e32_top4", "M": 1024, "K": 2048, "N": 256, "E": 32, "top_k": 4},
]

BLOCK_SIZE_M = 64
SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("moe_fused_gemm_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, K, N, E, top_k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    A = (torch.randn((M, K), generator=gen, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    B = (torch.randn((E, N, K), generator=gen, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    # Per-token top_k DISTINCT experts.
    scores = torch.rand((M, E), generator=gen, device=device)
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_weights = torch.rand(
        (M, top_k), generator=gen, device=device, dtype=torch.float32
    ).contiguous()
    return A, B, topk_ids, topk_weights


def _run_kernel(mod, A, B, topk_ids, topk_weights, top_k, mul_routed_weight):
    import torch
    import triton.language as tl

    M, K = A.shape
    E, N, _ = B.shape
    sorted_ids, expert_ids, num_post = mod.moe_align_block_size(topk_ids, BLOCK_SIZE_M, E)
    C = torch.zeros((M, top_k, N), device=A.device, dtype=B.dtype)
    config = {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    mod.fused_moe(
        A,
        B,
        C,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        mul_routed_weight,
        top_k,
        tl.bfloat16,
        config=config,
    )
    return C


# --- numerical reference -----------------------------------------------------
# Ported from aiter/aiter/fused_moe.py::torch_moe (the per-expert GEMM portion).
# torch_moe runs a full 2-layer MoE (w1 gate/up -> activation -> w2 down) and
# then combines the top_k slots via `(out * topk_weight).sum(dim=1)`. This
# harness only exercises the fused GEMM (a single [E,N,K] projection, no
# activation, no gate/up, no top_k reduction), and keeps the per-slot layout
# C[M, top_k, N]. So we port ONLY the matching sub-computation:
#     for each expert E_id: out[mask] = sub_tokens @ w[E_id].T
# with fp32 compute (torch_moe's `computeType`), plus the optional routing-weight
# multiply the kernel applies (MUL_ROUTED_WEIGHT), but WITHOUT the top_k sum.
# The kernel accumulates the bf16 A@B in fp32 then casts to bf16, so we mirror
# that: fp32 matmul of the exact bf16 values, per-slot.
def _ref_moe_gemm(A, B, topk_ids, topk_weights, top_k, mul_routed_weight):
    import torch

    M, K = A.shape
    E, N, _ = B.shape
    compute = torch.float32
    a = A.to(compute)
    b = B.to(compute)
    # hidden_states repeated top_k times -> [M, top_k, K] (mirrors torch_moe view/repeat)
    hidden = a.view(M, 1, K).repeat(1, top_k, 1)
    out = torch.zeros((M, top_k, N), dtype=compute, device=A.device)
    for E_id in range(E):
        mask = topk_ids.long() == E_id  # [M, top_k]
        if mask.any():
            sub_tokens = hidden[mask]                 # [n_sel, K]
            out[mask] = sub_tokens @ b[E_id].transpose(0, 1)  # [n_sel, N]
    if mul_routed_weight:
        out = out * topk_weights.to(compute).view(M, top_k, 1)
    return out


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    assert hasattr(mod, "moe_align_block_size"), "Missing moe_align_block_size"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    # Runs the Triton kernel on TEST_SHAPES and asserts finite output. No torch
    # comparison: the flydsl-vs-triton comparison is added when the FlyDSL target
    # lands (the Triton kernel is the reference here).
    import torch

    mod = _load_source()
    # bf16 inputs with fp32 accumulation: the only error vs the fp32 reference is
    # bf16 rounding of the accumulator on write-back (~2^-8) plus MFMA accumulation
    # order, so we gate on a tight normalized-max-error of 1e-2.
    NME_TOL = 1e-2
    failures = []
    for shape in TEST_SHAPES:
        for mul in (True, False):
            tag = f"{shape['name']}{'_w' if mul else ''}"
            try:
                A, B, topk_ids, topk_weights = _make_inputs(
                    shape["M"], shape["K"], shape["N"], shape["E"], shape["top_k"]
                )
                C = _run_kernel(mod, A, B, topk_ids, topk_weights, shape["top_k"], mul)
                torch.cuda.synchronize()
                finite = bool(torch.isfinite(C).all().item())

                ref = _ref_moe_gemm(A, B, topk_ids, topk_weights, shape["top_k"], mul)
                a = C.float()
                b = ref.float()
                abs_err = (a - b).abs()
                denom = b.abs().max().item()
                nme = float((abs_err.max() / denom).item()) if denom > 0 else float(abs_err.max().item())
                allclose = bool(torch.allclose(a, b, atol=1e-2, rtol=1e-2))
                numeric_ok = nme <= NME_TOL
                ok = finite and numeric_ok
                if verbose:
                    print(
                        f"  {'PASS' if ok else 'FAIL'}: {tag} "
                        f"(M={shape['M']},K={shape['K']},N={shape['N']},E={shape['E']},"
                        f"top_k={shape['top_k']}) out={tuple(C.shape)} finite={finite} "
                        f"nme={nme:.3e} allclose={allclose}"
                    )
                if not ok:
                    failures.append(tag)
            except Exception as e:  # noqa: BLE001
                failures.append(tag)
                if verbose:
                    print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES) * 2
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        A, B, topk_ids, topk_weights = _make_inputs(
            shape["M"], shape["K"], shape["N"], shape["E"], shape["top_k"]
        )
        fn = lambda: _run_kernel(  # noqa: E731
            mod, A, B, topk_ids, topk_weights, shape["top_k"], True
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
        flops = 2.0 * shape["M"] * shape["top_k"] * shape["N"] * shape["K"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "K", "N", "E", "top_k")},
                "tflops": flops / (ms * 1e-3) / 1e12,
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
    parser = argparse.ArgumentParser(description="moe_fused_gemm harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fused MoE GEMM")
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
