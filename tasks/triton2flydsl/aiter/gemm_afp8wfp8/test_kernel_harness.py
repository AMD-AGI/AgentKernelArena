#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_afp8wfp8 (FLAT layout, gfx950-only).

The kernel under test is AITER's MXFP8-activation x FP8-weight scaled GEMM Triton
kernel (`gemm_afp8wfp8` -> `_gemm_afp8wfp8_kernel`): Y = X @ W^T with 1x32 e8m0
activation scales and 128x128 e8m0 weight block scales, fp32 accumulation via
`tl.dot_scaled` (format "e4m3"), bf16/fp16 output. The standalone source keeps the
non-split-K (NUM_KSPLIT == 1) triton path with a static tile config.

ARCH NOTE: `tl.dot_scaled` (microscale MX matmul) lowers to a scaled-MFMA
instruction available only on CDNA4 (gfx950); on CDNA3 (gfx942) it fails to compile
("Unsupported DotScaleOp"). This task is therefore tagged supported_archs: [gfx950]
and the arch-guard below SKIPs (exit 0) on any non-gfx950 arch.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 dequant torch reference at the upstream MXFP8
                    tolerance (atol=0.03, rtol=1e-2)
                    [mirrors gemm/basic/test_gemm_afp8wfp8.py:run_torch_gemm_afp8wfp8]
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

SOURCE_FILE = "gemm_afp8wfp8.py"
ENTRY = "gemm_afp8wfp8"
KERNEL = "_gemm_afp8wfp8_kernel"

SCALE_GROUP_SIZE = 32  # A: 1x32 e8m0 scale group
W_SCALE_K_GROUP = 128  # B: 128 in K direction
W_SCALE_N_GROUP = 128  # B: 128 in N direction
FP8_MAX = 448.0  # e4m3fn (gfx950) max

# (M, N, K) with N % 128 == 0 and K % 128 == 0 (128x128 W-scale layout). Real
# fp8 GEMM tiles from op_tests/.../test_gemm_afp8wfp8.py:get_shapes, bounded subset.
TEST_SHAPES = [
    {"name": "m16_n1536_k4096", "M": 16, "N": 1536, "K": 4096},
    {"name": "m32_n4096_k1024", "M": 32, "N": 4096, "K": 1024},
    {"name": "m64_n512_k4096", "M": 64, "N": 512, "K": 4096},
    {"name": "m128_n8192_k1024", "M": 128, "N": 8192, "K": 1024},
    {"name": "m128_n2048_k7168", "M": 128, "N": 2048, "K": 7168},
]

SEED = 0  # match upstream test (torch.manual_seed(0))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("gemm_afp8wfp8_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e8m0_to_f32(x):
    import torch

    return torch.exp2((x.to(torch.int32) - 127).to(torch.float32))


def _make_inputs(M, N, K, device="cuda"):
    # Mirrors test_gemm_afp8wfp8.py:generate_inputs (shuffle=False).
    import torch

    torch.manual_seed(SEED)
    x_f32 = torch.clamp(
        torch.randn((M, K), dtype=torch.float32, device=device), -FP8_MAX, FP8_MAX
    )
    w_f32 = torch.clamp(
        torch.randn((N, K), dtype=torch.float32, device=device), -FP8_MAX, FP8_MAX
    )
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)
    x_scales = torch.randint(
        125, 130, (M, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        125,
        130,
        (N // W_SCALE_N_GROUP, K // W_SCALE_K_GROUP),
        dtype=torch.uint8,
        device=device,
    )
    return x_fp8, w_fp8, x_scales, w_scales


def _torch_ref(x_fp8, w_fp8, x_scales, w_scales, out_dtype):
    # fp32 dequant reference (matches test_gemm_afp8wfp8.py:run_torch_gemm_afp8wfp8).
    import torch

    M, K = x_fp8.shape
    N, _ = w_fp8.shape
    x_f32 = x_fp8.to(torch.float32)
    w_f32 = w_fp8.to(torch.float32)
    x_s = _e8m0_to_f32(x_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_s = _e8m0_to_f32(w_scales)
    w_s = w_s.repeat_interleave(W_SCALE_N_GROUP, dim=0).repeat_interleave(
        W_SCALE_K_GROUP, dim=1
    )
    x_dq = x_f32 * x_s
    w_dq = w_f32 * w_s
    return torch.mm(x_dq, w_dq.T).to(out_dtype)


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
    out_dtype = torch.bfloat16
    failures = []
    for shape in TEST_SHAPES:
        tag = shape["name"]
        try:
            x, w, x_scales, w_scales = _make_inputs(shape["M"], shape["N"], shape["K"])
            y = mod.gemm_afp8wfp8(x, w, x_scales, w_scales, dtype=out_dtype)
            torch.cuda.synchronize()
            ref = _torch_ref(x, w, x_scales, w_scales, out_dtype)
            finite = bool(torch.isfinite(y).all().item())
            close = torch.allclose(y, ref, atol=0.03, rtol=1e-2)
            ok = finite and close
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {tag} "
                    f"(M={shape['M']},N={shape['N']},K={shape['K']}) "
                    f"out={tuple(y.shape)} finite={finite} close={close}"
                )
            if not ok:
                failures.append(tag)
        except Exception as e:  # noqa: BLE001
            failures.append(tag)
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    out_dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, x_scales, w_scales = _make_inputs(shape["M"], shape["N"], shape["K"])
        fn = lambda: mod.gemm_afp8wfp8(x, w, x_scales, w_scales, dtype=out_dtype)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        flops = 2.0 * shape["M"] * shape["N"] * shape["K"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "N", "K")},
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
    parser = argparse.ArgumentParser(description="gemm_afp8wfp8 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl MXFP8 x FP8 scaled GEMM (gfx950-only)")
    print("=" * 62)

    # Arch-guard: tl.dot_scaled (MX scaled-MFMA) is gfx950 (CDNA4) only; it fails
    # to compile on gfx942 ("Unsupported DotScaleOp"). SKIP (exit 0) elsewhere.
    try:
        import torch as _t

        _arch = _t.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        _arch = ""
    if _arch != "gfx950":
        print(
            f"SKIPPED: gfx950-only task on arch={_arch or 'unknown'} "
            "(MXFP8 tl.dot_scaled requires CDNA4/gfx950)"
        )
        print("correctness: skip")
        sys.exit(0)

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
