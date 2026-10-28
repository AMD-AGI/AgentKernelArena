#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_a8w8 (FLAT layout).

The kernel under test is AITER's 8-bit scaled GEMM Triton kernel (`gemm_a8w8` ->
`_gemm_a8w8_kernel`): Y = (X @ W^T) * (x_scale * w_scale) (+ bias) with int32/fp32
accumulation, per-row x_scale [M,1] and per-column w_scale [1,N], an XCD-balanced
+ grouped pid remap, and bf16 output. The standalone source keeps the non-split-K
(NUM_KSPLIT == 1) triton path with a static tile config.

fp8 dtype is arch-specific: gfx942 = e4m3fnuz (max ~240), gfx950 = e4m3fn
(max 448). The harness quantizes inputs (and builds its reference) with the
arch-matched fp8 e4m3 dtype, mirroring aiter.ops.triton.utils.types.get_fp8_dtypes
so the torch reference is apples-to-apples with the kernel at the tight gate.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 dequant torch reference at the upstream fp8
                    tolerance (atol=0.02, rtol=1e-2)
                    [mirrors gemm/basic/test_gemm_a8w8.py:run_torch]
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
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_unchanged
from scripts.sample_controls import MeasuredScaledGemmInputs

from task_runtime import candidate_relative_path
SOURCE_FILE = candidate_relative_path()
ENTRY = "gemm_a8w8"
KERNEL = "_gemm_a8w8_kernel"

# (M, N, K) from op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py::get_x_vals /
# get_fewer_x_vals (real fp8 GEMMs: Llama-3 70B QKV input projection, square
# decode/prefill tiles) plus minimal/irregular edge cases.
TEST_SHAPES = [
    {"name": "m1_n1_k1", "M": 1, "N": 1, "K": 1},
    {"name": "m3_n5_k2", "M": 3, "N": 5, "K": 2},
    {"name": "m16_n1024_k1024", "M": 16, "N": 1024, "K": 1024},
    {"name": "m128_n8192_k512", "M": 128, "N": 8192, "K": 512},
    {"name": "m256_n512_k8192", "M": 256, "N": 512, "K": 8192},
    {"name": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "m2048_n2048_k2048", "M": 2048, "N": 2048, "K": 2048},
    {"name": "m4096_n4096_k4096", "M": 4096, "N": 4096, "K": 4096},
    {"name": "ll3_70b_qkv_m256_n10240_k8192", "M": 256, "N": 10240, "K": 8192},
]

SEED = 0  # match upstream generate_gemm_a8w8_inputs (torch.manual_seed(0))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _fp8_e4m3_dtype():
    import torch

    name = torch.cuda.get_device_properties(0).gcnArchName
    arch = name.split(":")[0]
    if arch in ("gfx950", "gfx1250", "gfx1200", "gfx1201"):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("gemm_a8w8_src", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, K, in_dtype, with_bias, device="cuda"):
    # Mirrors generate_gemm_a8w8_inputs (layout TN): per-row x_scale, per-row
    # w_scale (transposed to per-column), quantize to in_dtype.
    import torch

    torch.manual_seed(SEED)
    dtype_max = torch.finfo(in_dtype).max

    x = torch.randn((M, K), dtype=torch.float32, device=device)
    weight = torch.randn((N, K), dtype=torch.float32, device=device)

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max
    x = (x / x_scale).to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max
    weight = (weight / w_scale.T).to(in_dtype)

    bias = (
        torch.rand([1, N], dtype=torch.float32, device=device) * 10
        if with_bias
        else None
    )
    return x, weight, x_scale, w_scale, bias


def _torch_ref(x, weight, x_scale, w_scale, bias, out_dtype):
    # fp32 dequant reference (matches test_gemm_a8w8.py:run_torch).
    import torch
    import torch.nn.functional as F

    acc = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(acc, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(out_dtype)


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    print("Compilation: PASS")
    return True


def _checked_scaled_gemm_output(out, x, w, out_dtype):
    import torch
    if (not isinstance(out, torch.Tensor) or out.shape != (x.shape[0], w.shape[0])
            or out.dtype != out_dtype or out.device != x.device):
        raise AssertionError("Scaled GEMM output shape/dtype/device contract mismatch")


def run_correctness(verbose=True):
    import torch

    mod = _load_source()
    in_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    if verbose:
        print(f"  fp8 in_dtype = {in_dtype}")
    failures = []
    for shape in TEST_SHAPES:
        for with_bias in (False, True):
            tag = f"{shape['name']}_{'bias' if with_bias else 'nobias'}"
            try:
                x, w, x_scale, w_scale, bias = _make_inputs(
                    shape["M"], shape["N"], shape["K"], in_dtype, with_bias
                )
                protected_inputs = tuple(v for v in (x, w, x_scale, w_scale, bias,) if v is not None)
                originals = tuple(v.clone() for v in protected_inputs)
                y = mod.gemm_a8w8(x, w, x_scale, w_scale, bias, out_dtype)
                require_unchanged(protected_inputs, originals)
                _checked_scaled_gemm_output(y, x, w, out_dtype)
                torch.cuda.synchronize()
                ref = _torch_ref(x, w, x_scale, w_scale, bias, out_dtype)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=0.02, rtol=1e-2)
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

    from scripts.api_controls import API_CASES, run_api_controls
    failures.extend(run_api_controls(mod, _make_inputs, _torch_ref,
                                     _checked_scaled_gemm_output, in_dtype, verbose=verbose))
    run_domain_controls(mod)

    total = len(TEST_SHAPES) * 2 + len(API_CASES)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    in_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, x_scale, w_scale, bias = _make_inputs(
            shape["M"], shape["N"], shape["K"], in_dtype, False
        )
        def reference():
            return _torch_ref(x, w, x_scale, w_scale, bias, out_dtype)
        stream = MeasuredScaledGemmInputs(
            x, w, x_scale, w_scale, bias, seed=SEED, case_index=idx,
            samples=ITERS, output_dtype=out_dtype, reference=reference,
        )
        fn = lambda: mod.gemm_a8w8(x, w, x_scale, w_scale, None, out_dtype)  # noqa: E731
        try:
            fn()
            require_unchanged(stream.inputs, stream.originals)
            torch.cuda.synchronize()
            for _ in range(WARMUP):
                fn()
                require_unchanged(stream.inputs, stream.originals)
            torch.cuda.synchronize()
            timed = TimedRun()
            timed.before_sample = stream.before_sample
            timed.after_sample = stream.observe
            ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=ITERS, timed_run=timed,
                prepare_fn=stream.prepare,
            )
            stream.finish()
            bench_meta.update(stream.validate_replays(timed))
        finally:
            stream.restore()
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
    parser = argparse.ArgumentParser(description="gemm_a8w8 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl 8-bit (fp8) scaled GEMM")
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


def run_domain_controls(mod):
    """Unscored signed INT8 calls keep the FP8 scored workload unchanged."""
    import torch
    for m, n, k in ((3, 5, 33), (16, 32, 64)):
        x = (torch.arange(m*k, device="cuda").reshape(m, k) % 17 - 8).to(torch.int8)
        w = (torch.arange(n*k, device="cuda").reshape(n, k) % 13 - 6).to(torch.int8)
        xs = torch.linspace(.01, .02, m, device=x.device).reshape(m, 1)
        ws = torch.linspace(.02, .03, n, device=x.device).reshape(1, n)
        bias = torch.linspace(-.25, .25, n, device=x.device).reshape(1, n)
        inputs = (x, w, xs, ws, bias)
        originals = tuple(value.clone() for value in inputs)
        expected = _torch_ref(*inputs, torch.bfloat16)
        actual = mod.gemm_a8w8(*inputs, torch.bfloat16)
        require_unchanged(inputs, originals)
        _checked_scaled_gemm_output(actual, x, w, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.01)
    for in_dtype in (torch.int8, _fp8_e4m3_dtype()):
        for out_dtype in (torch.bfloat16, torch.float16):
            m, n, k = 16, 32, 64
            x_store = torch.empty((m, k * 2), device='cuda', dtype=in_dtype)
            w_store = torch.empty((n * 2, k), device='cuda', dtype=in_dtype)
            x, w = x_store[:, ::2], w_store[::2, :]
            x.copy_((torch.arange(m*k, device='cuda').reshape(m, k) % 17 - 8).to(in_dtype))
            w.copy_((torch.arange(n*k, device='cuda').reshape(n, k) % 13 - 6).to(in_dtype))
            xs = torch.full((m, 1), .015, device='cuda')
            ws = torch.full((1, n), .025, device='cuda')
            bias = torch.linspace(-.25, .25, n, device='cuda').reshape(1, n)
            inputs = (x, w, xs, ws, bias)
            originals = tuple(value.clone() for value in inputs)
            expected = _torch_ref(*inputs, out_dtype)
            supplied = torch.full((m, n), float('nan'), device='cuda', dtype=out_dtype)
            actual = mod.gemm_a8w8(*inputs, out_dtype, y=supplied)
            require_unchanged(inputs, originals)
            _checked_scaled_gemm_output(actual, x, w, out_dtype)
            if actual.data_ptr() != supplied.data_ptr():
                raise AssertionError('Scaled GEMM ignored caller-provided output storage')
            torch.testing.assert_close(actual, expected, atol=.02, rtol=.01)
    # Odd large signed INT8 values distinguish INT8 from an illegal FP8 narrowing.
    m, n, k = 3, 5, 64
    x = (torch.arange(m*k, device='cuda').reshape(m, k) % 255 - 127).to(torch.int8)
    w = (torch.arange(n*k, device='cuda').reshape(n, k) * 7 % 255 - 127).to(torch.int8)
    xs = torch.full((m, 1), .015, device='cuda')
    ws = torch.full((1, n), .025, device='cuda')
    bias = torch.zeros((1, n), device='cuda')
    inputs = (x, w, xs, ws, bias)
    originals = tuple(value.clone() for value in inputs)
    expected = _torch_ref(*inputs, torch.bfloat16)
    actual = mod.gemm_a8w8(*inputs, torch.bfloat16)
    require_unchanged(inputs, originals)
    _checked_scaled_gemm_output(actual, x, w, torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=.02, rtol=.01)
    print("Unscored signed, strided and supplied-output scaled-GEMM controls: PASS")
