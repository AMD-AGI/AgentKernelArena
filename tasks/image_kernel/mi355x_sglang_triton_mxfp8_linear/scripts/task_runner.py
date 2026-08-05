#!/usr/bin/env python3
"""Image-kernel harness for SGLang ``_mxfp8_linear_kernel`` (dense MXFP8 GEMM).

Target device kernel : ``_mxfp8_linear_kernel``  (tl.dot_scaled, CDNA4/gfx950)
Timed launcher       : ``_run_mxfp8_linear_kernel`` (inner GEMM only; excludes the
                       separate activation-quant kernel, matching the profiled hot leaf)
Source               : sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py

Shapes are the real MiniMax-M3-MXFP8 (TP=8) dense-linear families recovered from the
2026-07-23 session GEAK capture + model config (see session_cases.json). MXFP8 contract:
FP8-E4M3 operands, UE8M0 uint8 per-1x32 block scales, FP32 accumulate, BF16 output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Prefer the workspace-seeded editable copy so the agent's edits take effect;
    # fall back to the in-image install for standalone/dev runs.
    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python"))
    os.chdir(WORKSPACE)


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _relerr(a, b) -> float:
    a = a.float()
    b = b.float()
    return float(((a - b).norm() / (b.norm() + 1e-8)).item())


# --------------------------------------------------------------------------- #
# CUDA-graph benchmark: capture N kernel launches, replay, time the graph. This
# amortizes host launch overhead so the measurement reflects device time only.
# Falls back to per-call CUDA-event timing if graph capture is unavailable.
# --------------------------------------------------------------------------- #
def _measure_cuda_event(fn, repetition):
    import torch

    times_ms = []
    for _ in range(max(1, int(repetition))):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _benchmark_cuda_graph(fn, warmup=10, repetition=100, target_ms=1.0, max_graph_repeats=200):
    import torch

    for _ in range(max(0, int(warmup))):
        fn()
    torch.cuda.synchronize()

    meta = {"benchmark_target_ms": float(target_ms), "benchmark_samples": int(repetition)}
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            est = torch.cuda.CUDAGraph()
            with torch.cuda.graph(est):
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(stream)
            est.replay()
            e.record(stream)
            torch.cuda.synchronize()
            est_ms = s.elapsed_time(e) / 3
            repeats = min(max_graph_repeats, max(1, int(target_ms / max(est_ms, 1e-9))))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
            torch.cuda.synchronize()

            times = []
            for _ in range(max(1, int(repetition))):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(stream)
                graph.replay()
                e.record(stream)
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) / repeats)
        mean_ms = sum(times) / len(times)
        if mean_ms < 1e-5:
            raise RuntimeError("empty_cuda_graph_capture")
        meta.update(benchmark_method="cuda_graph", benchmark_effective_repeats=int(repeats))
        return mean_ms, meta
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        times = _measure_cuda_event(fn, repetition)
        meta.update(
            benchmark_method="cuda_event_fallback",
            benchmark_effective_repeats=int(repetition),
            benchmark_fallback_reason=f"{type(exc).__name__}: {str(exc)[:160]}",
        )
        return sum(times) / len(times), meta


# --------------------------------------------------------------------------- #
# Inputs / call / reference
# --------------------------------------------------------------------------- #
def _make(case: dict) -> dict:
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_e4m3_quantize_torch,
        mxfp8_e4m3_quantize,
    )

    p = case["params"]
    m, n, k = p["m"], p["n"], p["k"]
    torch.manual_seed(case.get("seed", 0))

    # Weight: FP8-E4M3 + UE8M0 per-1x32 block scale (the persisted model weight).
    w_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16)
    # Activation: MXFP8-quantized once (as the server does before the GEMM).
    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    x_fp8, x_scale = mxfp8_e4m3_quantize(x_bf16)
    return {
        "cfg": case,
        "x_fp8": x_fp8,
        "x_scale": x_scale,
        "w_fp8": w_fp8,
        "w_scale": w_scale,
    }


def _run(inputs: dict):
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import _run_mxfp8_linear_kernel

    return _run_mxfp8_linear_kernel(
        inputs["x_fp8"],
        inputs["x_scale"],
        inputs["w_fp8"],
        inputs["w_scale"],
        torch.bfloat16,
    )


def _reference(inputs: dict):
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import dequant_mxfp8_to_bf16

    x = dequant_mxfp8_to_bf16(inputs["x_fp8"], inputs["x_scale"])
    w = dequant_mxfp8_to_bf16(inputs["w_fp8"], inputs["w_scale"])
    return torch.nn.functional.linear(x, w).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _make(CASES[0])
    _run(inputs)
    _torch().cuda.synchronize()
    print("mxfp8_linear compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        err = _relerr(got, _reference(inputs))
        tol = case["params"].get("max_relerr", 0.06)
        assert err < tol, (case["id"], err, tol)
        print("correctness PASS", case["id"], f"relerr={err:.4f}")


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch.cuda.synchronize()
        ms, bmeta = _benchmark_cuda_graph(lambda: _run(inputs))
        row = {
            "test_case_id": case["id"],
            "execution_time_ms": ms,
            "metadata": {**case["params"], "family": case.get("family"),
                         "regime": case.get("regime"), **bmeta},
        }
        rows.append(row)
        print(case["id"], f"{ms:.6f} ms", bmeta.get("benchmark_method"),
              bmeta.get("benchmark_fallback_reason", ""))
    out = WORKSPACE / "build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance_report.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[mode]()


if __name__ == "__main__":
    main()
