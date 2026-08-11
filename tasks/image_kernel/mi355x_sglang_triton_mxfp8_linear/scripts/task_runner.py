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
# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


if "_TimedRun" not in globals():
    class _TimedRun:
        """Source-tree fallback; workspace materialization supplies the real class."""

        def __init__(self):
            self._rerun = None
            self.outputs = None

        def _bind(self, rerun, outputs=None):
            self._rerun = rerun
            self.outputs = outputs

        @property
        def bound(self):
            return self._rerun is not None

        def rerun(self):
            if self._rerun is None:
                raise RuntimeError("timed run was never bound")
            self.outputs = self._rerun()
            return self.outputs


def _benchmark_cuda_graph(*args, **kwargs):
    """Compatibility name used by the task's standalone/forge drivers."""

    return _benchmark_cuda_graph_or_events(*args, **kwargs)


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
def _assert_close(case: dict, inputs: dict, got, label: str = "") -> float:
    err = _relerr(got, _reference(inputs))
    tol = case["params"].get("max_relerr", 0.06)
    assert err < tol, (case["id"], label, err, tol)
    return err


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the activation in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. The activation is re-quantized
    the same way ``_make`` built it, so the quantized tensor and its scale stay
    consistent; the weight stays fixed.
    """
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

    p = inputs["cfg"]["params"]
    torch.manual_seed(59)
    x_bf16 = torch.randn(p["m"], p["k"], device="cuda", dtype=torch.bfloat16) * 0.5
    x_fp8, x_scale = mxfp8_e4m3_quantize(x_bf16)
    inputs["x_fp8"].copy_(x_fp8)
    inputs["x_scale"].copy_(x_scale)


def _assert_timed_outputs(case: dict, inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against a freshly perturbed
    activation and checks the buffer it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    _assert_close(case, inputs, timed.rerun(), label="timed run")


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
        err = _assert_close(case, inputs, got)
        print("correctness PASS", case["id"], f"relerr={err:.4f}")


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch.cuda.synchronize()
        timed = _TimedRun()
        ms, bmeta = _benchmark_cuda_graph(lambda: _run(inputs), timed_run=timed)
        _assert_timed_outputs(case, inputs, timed)
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
