#!/usr/bin/env python3
"""Image-kernel harness for SGLang ``_mxfp8_grouped_gemm_kernel`` (fused MoE grouped GEMM).

Target device kernel : ``_mxfp8_grouped_gemm_kernel`` (tl.dot_scaled, CDNA4/gfx950)
Timed launcher       : ``_grouped_gemm_mxfp8`` -- both specializations of a MoE forward
                       (GEMM1: a_div=top_k; GEMM2: a_div=1, weighted) are launched back
                       to back, matching the two hot leaves in the profile.
Source               : sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py

The MoE-align / activation-quant / SwiGLU setup that surrounds the two GEMMs is built once
per case (untimed); only the grouped-GEMM launches are timed under a CUDA graph, so the
measurement isolates the target kernel and excludes host dispatch.

Shapes are the real MiniMax-M3-MXFP8 (TP=8) MoE dims (hidden 6144, per-rank inter 384,
128 experts, top-k 4). MXFP8 contract: FP8-E4M3 operands, UE8M0 uint8 per-1x32 block
scales, FP32 accumulate, BF16 output. See session_cases.json for provenance.
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
BLOCK_M = 64
CORRECTNESS_MAX_TOKENS = 64  # cap T so the O(T*top_k) python reference stays cheap


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
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
# CUDA-graph benchmark (device-time only; falls back to CUDA events).
# --------------------------------------------------------------------------- #
def _measure_cuda_event(fn, repetition):
    import torch

    times_ms = []
    for _ in range(max(1, int(repetition))):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times_ms.append(s.elapsed_time(e))
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
# Build the two grouped-GEMM invocations of one MoE forward (untimed setup).
# --------------------------------------------------------------------------- #
def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    from sglang.kernels.ops.moe.minimax_m3_swiglu import swiglu_oai_split
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import _grouped_gemm_mxfp8
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_e4m3_quantize_torch,
        mxfp8_e4m3_quantize,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    p = case["params"]
    T = min(p["tokens"], CORRECTNESS_MAX_TOKENS) if correctness else p["tokens"]
    H, I, E, top_k = p["hidden"], p["inter"], p["experts"], p["top_k"]
    alpha, beta, limit = p["alpha"], p["beta"], p["limit"]
    torch.manual_seed(case.get("seed", 0))

    hidden = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.5
    w13_bf16 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(w13_bf16)
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16)

    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    M = T * top_k
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, BLOCK_M, E)
    a_q, a_s = mxfp8_e4m3_quantize(hidden)

    # GEMM1 (materialize the real intermediate so GEMM2 inputs are realistic).
    g1 = _grouped_gemm_mxfp8(
        a_q, a_s, w13_fp8, w13_scale, sorted_ids, expert_ids, num_post,
        M, top_k, BLOCK_M, hidden.dtype, a_div=top_k,
    )
    act = swiglu_oai_split(g1, alpha=alpha, beta=beta, limit=limit, out_dtype=hidden.dtype)
    act_q, act_s = mxfp8_e4m3_quantize(act)
    mul_weight = topk_weights.reshape(-1).to(torch.float32)

    gemm1_args = dict(
        a_q=a_q, a_scale=a_s, w=w13_fp8, w_scale=w13_scale, sorted_token_ids=sorted_ids,
        expert_ids=expert_ids, num_tokens_post_padded=num_post, num_valid_tokens=M,
        top_k=top_k, block_m=BLOCK_M, out_dtype=hidden.dtype, a_div=top_k,
    )
    gemm2_args = dict(
        a_q=act_q, a_scale=act_s, w=w2_fp8, w_scale=w2_scale, sorted_token_ids=sorted_ids,
        expert_ids=expert_ids, num_tokens_post_padded=num_post, num_valid_tokens=M,
        top_k=top_k, block_m=BLOCK_M, out_dtype=torch.float32, a_div=1, mul_weight_by=mul_weight,
    )
    return {
        "cfg": case, "T": T, "H": H, "I": I, "top_k": top_k,
        "alpha": alpha, "beta": beta, "limit": limit,
        "gemm1_args": gemm1_args, "gemm2_args": gemm2_args,
        "topk_weights": topk_weights, "topk_ids": topk_ids,
        "a_q": a_q, "a_s": a_s, "w13_fp8": w13_fp8, "w13_scale": w13_scale,
        "w2_fp8": w2_fp8, "w2_scale": w2_scale,
    }


def _run_gemms(inputs: dict):
    """The timed region: the two grouped-GEMM launches of one MoE forward."""
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import _grouped_gemm_mxfp8

    _grouped_gemm_mxfp8(**inputs["gemm1_args"])
    return _grouped_gemm_mxfp8(**inputs["gemm2_args"])


def _fused_output(inputs: dict):
    """Full MoE output reconstructed from the two grouped GEMMs (for correctness)."""
    torch = _torch()
    from sglang.kernels.ops.moe.minimax_m3_swiglu import swiglu_oai_split
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import _grouped_gemm_mxfp8
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

    g1 = _grouped_gemm_mxfp8(**inputs["gemm1_args"])
    act = swiglu_oai_split(
        g1, alpha=inputs["alpha"], beta=inputs["beta"], limit=inputs["limit"],
        out_dtype=torch.bfloat16,
    )
    act_q, act_s = mxfp8_e4m3_quantize(act)
    args = dict(inputs["gemm2_args"])
    args["a_q"], args["a_scale"] = act_q, act_s
    g2 = _grouped_gemm_mxfp8(**args)  # [M, H] fp32, top-k weighted
    T, top_k, H = inputs["T"], inputs["top_k"], inputs["H"]
    return g2.view(T, top_k, H).sum(dim=1).to(torch.bfloat16)


def _reference(inputs: dict):
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import dequant_mxfp8_to_bf16

    x = dequant_mxfp8_to_bf16(inputs["a_q"], inputs["a_s"]).float()
    w13 = dequant_mxfp8_to_bf16(inputs["w13_fp8"], inputs["w13_scale"]).float()
    w2 = dequant_mxfp8_to_bf16(inputs["w2_fp8"], inputs["w2_scale"]).float()
    T, I, top_k = inputs["T"], inputs["I"], inputs["top_k"]
    alpha, beta, limit = inputs["alpha"], inputs["beta"], inputs["limit"]
    topk_weights, topk_ids = inputs["topk_weights"], inputs["topk_ids"]
    H = inputs["H"]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            if e < 0 or e >= w13.shape[0]:
                continue
            g1 = x[t] @ w13[e].T
            gate, up = g1[:I], g1[I:]
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            out[t] += topk_weights[t, j].float() * (act @ w2[e].T)
    return out.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    _run_gemms(inputs)
    _torch().cuda.synchronize()
    print("mxfp8_grouped_gemm compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case, correctness=True)
        got = _fused_output(inputs)
        torch.cuda.synchronize()
        err = _relerr(got, _reference(inputs))
        tol = case["params"].get("max_relerr", 0.08)
        assert err < tol, (case["id"], err, tol)
        print("correctness PASS", case["id"], f"T={inputs['T']}", f"relerr={err:.4f}")


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        _run_gemms(inputs)
        torch.cuda.synchronize()
        ms, bmeta = _benchmark_cuda_graph(lambda: _run_gemms(inputs))
        row = {
            "test_case_id": case["id"],
            "execution_time_ms": ms,
            "metadata": {**case["params"], "regime": case.get("regime"), **bmeta},
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
