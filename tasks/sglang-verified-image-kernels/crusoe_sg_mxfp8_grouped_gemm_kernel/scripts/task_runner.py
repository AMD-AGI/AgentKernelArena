#!/usr/bin/env python3
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
    sgl_py = os.environ.get(
        "SGLANG_PYTHON", "/sgl-workspace/sglang/python"
    )
    if sgl_py not in sys.path:
        sys.path.insert(0, sgl_py)
    os.chdir(WORKSPACE)


def _torch():
    import torch
    return torch


def _relerr(a, b) -> float:
    a = a.float()
    b = b.float()
    return float(((a - b).norm() / (b.norm() + 1e-8)).item())


def _make_linear(case: dict) -> dict:
    torch = _torch()
    torch.manual_seed(case.get("seed", 0))
    p = case["params"]
    m, n, k = p["m"], p["n"], p["k"]
    dtype = getattr(torch, p.get("dtype", "bfloat16"))
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_e4m3_quantize_torch,
    )

    w_bf16 = torch.randn(n, k, device="cuda", dtype=dtype) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16)
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.5
    return {"x": x, "w_fp8": w_fp8, "w_scale": w_scale, "cfg": case}


def _run_linear(inputs: dict):
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_dot_scaled_linear,
    )
    return _mxfp8_dot_scaled_linear(inputs["x"], inputs["w_fp8"], inputs["w_scale"])


def _ref_linear(inputs: dict):
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import dequant_mxfp8_to_bf16

    w_deq = dequant_mxfp8_to_bf16(inputs["w_fp8"], inputs["w_scale"])
    return torch.nn.functional.linear(inputs["x"], w_deq).to(inputs["x"].dtype)


def _make_grouped(case: dict) -> dict:
    torch = _torch()
    torch.manual_seed(case.get("seed", 0))
    p = case["params"]
    T, H, inter, E, top_k = p["T"], p["H"], p["inter"], p["E"], p["top_k"]
    alpha, beta, limit = p.get("alpha", 1.702), p.get("beta", 1.0), p.get("limit", 7.0)
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_e4m3_quantize_torch,
    )

    w13_bf16 = torch.randn(E, 2 * inter, H, device="cuda", dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, inter, device="cuda", dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(w13_bf16)
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    return {
        "x": x,
        "w13_fp8": w13_fp8,
        "w13_scale": w13_scale,
        "w2_fp8": w2_fp8,
        "w2_scale": w2_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "alpha": alpha,
        "beta": beta,
        "limit": limit,
        "cfg": case,
    }


def _run_grouped(inputs: dict):
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import fused_moe_mxfp8_native

    return fused_moe_mxfp8_native(
        inputs["x"],
        inputs["w13_fp8"],
        inputs["w13_scale"],
        inputs["w2_fp8"],
        inputs["w2_scale"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        alpha=inputs["alpha"],
        beta=inputs["beta"],
        limit=inputs["limit"],
    )


def _ref_grouped(inputs: dict):
    torch = _torch()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import dequant_mxfp8_to_bf16

    x = inputs["x"]
    w13 = dequant_mxfp8_to_bf16(inputs["w13_fp8"], inputs["w13_scale"])
    w2 = dequant_mxfp8_to_bf16(inputs["w2_fp8"], inputs["w2_scale"])
    T, H = x.shape
    inter = w2.shape[-1]
    top_k = inputs["topk_ids"].shape[1]
    alpha, beta, limit = inputs["alpha"], inputs["beta"], inputs["limit"]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(inputs["topk_ids"][t, j].item())
            if e < 0 or e >= w13.shape[0]:
                continue
            g1 = x[t].float() @ w13[e].float().T
            gate = g1[:inter]
            up = g1[inter:]
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            g2 = act @ w2[e].float().T
            out[t] += inputs["topk_weights"][t, j].float() * g2
    return out.to(x.dtype)


def _make(case: dict, correctness: bool = False) -> dict:
    if OPERATOR == "mxfp8_linear":
        return _make_linear(case)
    if OPERATOR == "mxfp8_grouped_gemm":
        return _make_grouped(case)
    raise KeyError(OPERATOR)


def _run(inputs: dict):
    if OPERATOR == "mxfp8_linear":
        return _run_linear(inputs)
    if OPERATOR == "mxfp8_grouped_gemm":
        return _run_grouped(inputs)
    raise KeyError(OPERATOR)


def _reference(inputs: dict):
    if OPERATOR == "mxfp8_linear":
        return _ref_linear(inputs)
    if OPERATOR == "mxfp8_grouped_gemm":
        return _ref_grouped(inputs)
    raise KeyError(OPERATOR)


def run_compile() -> None:
    inputs = _make(CASES[0])
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        _torch().cuda.synchronize()
        ref = _reference(inputs)
        err = _relerr(got, ref)
        assert err < case["params"].get("max_relerr", 0.06), (case["id"], err)
        print("correctness PASS", case["id"], f"relerr={err:.4f}")


def run_performance() -> None:
    import time

    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch = _torch()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run(inputs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        med = sorted(times)[len(times) // 2]
        print(case["id"], f"{med:.6f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    args = parser.parse_args()
    if args.mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
