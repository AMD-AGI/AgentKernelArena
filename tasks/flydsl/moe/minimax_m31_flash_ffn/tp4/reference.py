# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""High-precision reference of the MiniMax-M3.1-Flash FFN launch on a tensor-parallel-4 group, used only for
correctness. It receives the logical inputs (inputs.py: rank-specific tensors stacked on a leading TP axis,
unshuffled codes and raw E8M0 scales) and returns the logical outputs; inputs.rank_expected gives them to
every rank.

The numerics are in ffn_numerics.py: float64 throughout, with the operator's own roundings (per-token FP8
of the attention output, bf16 o_proj partials, the bf16 attention all-reduce, the bf16 normalized input,
the bf16 SwiGLU-OAI intermediate and the bf16 MoE partial of each rank). The routed experts are computed
from their exact MXFP4 values (E2M1 x 2^(e8m0 - 127)) and the shared expert from its MXFP8 values, without
AITER's fused MoE. Returned values are taken before their final rounding: res_out = bf16(attention
all-reduce) + res, out = the sum of the ranks' bf16 MoE partials.
"""
from ffn_numerics import ffn


def reference(*, attn, res, wo, wos, norm_w, router_w, bias, w13, s13, w2, s2, w13s, s13s, w2s, s2s,
              **_reserved):
    return ffn({"attn": attn, "res": res, "wo": wo, "wos": wos, "norm_w": norm_w, "router_w": router_w,
                "bias": bias, "w13": w13, "s13": s13, "w2": w2, "s2": s2, "w13s": w13s, "s13s": s13s,
                "w2s": w2s, "s2s": s2s})
