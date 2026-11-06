# SPDX-License-Identifier: MIT
"""Sqrt-softplus top-k routing in Triton, one program per token.

Triton rewrite of the routing semantics of ROCm/aiter's topk_softplus for bfloat16 logits, 256 experts, top-8 and renormalized weights.
"""
import triton
import triton.language as tl


@triton.jit
def _topk_softplus_kernel(gating_ptr, bias_ptr, weights_ptr, ids_ptr, route_scale,
                          N_EXPERTS: tl.constexpr, BLOCK: tl.constexpr, TOPK: tl.constexpr,
                          RENORM: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    expert = tl.arange(0, BLOCK)
    valid = expert < N_EXPERTS
    logits = tl.load(gating_ptr + token * N_EXPERTS + expert, mask=valid, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + expert, mask=valid, other=0.0)

    # softplus(x) = log(1 + e^x), evaluated as log(v) * u / (v - 1) with v = 1 + u to stay accurate
    # for very negative x; the identity is used above the usual threshold.
    u = tl.exp(logits)
    v = 1.0 + u
    softplus = tl.where(v == 1.0, u, tl.log(v) * u / (v - 1.0))
    softplus = tl.where(logits > 20.0, logits, softplus)
    scores = tl.sqrt(softplus)
    choice = tl.where(valid, scores + bias, float("-inf"))

    k_idx = tl.arange(0, TOPK)
    out_w = tl.zeros([TOPK], dtype=tl.float32)
    out_id = tl.zeros([TOPK], dtype=tl.int32)
    for i in tl.static_range(TOPK):
        best = tl.max(choice, axis=0)
        best_e = tl.min(tl.where(choice == best, expert, BLOCK), axis=0)
        weight = tl.sum(tl.where(expert == best_e, scores, 0.0), axis=0)
        out_w = tl.where(k_idx == i, weight, out_w)
        out_id = tl.where(k_idx == i, best_e, out_id)
        choice = tl.where(expert == best_e, float("-inf"), choice)

    if RENORM:
        out_w = out_w / tl.sum(out_w, axis=0)
    out_w = out_w * route_scale
    tl.store(weights_ptr + token * TOPK + k_idx, out_w)
    tl.store(ids_ptr + token * TOPK + k_idx, out_id)


def run(*, gating_output, correction_bias, route_scale, topk_weights, topk_ids):
    n_tokens, n_experts = gating_output.shape
    _topk_softplus_kernel[(n_tokens,)](
        gating_output, correction_bias, topk_weights, topk_ids, route_scale,
        N_EXPERTS=n_experts, BLOCK=triton.next_power_of_2(n_experts), TOPK=topk_ids.shape[1],
        RENORM=True, num_warps=1)
