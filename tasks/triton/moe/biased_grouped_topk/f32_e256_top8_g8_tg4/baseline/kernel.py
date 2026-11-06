# SPDX-License-Identifier: MIT
"""DeepSeek-V3 biased grouped top-k routing in Triton, one program per token.

Triton rewrite of the routing semantics of ROCm/aiter's biased_grouped_topk_hip for 256 experts in
8 groups, 4 selected groups and 8 selected experts.
"""
import triton
import triton.language as tl


@triton.jit
def _biased_grouped_topk_kernel(gating_ptr, bias_ptr, weights_ptr, ids_ptr, route_scale,
                                N_GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                                TOPK_GROUPS: tl.constexpr, TOPK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    g_idx = tl.arange(0, N_GROUPS)
    e_in_group = tl.arange(0, GROUP_SIZE)
    expert = g_idx[:, None] * GROUP_SIZE + e_in_group[None, :]
    n_experts: tl.constexpr = N_GROUPS * GROUP_SIZE

    logits = tl.load(gating_ptr + token * n_experts + expert)
    bias = tl.load(bias_ptr + expert)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    choice = scores + bias

    # Group score: sum of the two largest selection scores in the group.
    first = tl.max(choice, axis=1)
    first_pos = tl.min(tl.where(choice == first[:, None], e_in_group[None, :], GROUP_SIZE), axis=1)
    rest = tl.where(e_in_group[None, :] == first_pos[:, None], float("-inf"), choice)
    group_score = first + tl.max(rest, axis=1)

    # Keep the TOPK_GROUPS best groups.
    keep = g_idx < 0
    for _ in tl.static_range(TOPK_GROUPS):
        best = tl.max(group_score, axis=0)
        best_g = tl.min(tl.where(group_score == best, g_idx, N_GROUPS), axis=0)
        keep = keep | (g_idx == best_g)
        group_score = tl.where(g_idx == best_g, float("-inf"), group_score)
    masked = tl.where(keep[:, None], choice, 0.0)

    # Pick the TOPK best experts among the kept groups.
    k_idx = tl.arange(0, TOPK)
    out_w = tl.zeros([TOPK], dtype=tl.float32)
    out_id = tl.zeros([TOPK], dtype=tl.int32)
    for i in tl.static_range(TOPK):
        best = tl.max(tl.max(masked, axis=1), axis=0)
        best_e = tl.min(tl.min(tl.where(masked == best, expert, n_experts), axis=1), axis=0)
        weight = tl.sum(tl.sum(tl.where(expert == best_e, scores, 0.0), axis=1), axis=0)
        out_w = tl.where(k_idx == i, weight, out_w)
        out_id = tl.where(k_idx == i, best_e, out_id)
        masked = tl.where(expert == best_e, float("-inf"), masked)

    out_w = out_w / tl.sum(out_w, axis=0) * route_scale
    tl.store(weights_ptr + token * TOPK + k_idx, out_w)
    tl.store(ids_ptr + token * TOPK + k_idx, out_id)


def run(*, gating_output, correction_bias, route_scale, topk_weights, topk_ids):
    n_tokens, n_experts = gating_output.shape
    _biased_grouped_topk_kernel[(n_tokens,)](
        gating_output, correction_bias, topk_weights, topk_ids, route_scale,
        N_GROUPS=8, GROUP_SIZE=n_experts // 8, TOPK_GROUPS=4, TOPK=topk_ids.shape[1], num_warps=1)
