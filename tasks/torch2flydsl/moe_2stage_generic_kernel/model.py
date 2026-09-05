# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the generic two-stage fused MoE (``ck_moe`` /
``moe_stage1_g1u1``).

The op is a top-k Mixture-of-Experts feed-forward block evaluated as a grouped
two-stage GEMM in bf16, matching the AMD runtime generic CK MoE path
(``aiter.fused_moe`` with ``QuantType.No``):

- a softmax router selects ``topk`` experts per token and renormalizes the
  weights;
- stage 1 runs the grouped gate/up GEMM (``w1``) followed by ``silu(gate) * up``;
- the stage-1 intermediate is truncated to bf16 (the operand the stage-2 GEMM
  consumes);
- stage 2 runs the grouped down GEMM (``w2``) and combines the experts with the
  renormalized router weights.

Each GEMM accumulates in fp32 over bf16 operands and the final output is bf16,
mirroring the upstream (b)-faithful reference
``aiter.fused_moe.torch_moe_stage1`` / ``torch_moe_stage2`` (no quantization).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_expert_plan(topk_ids, num_experts):
    """Resolve data-dependent routing indices once, outside graph capture."""
    plan = []
    for expert in range(num_experts):
        token_id, topk_pos = torch.where(topk_ids == expert)
        if token_id.numel():
            plan.append((expert, token_id, topk_pos))
    return tuple(plan)


def _grouped_gemm_stage1(acts, weights, topk_ids, expert_plan=None):
    """Per-expert grouped GEMM: out[b, k] = acts[b] @ weights[topk_ids[b, k]].T."""
    acts = acts.float()
    B, D = acts.shape
    topk = topk_ids.shape[1]
    N = weights.shape[1]
    out = torch.zeros(B, topk, N, dtype=torch.float32, device=acts.device)
    if expert_plan is None:
        h = acts.view(B, 1, D).repeat(1, topk, 1)
        for e in range(weights.shape[0]):
            mask = topk_ids == e
            if mask.any():
                out[mask] = h[mask] @ weights[e].float().transpose(0, 1)
    else:
        for e, token_id, topk_pos in expert_plan:
            out[token_id, topk_pos] = (
                acts[token_id] @ weights[e].float().transpose(0, 1)
            )
    return out


def _grouped_gemm_stage2(
    acts, weights, topk_ids, topk_weights, expert_plan=None
):
    """Per-expert down GEMM with weighted top-k combine to a single output row."""
    acts = acts.float()
    B, topk = topk_ids.shape
    model_dim = weights.shape[1]
    out = torch.zeros(B, topk, model_dim, dtype=torch.float32, device=acts.device)
    if expert_plan is None:
        for e in range(weights.shape[0]):
            mask = topk_ids == e
            if mask.any():
                out[mask] = acts[mask] @ weights[e].float().transpose(0, 1)
    else:
        for e, token_id, topk_pos in expert_plan:
            out[token_id, topk_pos] = (
                acts[token_id, topk_pos]
                @ weights[e].float().transpose(0, 1)
            )
    out = out * topk_weights.view(B, topk, 1)
    return out.sum(1)


def route_topk(logits, topk):
    """Softmax router + top-k with renormalized weights. Shared by the harness.

    Ties are broken by ascending expert index via a stable descending sort. The
    bf16 gate produces many duplicate logits across the large expert count, and a
    nondeterministic top-k tie-break would let the reference and the runtime op
    select different experts; the stable order keeps both routings identical.
    """
    gate = torch.softmax(logits.float(), dim=-1)
    order = torch.sort(gate, dim=-1, descending=True, stable=True).indices
    ids = order[..., :topk]
    weights = torch.gather(gate, -1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), ids.to(torch.int32)


class Model(nn.Module):
    def __init__(self, model_dim, inter_dim, experts, topk, activation="silu"):
        super().__init__()
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        self.activation = activation
        self.gate = nn.Linear(model_dim, experts, bias=False).to(torch.bfloat16)
        self.w1 = nn.Parameter(
            (torch.randn(experts, 2 * inter_dim, model_dim) / 10).to(torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            (torch.randn(experts, model_dim, inter_dim) / 10).to(torch.bfloat16)
        )

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        topk_weights, topk_ids = route_topk(logits, self.topk)
        return self.forward_with_routing(hidden_states, topk_weights, topk_ids)

    def forward_with_routing(
        self, hidden_states, topk_weights, topk_ids, expert_plan=None
    ):
        I = self.inter_dim

        stage1 = _grouped_gemm_stage1(
            hidden_states, self.w1, topk_ids, expert_plan
        )
        gate, up = stage1.split([I, I], dim=-1)
        if self.activation == "gelu":
            act = F.gelu(gate) * up
        else:
            act = F.silu(gate) * up
        a2 = act.to(torch.bfloat16)

        out = _grouped_gemm_stage2(
            a2, self.w2, topk_ids, topk_weights, expert_plan
        )
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(32, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return [4096, 1024, 32, 5]
