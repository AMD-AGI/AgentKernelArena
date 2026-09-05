# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the FP8 per-token fused MoE with stage-1 token
weighting (``fmoe_g1u1_tkw1``).

The op is a top-k Mixture-of-Experts feed-forward block run in FP8
(``float8_e4m3fn``) with per-token scales, where the router weight is applied at
stage 1 (``tkw1``) instead of at the final combine, matching the AMD runtime op
(``aiter.fused_moe_bf16_asm.asm_moe_tkw1`` -> ``fused_moe`` with
``QuantType.per_Token`` and ``doweight_stage1=True``):

- a softmax router selects ``topk`` experts per token and renormalizes the
  weights;
- activations and expert weights are quantized per-token (per row) to FP8;
- stage 1 runs the grouped gate/up GEMM; the router weight scales BOTH the gate
  and the up projections (``gate*w``, ``up*w``) before ``silu(gate*w) * (up*w)``;
- the deployed kernel fuses both GEMMs in a single launch and keeps the stage-1
  intermediate in bf16 (not requantized to FP8);
- stage 2 runs the grouped down GEMM and sums the experts WITHOUT re-applying the
  router weight (it was already folded in at stage 1).

Each GEMM dequantizes its FP8 operands (value times per-token scale), accumulates
in fp32, and the output is bf16. This mirrors the upstream (b)-faithful reference
``aiter.fused_moe_bf16_asm.torch_moe_tkw1`` with the input/weight FP8 per-token
quantization made explicit so the reference matches the deployed FP8 op.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _amd_fp8_dtype():
    """fp8 storage dtype the matching aiter op uses on the active GPU arch,
    mirroring ``aiter/utility/dtypes.py``: gfx942/CDNA3 -> ``float8_e4m3fnuz``
    (finite max 240); gfx950/CDNA4 and others -> ``float8_e4m3fn`` (max 448)."""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        arch = ""
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn


# Arch-selected FP8 type and its saturation magnitude (per-token scale divisor).
_FP8_DTYPE = _amd_fp8_dtype()
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


def _pertoken_dequant(x):
    """FP8 per-token (per last-dim row) quantize+dequantize, returning the fp32
    values the FP8 GEMM sees. Matches ``aiter.pertoken_quant`` (amax/dtype_max,
    arch-selected e4m3 round, multiply back)."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (xf / scale).to(_FP8_DTYPE).float() * scale


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
    B, D = acts.shape
    topk = topk_ids.shape[1]
    N = weights.shape[1]
    out = torch.zeros(B, topk, N, dtype=torch.float32, device=acts.device)
    if expert_plan is None:
        h = acts.view(B, 1, D).repeat(1, topk, 1)
        for e in range(weights.shape[0]):
            mask = topk_ids == e
            if mask.any():
                out[mask] = h[mask] @ weights[e].transpose(0, 1)
    else:
        for e, token_id, topk_pos in expert_plan:
            out[token_id, topk_pos] = (
                acts[token_id] @ weights[e].transpose(0, 1)
            )
    return out


def _grouped_gemm_stage2_sum(acts, weights, topk_ids, expert_plan=None):
    """Per-expert down GEMM summed over top-k (no router-weight scaling; the
    weight was applied at stage 1)."""
    B, topk = topk_ids.shape
    model_dim = weights.shape[1]
    out = torch.zeros(B, topk, model_dim, dtype=torch.float32, device=acts.device)
    if expert_plan is None:
        for e in range(weights.shape[0]):
            mask = topk_ids == e
            if mask.any():
                out[mask] = acts[mask] @ weights[e].transpose(0, 1)
    else:
        for e, token_id, topk_pos in expert_plan:
            out[token_id, topk_pos] = (
                acts[token_id, topk_pos] @ weights[e].transpose(0, 1)
            )
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
        B = hidden_states.shape[0]

        a1 = _pertoken_dequant(hidden_states)
        w1 = _pertoken_dequant(self.w1)
        w2 = _pertoken_dequant(self.w2)

        stage1 = _grouped_gemm_stage1(a1, w1, topk_ids, expert_plan)
        gate, up = stage1.split([I, I], dim=-1)
        tk = topk_weights.view(B, self.topk, 1)
        gate = gate * tk
        up = up * tk
        if self.activation == "gelu":
            act = F.gelu(gate) * up
        else:
            act = F.silu(gate) * up

        a2 = act.to(torch.bfloat16).float()
        out = _grouped_gemm_stage2_sum(a2, w2, topk_ids, expert_plan)
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(128, 5120, dtype=torch.bfloat16)]


def get_init_inputs():
    return [5120, 1024, 16, 2]
