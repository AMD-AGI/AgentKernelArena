# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the FP8 block-scaled fused MoE
``fmoe_fp8_blockscale_g1u1``.

The op is a top-k Mixture-of-Experts feed-forward block run in FP8
(``float8_e4m3fn``) with 128x128 block scales, matching the AMD runtime
two-stage g1u1 MoE GEMM:

- a softmax router selects ``topk`` experts per token and renormalizes the
  weights;
- activations are quantized per-token, per-1x128 along the contraction dim;
- expert weights (``w1`` gate/up and ``w2`` down) are quantized per-128x128
  block;
- stage 1 runs the grouped gate/up GEMM followed by ``silu(gate) * up``;
- the stage-1 intermediate is requantized to FP8 per-token, per-1x128 along the
  intermediate dim (the fused kernel feeds FP8 operands into stage 2);
- stage 2 runs the grouped down GEMM and combines the experts with the
  renormalized router weights.

Each quantized GEMM dequantizes its FP8 operands (value times block scale),
accumulates in fp32, and the final output is truncated to bf16. The FP8 block
quantization here (``quantize_blockscale_moe``) is the single source of the FP8
operands and scales reused by the harness to drive the real AMD runtime op, so
the reference and the hardware kernel operate on byte-identical inputs and
differ only by fp32 accumulation order. This mirrors the upstream (b)-faithful
reference ``op_tests/test_moe_blockscale.py:torch_moe_blockscale``.
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


# Arch-selected FP8 type and its saturation magnitude (per-block scale divisor).
_FP8_DTYPE = _amd_fp8_dtype()
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_BLOCK = 128


def _quantize_act_1x128(a):
    """Per-token, per-1x128 (contraction dim) FP8 quantization of ``a``
    (``[token, D]``). Returns FP8 codes (``[token, D]``) and fp32 scales
    (``[token, D/128]``)."""
    token, d = a.shape
    nblk = d // _BLOCK
    af = a.float().view(token, nblk, _BLOCK)
    amax = af.abs().amax(dim=-1)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (af / scale.unsqueeze(-1)).view(token, d).to(_FP8_DTYPE)
    return q, scale


def _quantize_weight_128x128(w):
    """Per-128x128 block FP8 quantization of expert weights ``w``
    (``[E, dim1, dim2]``). Returns FP8 codes (same shape) and fp32 scales
    (``[E, dim1/128, dim2/128]``)."""
    e, dim1, dim2 = w.shape
    nb1, nb2 = dim1 // _BLOCK, dim2 // _BLOCK
    blocks = (
        w.float()
        .view(e, nb1, _BLOCK, nb2, _BLOCK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, nb1, nb2, _BLOCK * _BLOCK)
    )
    amax = blocks.abs().amax(dim=-1)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (blocks / scale.unsqueeze(-1)).to(_FP8_DTYPE)
    q = (
        q.view(e, nb1, nb2, _BLOCK, _BLOCK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, dim1, dim2)
        .contiguous()
    )
    return q, scale


def quantize_blockscale_moe(hidden_states, w1, w2):
    """Quantize the activation and expert weights to the FP8 block-scaled
    operands the deployed g1u1 MoE GEMM consumes.

    Returns ``(a1_fp8, a1_scale, w1_fp8, w1_scale, w2_fp8, w2_scale)`` with
    layouts matching the AMD runtime FP8 block-scaled fused MoE."""
    a1_fp8, a1_scale = _quantize_act_1x128(hidden_states)
    w1_fp8, w1_scale = _quantize_weight_128x128(w1)
    w2_fp8, w2_scale = _quantize_weight_128x128(w2)
    return a1_fp8, a1_scale, w1_fp8, w1_scale, w2_fp8, w2_scale


def _dequant_act(a_fp8, a_scale):
    token, d = a_fp8.shape
    nblk = a_scale.shape[1]
    return (a_fp8.float().view(token, nblk, _BLOCK) * a_scale.unsqueeze(-1)).view(
        token, d
    )


def _requant_act_1x128(a):
    """FP8 per-token, per-1x128 quantize+dequantize over the last dim, matching
    the FP8 operands the fused kernel feeds into the stage-2 down GEMM."""
    *lead, d = a.shape
    nblk = d // _BLOCK
    af = a.float().reshape(*lead, nblk, _BLOCK)
    amax = af.abs().amax(dim=-1, keepdim=True)
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    q = (af / scale).to(_FP8_DTYPE).float() * scale
    return q.reshape(*lead, d)


def _dequant_weight(w_fp8, w_scale):
    e, dim1, dim2 = w_fp8.shape
    scale_full = w_scale.repeat_interleave(_BLOCK, dim=1).repeat_interleave(
        _BLOCK, dim=2
    )
    return w_fp8.float() * scale_full


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


def _grouped_gemm_stage2(
    acts, weights, topk_ids, topk_weights, expert_plan=None
):
    """Per-expert down GEMM with weighted top-k combine to a single output row."""
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
    def __init__(self, model_dim, inter_dim, experts, topk):
        super().__init__()
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
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

        a1_fp8, a1_scale, w1_fp8, w1_scale, w2_fp8, w2_scale = quantize_blockscale_moe(
            hidden_states, self.w1, self.w2
        )
        a1 = _dequant_act(a1_fp8, a1_scale)
        w1 = _dequant_weight(w1_fp8, w1_scale)
        w2 = _dequant_weight(w2_fp8, w2_scale)

        stage1 = _grouped_gemm_stage1(a1, w1, topk_ids, expert_plan)
        gate, up = stage1.split([I, I], dim=-1)
        act = _requant_act_1x128(F.silu(gate) * up)

        out = _grouped_gemm_stage2(
            act, w2, topk_ids, topk_weights, expert_plan
        )
        return out.to(torch.bfloat16)


def get_inputs():
    return [torch.randn(16, 7168, dtype=torch.bfloat16)]


def get_init_inputs():
    return [7168, 256, 257, 9]
