#!/usr/bin/env python3
"""Shared MoE blockscale-fp8 harness for K13 rewrite candidates (MiniMax-M2.5).

Same contract as the moe_opt 122B harness, RE-SHAPED to the real MiniMax-M2.5
production MoE op (`aiter.fmoe_fp8_blockscale_g1u1`, dispatched in
aiter/fused_moe.py for QuantType.per_1x128):

    model_dim=3072, inter_dim=1536, experts=256, topk=8, per-128 blockscale,
    bf16 activations / fp8 e4m3 weights+act.  (config.json of /wekafs/ethany/
    vllm_workspace/MiniMax-M2.5: hidden_size=3072, intermediate_size=1536,
    num_local_experts=256, num_experts_per_tok=8, weight_block_size=[128,128].)

All candidates use IDENTICAL quantized inputs and are validated against the SAME
PyTorch fp32 golden (torch_moe_blockscale from this aiter's op_tests). Candidate
contract:
    def candidate(prep: dict) -> torch.Tensor   # [token, model_dim], bf16

Do NOT call aiter.fmoe_fp8_blockscale_g1u1 in a candidate (that is the ASM .co
target we want to BEAT with pure-source CK / FlyDSL / Triton).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import aiter_local as aiter
from aiter_local import dtypes, pertoken_quant
from einops import rearrange

OUT_DIR = Path(__file__).resolve().parent


# --- golden reference (vendored verbatim from aiter v0.1.10 op_tests/test_moe_blockscale.py
# `torch_moe_blockscale`, so the harness is self-contained on the installed aiter wheel and
# does NOT need the aiter source tree / op_tests). fp32 blockscale-dequant + grouped SwiGLU MoE. ---
def torch_moe_blockscale(
    hidden_states,
    w1,  # [expert, inter_dim*2, model_dim]
    w2,  # [expert, model_dim, inter_dim]
    topk_weight,
    topk_ids,
    dtype,
    scale_blks=(128, 128),
    a_scale=None,
    fc1_scale=None,  # [expert, inter_dim/blk_m, model_dim/blk_k]
    fc2_scale=None,  # [expert, model_dim/blk_m, inter_dim/blk_k]
    expert_mask=None,
):
    computeType = dtypes.fp32
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    token_num, topk = topk_ids.shape
    expert, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    blk_n, blk_k = scale_blks
    if a_scale is not None:
        hidden_states = hidden_states.view(token_num, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(token_num, -1)

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros((B, topk, D), dtype=computeType, device=hidden_states.device)
    if w2.shape[2] * 2 == w1.shape[1]:
        moeType = "g1u1"
    else:
        moeType = "g1u0"

    nblk_n = inter_dim // blk_n
    nblk_k = model_dim // blk_k
    if fc1_scale is not None:
        fc1_scale = rearrange(
            fc1_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, -1, nblk_k, blk_n, blk_k),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        fc2_scale = rearrange(
            fc2_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, nblk_k, nblk_n, blk_k, blk_n),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w1 = w1 * fc1_scale
        w2 = w2 * fc2_scale

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)

# --- MiniMax-M2.5 production MoE shape (PER-GPU, TP=2) ---
# config.json: hidden 3072, intermediate 1536, 256 experts, topk 8. The benchmark
# runs tensor-parallel TP=2, which shards the expert FFN intermediate across GPUs,
# so each GPU's fmoe kernel sees inter_dim = 1536/2 = 768 (w1=[E,2*768,3072],
# w2=[E,3072,768]). This is the shape the kernel actually processes e2e (and keeps
# the flattened weight buffer < int32, which the full 1536 would overflow).
TP = 2
SHAPE = dict(model_dim=3072, inter_dim=1536 // TP, expert=256, topk=8, scale_blks=(128, 128))

# token regimes that matter for the e2e workload (CONC=64 decode; ISL=1024 prefill chunk)
CASES = [("mm_decode", 64), ("mm_med", 256), ("mm_prefill", 1024), ("mm_big", 4096)]

COS_THRESH = 0.01  # aiter's strict tune threshold (matches moe_opt)


def prepare(token: int, seed: int = 0) -> dict:
    md = SHAPE["model_dim"]
    idim = SHAPE["inter_dim"]
    E = SHAPE["expert"]
    topk = SHAPE["topk"]
    bn, bk = SHAPE["scale_blks"]
    dtype = dtypes.bf16
    g = torch.Generator(device="cuda").manual_seed(seed)

    inp = torch.randn((token, md), dtype=dtype, device="cuda", generator=g)
    w1 = torch.randn((E, idim * 2, md), dtype=dtype, device="cuda", generator=g) / 10
    w2 = torch.randn((E, md, idim), dtype=dtype, device="cuda", generator=g) / 10
    score = torch.randn((token, E), dtype=dtype, device="cuda", generator=g)
    topk_weights, topk_ids = torch.topk(score.float(), topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    def bq(w):
        tmp = rearrange(
            w.view(-1, w.shape[1] // bn, bn, w.shape[2] // bk, bk),
            "e nbn blkn nbk blkk -> e nbn nbk (blkn blkk)",
        ).contiguous()
        wq, ws = pertoken_quant(tmp, quant_dtype=dtypes.fp8)
        wq = rearrange(
            wq.view(-1, w.shape[1] // bn, w.shape[2] // bk, bn, bk),
            "e nbn nbk blkn blkk -> e (nbn blkn) (nbk blkk)",
        ).contiguous()
        return wq, ws.view(E, -1)

    w1_q, w1_scale = bq(w1)
    w2_q, w2_scale = bq(w2)
    a1_q, a1_scale = pertoken_quant(inp.view(-1, md // bk, bk), quant_dtype=dtypes.fp8)
    a1_q = a1_q.view(-1, md)
    a1_scale = a1_scale.squeeze(-1)

    return dict(
        token=token, model_dim=md, inter_dim=idim, expert=E, topk=topk,
        scale_blks=(bn, bk), dtype=dtype,
        input=inp,
        a1_q=a1_q, a1_scale=a1_scale,
        w1_q=w1_q, w1_scale=w1_scale,
        w2_q=w2_q, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids,
    )


def golden(prep: dict) -> torch.Tensor:
    return torch_moe_blockscale(
        prep["a1_q"], prep["w1_q"], prep["w2_q"],
        prep["topk_weights"], prep["topk_ids"], prep["dtype"],
        scale_blks=prep["scale_blks"],
        fc1_scale=prep["w1_scale"], fc2_scale=prep["w2_scale"], a_scale=prep["a1_scale"],
    )


def cosine_diff(ref: torch.Tensor, res: torch.Tensor) -> float:
    x, y = ref.double().flatten(), res.double().flatten()
    denom = (x * x + y * y).sum().clamp_min(1e-12)
    return float(1 - 2 * (x * y).sum() / denom)


def snr_db(ref: torch.Tensor, res: torch.Tensor) -> float:
    num = (ref.float() - res.float()).pow(2).mean().item()
    den = ref.float().pow(2).mean().item()
    return 10.0 * torch.log10(torch.tensor(den / (num + 1e-20))).item()


def asm_baseline(prep: dict) -> torch.Tensor:
    """The ASM .co target (the bar to beat). Used only to measure the baseline."""
    return aiter.fmoe_fp8_blockscale_g1u1  # placeholder; real call wired in driver


if __name__ == "__main__":
    from aiter_local.jit.utils.chip_info import get_gfx, get_cu_num
    print(f"GPU {get_gfx()} CU={get_cu_num()} | shape {SHAPE}")
    for nm, tok in CASES[:2]:
        p = prepare(tok)
        gout = golden(p)
        print(f"{nm}: token={tok} golden out {tuple(gout.shape)} dtype={gout.dtype} "
              f"norm={gout.float().norm().item():.3f}")
