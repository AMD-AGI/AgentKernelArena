#!/usr/bin/env python3
"""Shared builder for the MiniMax-M2.5 CK block-scale MoE GEMM GEAK harnesses.

Reconstructs the EXACT inputs the MiniMax-M2.5 (TP=2) MoE passes to the AITER CK
block-scale gridwise MoE GEMM, and exposes the identical production stage callables
(via aiter.fused_moe.get_2stage_cfgs) so the harness benchmarks the same kernel that
runs in run_sglang_test_minimax.sh.

Ground truth (captured from the real fused_moe path, see moe_captured_shapes.json):
  model_dim=3072, inter_dim=768 (=1536/2 per TP=2 rank), E=256, topk=8,
  activation=Silu, g1u1=True, dtype=fp16, q_dtype_a=q_dtype_w=fp8_e4m3fnuz,
  quant_type=per_1x128, doweight_stage1=False.
  fused_moe dispatch key: (cu_num=304, token, 3072, 768, 256, 8, Silu, fp16, fp8, fp8, per_1x128, True, False)

Per-stage CK GEMM args (token=2048 example):
  stage1 (ck_moe_stage1, InMemoryDataOp=Set, gate/up):
    a1            (token, 3072)        fp8_e4m3fnuz
    w1            (256, 1536, 3072)    fp8  (shuffled (16,16))
    w2            (256, 3072, 768)     fp8  (shuffled)
    sorted_ids    (token*topk + E*block_m - topk,) int32
    sorted_eids   (ceil(sorted_ids/block_m),)       int32
    num_valid_ids (2,) int32
    out=a2        (token, 8, 768)      fp16   <- stage1 output (intermediate)
    topk=8 ; a1_scale (token,24) fp32 ; w1_scale (256,288) fp32 ; block_m={16(decode)..64}
  stage2 (ck_moe_stage2_fwd, InMemoryDataOp=AtomicAdd, down):
    inter_states  (token, 8, 768)      fp8_e4m3fnuz
    w1,w2 as above (shuffled)
    sorted_* as above
    out           (token, 3072)        fp16   <- final MoE output
    topk=8 ; w2_scale (256,144) fp32 ; a2_scale (token,8,6) fp32 ; sorted_weights (token*topk+...) fp32
"""
import os
import sys

import torch
from einops import rearrange

REPO_ROOT = os.environ.get("GEAK_WORK_DIR", os.environ.get("GEAK_REPO_ROOT", "/sgl-workspace/aiter"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aiter
from aiter import dtypes, pertoken_quant, get_hip_quant
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight

# ---- MiniMax-M2.5 @ TP=2 MoE config (exact) ----
MODEL_DIM = 3072
INTER_DIM = 768          # 1536 / 2  (per TP rank)
E = 256
TOPK = 8
SCALE_BLK = (128, 128)
DTYPE = torch.float16
QDTYPE_A = dtypes.fp8    # fp8_e4m3fnuz on MI300
QDTYPE_W = dtypes.fp8
QUANT_TYPE = aiter.QuantType.per_1x128
ACT = aiter.ActivationType.Silu
USE_G1U1 = True
DOWEIGHT_STAGE1 = False

# Arena workload regime: token-parallel MoE GEMM, M = B*1024 (prefill token count) for
# concurrency B in {2,32,64} at seqlen 1024 -> tokens {2048, 32768, 65536}.
# Override via GEAK_ALL_TOKENS="t1,t2,..." (comma-separated).
# (Historical real per-forward counts were 16/256/2048/4096/8192/11264/16384/17920.)
_DEFAULT_ALL_TOKENS = [2048, 32768, 65536]
_env_tokens = os.environ.get("GEAK_ALL_TOKENS", "")
if _env_tokens.strip():
    ALL_TOKENS = [int(x) for x in _env_tokens.split(",") if x.strip()]
else:
    ALL_TOKENS = list(_DEFAULT_ALL_TOKENS)


def _block_quant_weight(w):
    """Per (128,128) block FP8 quant of a weight (E,N,K) -> (wq fp8 (E,N,K), ws fp32 (E, N//128*K//128))."""
    n, k = w.shape[1], w.shape[2]
    bn, bk = SCALE_BLK
    tmp = rearrange(
        w.view(-1, n // bn, bn, k // bk, bk),
        "e nb pn kb pk -> e nb kb (pn pk)",
    ).contiguous()
    wq, ws = pertoken_quant(tmp, quant_dtype=QDTYPE_W)
    wq = rearrange(
        wq.view(-1, n // bn, k // bk, bn, bk),
        "e nb kb pn pk -> e (nb pn) (kb pk)",
    ).contiguous()
    return wq, ws.view(E, -1)


def build_base(token, seed=0):
    """Build the routing + weights once. Returns a dict of all tensors for both stages."""
    torch.manual_seed(seed)
    dev = "cuda"
    inp = torch.randn((token, MODEL_DIM), dtype=DTYPE, device=dev)
    w1 = torch.randn((E, INTER_DIM * 2, MODEL_DIM), dtype=DTYPE, device=dev) / 10  # gate+up
    w2 = torch.randn((E, MODEL_DIM, INTER_DIM), dtype=DTYPE, device=dev) / 10       # down
    score = torch.randn((token, E), dtype=DTYPE, device=dev)
    topk_w, topk_ids = fused_topk(inp, score, TOPK, True)

    w1_qt, w1_scale = _block_quant_weight(w1)   # unshuffled (for torch ref)
    w2_qt, w2_scale = _block_quant_weight(w2)
    w1q = shuffle_weight(w1_qt, layout=(16, 16))  # shuffled (for CK kernel)
    w2q = shuffle_weight(w2_qt, layout=(16, 16))

    metadata = get_2stage_cfgs(
        get_padded_M(token), MODEL_DIM, INTER_DIM, E, TOPK, DTYPE,
        QDTYPE_A, QDTYPE_W, QUANT_TYPE, USE_G1U1, ACT, DOWEIGHT_STAGE1, 0, 0,
        is_shuffled=True,
    )
    block_m = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = moe_sorting(
        topk_ids, topk_w, E, MODEL_DIM, DTYPE, block_size=block_m
    )

    return {
        "token": token, "dev": dev, "metadata": metadata, "block_m": block_m,
        "inp": inp, "topk_w": topk_w, "topk_ids": topk_ids,
        "w1_qt": w1_qt, "w2_qt": w2_qt, "w1_scale": w1_scale, "w2_scale": w2_scale,
        "w1q": w1q, "w2q": w2q,
        "sorted_ids": sorted_ids, "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids, "num_valid_ids": num_valid_ids,
    }


def quant_a1(inp):
    qf = get_hip_quant(QUANT_TYPE)
    a1, a1_scale = qf(inp, quant_dtype=QDTYPE_A)
    return a1, a1_scale
