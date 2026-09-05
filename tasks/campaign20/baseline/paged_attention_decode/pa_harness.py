#!/usr/bin/env python3
"""Shared decode paged-attention golden harness — MiniMax-M2.5 (TP=2 per-GPU).

Profiled/correctness-gated template: <bf16, bf16, kAuto, bf16>
  num_seqs=64 (decode batch), ctx_len=2048, q_heads=24, kv_heads=4 (GQA 6),
  head_dim=128, block_size=16, bf16 KV.
(fp8 KV is a timing-only variant — the torch golden does not dequantize fp8.)

Golden = run_torch (per-seq masked attention, fp32 softmax). Used by all three K11
rewrite drivers (CK / FlyDSL / Triton) so candidates are scored vs the SAME reference.
"""
import random
import torch
from einops import rearrange

# MiniMax-M2.5 decode shape (TP=2 per GPU)
NUM_SEQS = 64
CTX_LEN = 2048
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_SIZE = 128
BLOCK_SIZE = 16
PARTITION_SIZE = 256
SNR_THRESH = 30.0
UNIFORM = (-1, 1)


def ref_masked_attention(query, key, value, scale):
    attn = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn = torch.softmax(attn, dim=-1).to(value.dtype)
    return torch.einsum("hqk,khd->qhd", attn, value)


def run_torch(query, key_cache, value_cache, block_tables, seq_lens, num_kv_heads, scale):
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = query.shape[0]
    nqpkv = num_query_heads // num_kv_heads
    bt = block_tables.cpu().tolist()
    sl = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        seq_len = int(sl[i])
        ks, vs = [], []
        for j in range(seq_len):
            bn = int(bt[i][j // block_size]); bo = j % block_size
            ks.append(key_cache[bn, :, bo, :].reshape(num_kv_heads, head_size))
            vs.append(value_cache[bn, :, bo, :])
        keys = torch.stack(ks, dim=0); values = torch.stack(vs, dim=0)
        if nqpkv > 1:
            keys = torch.repeat_interleave(keys, nqpkv, dim=1)
            values = torch.repeat_interleave(values, nqpkv, dim=1)
        out = ref_masked_attention(q, keys, values, scale).view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output


def prepare(num_seqs=NUM_SEQS, ctx_len=CTX_LEN, dtype=torch.bfloat16, device="cuda:0"):
    torch.manual_seed(0); random.seed(0)
    nq, nkv, hd, bs = NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE
    max_seq_len = ctx_len
    max_blocks = (max_seq_len + bs - 1) // bs
    num_blocks = max_blocks * num_seqs
    scale = float(1.0 / (hd ** 0.5))
    query = torch.empty(num_seqs, nq, hd, dtype=dtype, device=device).uniform_(*UNIFORM)
    x = 16 // dtype.itemsize
    key_cache = torch.empty(num_blocks, nkv, hd // x, bs, x, dtype=dtype, device=device).uniform_(*UNIFORM)
    value_cache = torch.empty(num_blocks, nkv, hd, bs, dtype=dtype, device=device).uniform_(*UNIFORM)
    block_tables = rearrange(torch.randperm(num_blocks, dtype=torch.int32, device=device),
                             "(b n) -> b n", b=num_seqs)
    seq_lens = torch.full((num_seqs,), max_seq_len, dtype=torch.int, device=device)
    cu_query_lens = torch.arange(0, num_seqs + 1, dtype=torch.int, device=device)
    # BHSD layout [num_blks, num_kv_heads, kv_blk_sz, head_sz]; MUST be contiguous
    # (the Triton decode-PA value indexing assumes contiguous value cache).
    key_cache_bhsd = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)").contiguous()
    value_cache_bhsd = rearrange(value_cache, "b h d s -> b h s d").contiguous()
    # NHD layout for the aiter kernel
    kc_nhd = rearrange(key_cache_bhsd, "b h s d -> b s h d").contiguous()
    vc_nhd = rearrange(value_cache_bhsd, "b h s d -> b s h d").contiguous()
    return dict(query=query, key_cache=key_cache, value_cache=value_cache,
                key_cache_bhsd=key_cache_bhsd, value_cache_bhsd=value_cache_bhsd,
                kc_nhd=kc_nhd, vc_nhd=vc_nhd, block_tables=block_tables,
                seq_lens=seq_lens, cu_query_lens=cu_query_lens, max_seq_len=max_seq_len,
                scale=scale, num_kv_heads=nkv, num_q_heads=nq, head_size=hd,
                block_size=bs, num_seqs=num_seqs, dtype=dtype, device=device)


def golden(prep):
    return run_torch(prep["query"], prep["key_cache_bhsd"], prep["value_cache_bhsd"],
                     prep["block_tables"], prep["seq_lens"], prep["num_kv_heads"], prep["scale"])


def snr_db(ref, out):
    diff = (out.float() - ref.float())
    num = diff.pow(2).mean().item()
    den = ref.float().pow(2).mean().item()
    return 10.0 * torch.log10(torch.tensor(den / (num + 1e-20))).item()


def cosine_diff(ref, out):
    r = ref.float().reshape(-1); o = out.float().reshape(-1)
    return 1.0 - torch.nn.functional.cosine_similarity(r, o, dim=0).item()
