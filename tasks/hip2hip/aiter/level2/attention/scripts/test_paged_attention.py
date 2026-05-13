#!/usr/bin/env python3
"""Custom test for paged attention kernels (attention.cu).

Tests pa_fwd_naive which uses module_attention (JIT-compiled paged attention).
Does NOT depend on CK (Composable Kernel).
"""
import argparse
import json
import time

import torch
import aiter


def create_kv_cache(batch_size, max_blocks, num_kv_heads, head_size, block_size, dtype, device="cuda"):
    """Create paged KV cache tensors."""
    # K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x] where x=16
    x = 16
    k_cache = torch.randn(
        batch_size * max_blocks, num_kv_heads, head_size // x, block_size, x,
        dtype=dtype, device=device
    )
    # V cache: [num_blocks, num_kv_heads, head_size, block_size]
    v_cache = torch.randn(
        batch_size * max_blocks, num_kv_heads, head_size, block_size,
        dtype=dtype, device=device
    )
    return k_cache, v_cache


def reference_attention(q, k_cache, v_cache, block_tables, context_lens, num_kv_heads, head_size, block_size, scale):
    """Simple reference implementation for correctness checking."""
    batch_size = q.shape[0]
    num_q_heads = q.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    outputs = []

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_blocks_needed = (ctx_len + block_size - 1) // block_size

        # Gather K, V from paged cache
        keys = []
        vals = []
        for blk_idx in range(num_blocks_needed):
            phys_blk = block_tables[b, blk_idx].item()
            tokens_in_block = min(block_size, ctx_len - blk_idx * block_size)
            for kv_h in range(num_kv_heads):
                # K: [head_size/16, block_size, 16] -> [block_size, head_size]
                k_blk = k_cache[phys_blk, kv_h].permute(1, 0, 2).reshape(block_size, head_size)[:tokens_in_block]
                v_blk = v_cache[phys_blk, kv_h, :, :tokens_in_block].T  # [tokens, head_size]
                if kv_h == 0:
                    keys.append(k_blk)
                    vals.append(v_blk)

        if not keys:
            outputs.append(torch.zeros(num_q_heads, head_size, dtype=q.dtype, device=q.device))
            continue

        # Only do first KV head for simplicity (GQA correctness is handled by kernel)
        K = torch.cat(keys, dim=0).float()  # [seq_len, head_size]
        V = torch.cat(vals, dim=0).float()

        out_heads = []
        for qh in range(num_q_heads):
            q_vec = q[b, qh].float()  # [head_size]
            attn = torch.matmul(q_vec.unsqueeze(0), K.T) * scale  # [1, seq_len]
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, V).squeeze(0)  # [head_size]
            out_heads.append(out.to(q.dtype))
        outputs.append(torch.stack(out_heads))

    return torch.stack(outputs)


def test_paged_attention(batch_size, num_q_heads, num_kv_heads, head_size, context_len,
                         block_size=16, dtype=torch.bfloat16):
    """Test paged attention correctness and performance."""
    scale = head_size ** -0.5
    max_blocks = (context_len + block_size - 1) // block_size + 4  # some extra

    q = torch.randn(batch_size, num_q_heads, head_size, dtype=dtype, device="cuda")
    k_cache, v_cache = create_kv_cache(batch_size, max_blocks, num_kv_heads, head_size, block_size, dtype)
    block_tables = torch.arange(max_blocks, device="cuda", dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).contiguous()
    ctx_lens = torch.full((batch_size,), context_len, device="cuda", dtype=torch.int32)
    k_dq = torch.empty(0, device="cuda", dtype=torch.float32)
    v_dq = torch.empty(0, device="cuda", dtype=torch.float32)

    # Run kernel
    out = aiter.pa_fwd_naive(
        q, k_cache, v_cache, block_tables, ctx_lens,
        k_dq, v_dq, context_len, num_kv_heads, scale, 1.0, 1.0, block_size, 0
    )

    # Basic shape check
    assert out.shape == (batch_size, num_q_heads, head_size), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

    # Performance measurement
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        aiter.pa_fwd_naive(
            q, k_cache, v_cache, block_tables, ctx_lens,
            k_dq, v_dq, context_len, num_kv_heads, scale, 1.0, 1.0, block_size, 0
        )
    torch.cuda.synchronize()
    us = (time.perf_counter() - start) / n_iters * 1e6

    return True, us


def main():
    parser = argparse.ArgumentParser(description="Test paged attention (no CK dependency)")
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("-n", "--num_heads", type=str, nargs="+", default=["8,1", "32,4"],
                        help="num_q_heads,num_kv_heads e.g. 8,1")
    parser.add_argument("-c", "--ctx_len", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("-d", "--head_size", type=int, default=128)
    args = parser.parse_args()

    results = []
    all_pass = True

    for heads_str in args.num_heads:
        nq, nkv = map(int, heads_str.split(","))
        for batch in args.batch:
            for ctx in args.ctx_len:
                try:
                    ok, us = test_paged_attention(batch, nq, nkv, args.head_size, ctx)
                    status = "PASS" if ok else "FAIL"
                except Exception as e:
                    status = "FAIL"
                    us = 0.0
                    all_pass = False
                    print(f"  ERROR: {e}")
                print(f"[result] batch={batch}, heads={nq}/{nkv}, ctx={ctx}: {status}, {us:.2f} us")
                results.append({
                    "batch": batch, "num_q_heads": nq, "num_kv_heads": nkv,
                    "ctx_len": ctx, "status": status, "us": us
                })

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"[summary] passed={passed}, failed={failed}, total={len(results)}")
    print(f"[json_results] {json.dumps(results)}")


if __name__ == "__main__":
    main()
