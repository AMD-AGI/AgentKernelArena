#!/usr/bin/env python3
"""Task runner for triton2triton/triton_paged_prefix_prefill"""
import sys
import os
import json
import argparse
import time
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_paged_prefix_prefill"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_paged_prefix_prefill.py")

# Test configurations:
# (batch_size, ctx_len, query_len, num_heads, num_kv_heads, head_dim, block_size)
TEST_SHAPES = [
    (1, 64, 32, 8, 8, 64, 16),      # small, MHA
    (2, 128, 64, 16, 4, 64, 16),     # medium, GQA
    (1, 256, 128, 32, 8, 128, 32),   # large, GQA
    (2, 512, 64, 16, 16, 64, 32),    # long ctx, MHA
    (4, 64, 32, 8, 1, 64, 16),       # batched, MQA
]
PERF_SHAPE_IDX = 2


def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_paged_kv_cache(
    batch_size, ctx_len, num_kv_heads, head_dim, block_size, device, dtype
):
    """
    Create a paged KV cache and populate it with random data.
    Returns: k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx

    full_k_ctx and full_v_ctx are the dense context K/V for reference comparison.
    """
    import torch

    x = 8  # vectorization factor
    assert head_dim % x == 0

    num_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq + 4  # extra padding blocks

    # 5D K cache: [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    k_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim // x, block_size, x,
        device=device, dtype=dtype,
    )
    # 4D V cache: [num_blocks, num_kv_heads, head_dim, block_size]
    v_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim, block_size,
        device=device, dtype=dtype,
    )

    # Block table: [batch, max_num_blocks]
    b_loc = torch.zeros(
        batch_size, num_blocks_per_seq, device=device, dtype=torch.int32
    )

    # Dense context K/V for reference: [batch, ctx_len, num_kv_heads, head_dim]
    full_k_ctx = torch.randn(
        batch_size, ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    full_v_ctx = torch.randn(
        batch_size, ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    # Populate the paged cache
    block_idx = 0
    for b in range(batch_size):
        for blk in range(num_blocks_per_seq):
            # Assign physical block
            b_loc[b, blk] = block_idx
            start_pos = blk * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            length = end_pos - start_pos

            for h in range(num_kv_heads):
                # Fill K cache (5D layout)
                k_vals = full_k_ctx[b, start_pos:end_pos, h, :]  # [length, head_dim]
                for t in range(length):
                    for d in range(head_dim):
                        d_outer = d // x
                        d_inner = d % x
                        k_cache[block_idx, h, d_outer, t, d_inner] = k_vals[t, d]

                # Fill V cache (4D layout)
                v_vals = full_v_ctx[b, start_pos:end_pos, h, :]  # [length, head_dim]
                for t in range(length):
                    for d in range(head_dim):
                        v_cache[block_idx, h, d, t] = v_vals[t, d]

            block_idx += 1

    return k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx


def reference_paged_attention(
    q_packed, k_new_packed, v_new_packed,
    full_k_ctx, full_v_ctx,
    b_start_loc, b_seq_len,
    batch_size, ctx_len, query_len,
    num_heads, num_kv_heads, head_dim,
):
    """
    CPU/PyTorch reference for paged prefix prefill attention.

    Phase 1: query attends to context (non-causal)
    Phase 2: query attends to query (causal)
    Combined with online softmax.
    """
    import torch

    kv_group_num = num_heads // num_kv_heads
    out = torch.zeros_like(q_packed)
    sm_scale = 1.0 / (head_dim ** 0.5)

    for b in range(batch_size):
        start = b_start_loc[b].item()
        total_len = b_seq_len[b].item()
        q_len = query_len

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_b = q_packed[start:start + q_len, h, :]  # [Q, D]

            # Context K/V
            k_ctx = full_k_ctx[b, :ctx_len, kv_h, :]  # [C, D]
            v_ctx = full_v_ctx[b, :ctx_len, kv_h, :]  # [C, D]

            # New K/V
            k_new = k_new_packed[start:start + q_len, kv_h, :]  # [Q, D]
            v_new = v_new_packed[start:start + q_len, kv_h, :]  # [Q, D]

            # Concatenate full K/V: context + new
            k_full = torch.cat([k_ctx, k_new], dim=0)  # [C+Q, D]
            v_full = torch.cat([v_ctx, v_new], dim=0)  # [C+Q, D]

            # Compute attention scores
            scores = (q_b @ k_full.T) * sm_scale  # [Q, C+Q]

            # Mask: no mask for context part, causal for query part
            S = scores.shape[1]
            mask = torch.zeros(q_len, S, device=scores.device, dtype=torch.bool)
            # Causal mask for query-vs-query part (last q_len columns)
            for qi in range(q_len):
                for ki in range(q_len):
                    if ki > qi:
                        mask[qi, ctx_len + ki] = True

            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores.float(), dim=-1).to(q_b.dtype)
            out[start:start + q_len, h, :] = attn @ v_full

    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "context_attention_fwd"), "Missing context_attention_fwd"
        assert hasattr(mod, "_fwd_kernel"), "Missing _fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    dtype = torch.float16

    for i, (bs, ctx_len, q_len, nh, nkv, hd, blk_sz) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            total_tokens = bs * q_len

            # Create query and new K/V (packed)
            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)

            # Create paged KV cache
            k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx = setup_paged_kv_cache(
                bs, ctx_len, nkv, hd, blk_sz, device, dtype
            )

            # b_start_loc has length batch+1 (cumulative)
            b_start_loc = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            for j in range(bs):
                b_start_loc[j] = j * q_len
            b_start_loc[bs] = bs * q_len

            # b_seq_len = ctx_len + q_len for each batch
            b_seq_len = torch.full(
                (bs,), ctx_len + q_len, device=device, dtype=torch.int32
            )

            # Run Triton kernel
            mod.context_attention_fwd(
                q, k_new, v_new, o,
                k_cache, v_cache, b_loc,
                b_start_loc, b_seq_len,
                max_input_len=q_len,
            )
            torch.cuda.synchronize()

            # Run reference
            ref = reference_paged_attention(
                q, k_new, v_new,
                full_k_ctx, full_v_ctx,
                b_start_loc, b_seq_len,
                bs, ctx_len, q_len,
                nh, nkv, hd,
            )

            if not torch.allclose(o, ref, atol=1e-2, rtol=1e-2):
                max_diff = (o - ref).abs().max().item()
                return False, (
                    f"Shape {i + 1} (bs={bs}, ctx={ctx_len}, q={q_len}, "
                    f"nh={nh}, nkv={nkv}, hd={hd}, blk={blk_sz}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i + 1} (bs={bs}, ctx={ctx_len}, q={q_len}, "
                f"nh={nh}, nkv={nkv}, hd={hd}, blk={blk_sz}): "
                f"exception: {e}"
            )

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    bs, ctx_len, q_len, nh, nkv, hd, blk_sz = TEST_SHAPES[PERF_SHAPE_IDX]
    total_tokens = bs * q_len

    torch.manual_seed(0)
    q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
    k_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
    v_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
    o = torch.zeros_like(q)

    k_cache, v_cache, b_loc, _, _ = setup_paged_kv_cache(
        bs, ctx_len, nkv, hd, blk_sz, device, dtype
    )

    b_start_loc = torch.zeros(bs + 1, device=device, dtype=torch.int32)
    for j in range(bs):
        b_start_loc[j] = j * q_len
    b_start_loc[bs] = bs * q_len
    b_seq_len = torch.full((bs,), ctx_len + q_len, device=device, dtype=torch.int32)

    # Warmup
    for _ in range(5):
        mod.context_attention_fwd(
            q, k_new, v_new, o,
            k_cache, v_cache, b_loc,
            b_start_loc, b_seq_len,
            max_input_len=q_len,
        )
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.context_attention_fwd(
            q, k_new, v_new, o,
            k_cache, v_cache, b_loc,
            b_start_loc, b_seq_len,
            max_input_len=q_len,
        )
        end_events[j].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        elapsed_ms = run_performance()
        bs, ctx_len, q_len, nh, nkv, hd, blk_sz = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "batch_size": bs,
                "ctx_len": ctx_len,
                "query_len": q_len,
                "num_heads": nh,
                "num_kv_heads": nkv,
                "head_dim": hd,
                "block_size": blk_sz,
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
