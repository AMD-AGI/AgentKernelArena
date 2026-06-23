#!/usr/bin/env python3
"""Real launcher + benchmark harness for kernel_unified_attention_2d (Triton, decode regime).

Builds semantically-valid paged-attention inputs at the workload regime
(batch B in {2,32,64}; q_len=1 decode; kv/context len=1024), launches the raw
@triton.jit kernel with a recomputed grid, runs golden-vs-editable correctness,
and times each case with CUDA events (10 warmup + 100 timed).

The kernel source references module-level @triton.jit helpers (find_seq_idx,
cdiv_fn, apply_softcap) that were NOT captured by inspect.getsource. We inject
frozen copies of those helpers (from aiter) into the loaded module's globals so
the JIT can resolve them, WITHOUT modifying the kernel source file.
"""
import os
import importlib.util

import torch
import triton
import triton.language as tl

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_kernel_unified_attention_2d.py")
GOLDEN_FILE = os.path.join(TASK_DIR, "source_golden", "triton_kernel_unified_attention_2d.py")
KERNEL_NAME = "kernel_unified_attention_2d"

# ----------------------------------------------------------------------------
# Captured model dims (from canonical/test_cases.json base case) -- kept fixed.
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
NUM_QUERIES_PER_KV = NUM_QUERY_HEADS // NUM_KV_HEADS  # 8
HEAD_SIZE = 64
HEAD_SIZE_PADDED = 64
BLOCK_SIZE = 64          # paged kv block size
TILE_SIZE = 64
SCALE = HEAD_SIZE ** -0.5
SLIDING_WINDOW = 0       # disable to attend full context (regime: full 1024 ctx)
USE_SINKS = False
USE_FP8 = False

# Regime
SEQLEN = 1024            # kv/context length (decode: q_len=1)
CONCURRENCIES = [2, 32, 64]
CASE_IDS = {2: "c2", 32: "c32", 64: "c64"}


# ----------------------------------------------------------------------------
# Frozen @triton.jit helpers (the kernel resolves these from module globals).
@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs,
                  BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("uak_" + os.path.basename(os.path.dirname(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Inject the missing @triton.jit helpers into the module's globals so the
    # JITFunction can resolve them at compile time.
    mod.find_seq_idx = _find_seq_idx
    mod.cdiv_fn = _cdiv_fn
    mod.apply_softcap = _apply_softcap
    kern = getattr(mod, KERNEL_NAME)
    # JITFunction caches its resolved globals; patch directly too.
    try:
        kern.__globals__["find_seq_idx"] = _find_seq_idx
        kern.__globals__["cdiv_fn"] = _cdiv_fn
        kern.__globals__["apply_softcap"] = _apply_softcap
    except Exception:
        pass
    return kern


# ----------------------------------------------------------------------------
def build_inputs(B, seed=42, dtype=torch.bfloat16, device="cuda"):
    """Decode regime: B sequences, each q_len=1, context/kv len = SEQLEN.

    Returns (args_dict, output_tensor, grid, launch_meta).
    """
    g = torch.Generator(device=device).manual_seed(seed)

    num_tokens = B  # one query token per seq (decode)
    q_len = 1
    kv_len = SEQLEN

    # BLOCK_M / BLOCK_Q per aiter select rule for decode:
    BLOCK_M = 16 if NUM_QUERIES_PER_KV <= 16 else triton.next_power_of_2(NUM_QUERIES_PER_KV)
    BLOCK_Q = max(1, BLOCK_M // NUM_QUERIES_PER_KV)

    # query: [num_tokens, num_query_heads, head_size]
    query = torch.randn(num_tokens, NUM_QUERY_HEADS, HEAD_SIZE,
                        generator=g, dtype=dtype, device=device)
    output = torch.empty_like(query)

    # paged kv cache: contiguous blocks, enough for B seqs * ceil(kv_len/BLOCK_SIZE)
    blocks_per_seq = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_num_blocks_per_seq = blocks_per_seq
    total_blocks = B * blocks_per_seq + 1
    key_cache = torch.randn(total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                            generator=g, dtype=dtype, device=device)
    value_cache = torch.randn(total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                              generator=g, dtype=dtype, device=device)

    # block_tables: [num_seqs, max_num_blocks_per_seq]
    bt = torch.arange(B * blocks_per_seq, dtype=torch.int32, device=device).reshape(B, blocks_per_seq)
    block_tables = bt.contiguous()

    # seq_lens (context+query length per seq) = kv_len
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device=device)

    # cu_seqlens_q (query_start_len_ptr): prefix sum of query lens, length num_seqs+1
    query_start_len = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=device)

    scales = torch.ones(1, NUM_KV_HEADS, dtype=torch.float32, device=device)

    grid = (NUM_KV_HEADS, num_tokens // BLOCK_Q + B)

    args = dict(
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=SCALE,
        k_scale=scales,
        v_scale=scales,
        out_scale=1.0,
        softcap=0.0,
        num_query_heads=NUM_QUERY_HEADS,
        num_queries_per_kv=NUM_QUERIES_PER_KV,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=USE_SINKS,
        SLIDING_WINDOW=SLIDING_WINDOW,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        query_start_len_ptr=query_start_len,
        BLOCK_Q=BLOCK_Q,
        num_seqs=B,
        BLOCK_M=BLOCK_M,
        USE_FP8=USE_FP8,
        ALL_DECODE=True,
    )
    meta = dict(num_warps=2, num_stages=3, waves_per_eu=2)
    return args, output, grid, meta


def launch(kern, B, seed=42):
    args, output, grid, meta = build_inputs(B, seed=seed)
    kern[grid](**args, **meta)
    torch.cuda.synchronize()
    return output


# ----------------------------------------------------------------------------
def _cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _max_rel_err(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().clamp_min(1e-4)
    return ((a - b).abs() / denom).max().item()


def run_correctness():
    kern_edit = _load_kernel(SOURCE_FILE)
    kern_gold = _load_kernel(GOLDEN_FILE)
    for B in CONCURRENCIES:
        out_edit = launch(kern_edit, B, seed=42)
        out_gold = launch(kern_gold, B, seed=42)
        cos = _cosine(out_edit, out_gold)
        mre = _max_rel_err(out_edit, out_gold)
        if not (cos >= 0.99 and mre < 1e-2):
            return False, f"B={B}: cosine={cos:.5f} max_rel_err={mre:.5f}"
    return True, None


def run_performance(n_warmup=10, n_iter=100):
    kern = _load_kernel(SOURCE_FILE)
    results = []
    for B in CONCURRENCIES:
        args, output, grid, meta = build_inputs(B, seed=42)
        for _ in range(n_warmup):
            kern[grid](**args, **meta)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record()
            kern[grid](**args, **meta)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        results.append({
            "test_case_id": CASE_IDS[B],
            "execution_time_ms": avg,
            "params": {"B": B, "q_len": 1, "kv_len": SEQLEN,
                       "num_query_heads": NUM_QUERY_HEADS, "head_size": HEAD_SIZE},
        })
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
    else:
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else f"FAIL {err}")
