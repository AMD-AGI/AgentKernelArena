#!/usr/bin/env python3
"""Task runner for triton2triton/triton_paged_prefix_prefill_alibi"""
import sys
import os
import json
import argparse
import time
import importlib.util
import math

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_paged_prefix_prefill_alibi"
SOURCE_FILE = os.path.join(
    TASK_DIR, "source", "triton_paged_prefix_prefill_alibi.py"
)

# Test configurations:
# (batch_size, ctx_len, query_len, num_heads, num_kv_heads, head_dim, block_size)
TEST_SHAPES = [
    (1, 64, 32, 8, 8, 64, 16),      # small, MHA
    (2, 128, 64, 16, 4, 64, 16),     # medium, GQA
    (1, 256, 128, 32, 8, 128, 32),   # large, GQA
    (2, 512, 64, 16, 16, 64, 32),    # long ctx, MHA
    (4, 64, 32, 8, 1, 64, 16),       # batched, MQA
]

# Correctness-only cases that exercise packing and cache indirection without
# changing the shapes or methodology used for performance scoring.
ADDITIONAL_CORRECTNESS_CASES = [
    {
        "name": "ragged_partial_shuffled_explicit_scale",
        "context_lens": (17, 32, 47),
        "query_lens": (7, 129, 33),
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "block_size": 16,
        "sm_scale": 0.2,
    },
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_alibi_slopes(num_heads):
    """Generate ALiBi slopes for the given number of heads."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = [base ** (i + 1) for i in range(closest_power_of_2)]
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = [extra_base ** (2 * i + 1) for i in range(num_heads - closest_power_of_2)]
        powers = powers + extra_powers
    return powers[:num_heads]


def setup_paged_kv_cache(
    batch_size, ctx_len, num_kv_heads, head_dim, block_size, device, dtype
):
    """Create paged KV cache and populate with random data."""
    import torch

    x = 8
    assert head_dim % x == 0

    num_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq + 4

    k_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim // x, block_size, x,
        device=device, dtype=dtype,
    )
    v_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim, block_size,
        device=device, dtype=dtype,
    )
    b_loc = torch.zeros(
        batch_size, num_blocks_per_seq, device=device, dtype=torch.int32
    )

    full_k_ctx = torch.randn(
        batch_size, ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    full_v_ctx = torch.randn(
        batch_size, ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    block_idx = 0
    for b in range(batch_size):
        for blk in range(num_blocks_per_seq):
            b_loc[b, blk] = block_idx
            start_pos = blk * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            length = end_pos - start_pos

            for h in range(num_kv_heads):
                k_vals = full_k_ctx[b, start_pos:end_pos, h, :]
                for t in range(length):
                    for d in range(head_dim):
                        d_outer = d // x
                        d_inner = d % x
                        k_cache[block_idx, h, d_outer, t, d_inner] = k_vals[t, d]

                v_vals = full_v_ctx[b, start_pos:end_pos, h, :]
                for t in range(length):
                    for d in range(head_dim):
                        v_cache[block_idx, h, d, t] = v_vals[t, d]

            block_idx += 1

    return k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx


def setup_ragged_paged_kv_cache(
    context_lens, num_kv_heads, head_dim, block_size, device, dtype
):
    """Create a ragged cache with deterministic, gapped physical block IDs."""
    import torch

    batch_size = len(context_lens)
    x = 8
    assert head_dim % x == 0

    blocks_per_seq = [
        (ctx_len + block_size - 1) // block_size for ctx_len in context_lens
    ]
    num_mapped_blocks = sum(blocks_per_seq)
    max_blocks_per_seq = max(blocks_per_seq)

    # Odd IDs leave holes in the physical cache. Rotating their two halves
    # makes logical neighbors map to neither adjacent nor monotonic IDs.
    physical_block_ids = list(range(1, 2 * num_mapped_blocks, 2))
    split = (num_mapped_blocks + 1) // 2
    physical_block_ids = physical_block_ids[split:] + physical_block_ids[:split]
    total_blocks = 2 * num_mapped_blocks + 2

    k_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim // x, block_size, x,
        device=device, dtype=dtype,
    )
    v_cache = torch.zeros(
        total_blocks, num_kv_heads, head_dim, block_size,
        device=device, dtype=dtype,
    )
    b_loc = torch.zeros(
        batch_size, max_blocks_per_seq, device=device, dtype=torch.int32
    )

    max_context_len = max(context_lens)
    full_k_ctx = torch.zeros(
        batch_size, max_context_len, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    full_v_ctx = torch.zeros_like(full_k_ctx)

    mapping_idx = 0
    for batch_idx, ctx_len in enumerate(context_lens):
        full_k_ctx[batch_idx, :ctx_len] = torch.randn(
            ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        full_v_ctx[batch_idx, :ctx_len] = torch.randn(
            ctx_len, num_kv_heads, head_dim, device=device, dtype=dtype
        )

        for logical_block in range(blocks_per_seq[batch_idx]):
            physical_block = physical_block_ids[mapping_idx]
            mapping_idx += 1
            b_loc[batch_idx, logical_block] = physical_block

            start_pos = logical_block * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            length = end_pos - start_pos
            k_values = full_k_ctx[batch_idx, start_pos:end_pos]
            v_values = full_v_ctx[batch_idx, start_pos:end_pos]

            k_cache[physical_block, :, :, :length, :] = (
                k_values.permute(1, 2, 0)
                .reshape(num_kv_heads, head_dim // x, x, length)
                .permute(0, 1, 3, 2)
            )
            v_cache[physical_block, :, :, :length] = v_values.permute(1, 2, 0)

    return k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx


def reference_attention_alibi(
    q_packed, k_new_packed, v_new_packed,
    full_k_ctx, full_v_ctx,
    b_start_loc, b_seq_len,
    alibi_slopes,
    batch_size, ctx_len, query_len,
    num_heads, num_kv_heads, head_dim,
    sm_scale=None,
):
    """
    CPU/PyTorch reference for paged prefix prefill attention with ALiBi.
    """
    import torch

    kv_group_num = num_heads // num_kv_heads
    out = torch.zeros_like(q_packed)
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    context_lens = (
        [ctx_len] * batch_size if isinstance(ctx_len, int) else list(ctx_len)
    )
    query_lens = (
        [query_len] * batch_size
        if isinstance(query_len, int)
        else list(query_len)
    )

    for b in range(batch_size):
        start = b_start_loc[b].item()
        total_len = b_seq_len[b].item()
        ctx_len_b = context_lens[b]
        q_len = query_lens[b]
        assert total_len == ctx_len_b + q_len

        for h in range(num_heads):
            kv_h = h // kv_group_num
            slope = alibi_slopes[h].item()

            q_b = q_packed[start:start + q_len, h, :]

            k_ctx = full_k_ctx[b, :ctx_len_b, kv_h, :]
            v_ctx = full_v_ctx[b, :ctx_len_b, kv_h, :]
            k_new = k_new_packed[start:start + q_len, kv_h, :]
            v_new = v_new_packed[start:start + q_len, kv_h, :]

            k_full = torch.cat([k_ctx, k_new], dim=0)
            v_full = torch.cat([v_ctx, v_new], dim=0)

            scores = (q_b @ k_full.T) * sm_scale

            # ALiBi bias: slope * (key_pos - query_pos)
            # query positions: ctx_len, ctx_len+1, ..., ctx_len+q_len-1
            # key positions: 0, 1, ..., ctx_len+q_len-1
            S = scores.shape[1]
            q_positions = torch.arange(
                ctx_len_b, ctx_len_b + q_len, device=scores.device
            ).float()
            k_positions = torch.arange(0, S, device=scores.device).float()
            alibi_bias = slope * (k_positions[None, :] - q_positions[:, None])
            # Only negative biases (causal direction)
            alibi_bias = torch.where(alibi_bias <= 0, alibi_bias, torch.tensor(float("-inf")))

            # Causal mask for query-vs-query part
            for qi in range(q_len):
                for ki in range(q_len):
                    if ki > qi:
                        scores[qi, ctx_len_b + ki] = float("-inf")
                        alibi_bias[qi, ctx_len_b + ki] = float("-inf")

            scores = scores + alibi_bias
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
        assert hasattr(mod, "context_attention_fwd_alibi"), "Missing context_attention_fwd_alibi"
        assert hasattr(mod, "_fwd_kernel_alibi"), "Missing _fwd_kernel_alibi"
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

            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)

            k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx = setup_paged_kv_cache(
                bs, ctx_len, nkv, hd, blk_sz, device, dtype
            )

            b_start_loc = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            for j in range(bs):
                b_start_loc[j] = j * q_len
            b_start_loc[bs] = bs * q_len

            b_seq_len = torch.full(
                (bs,), ctx_len + q_len, device=device, dtype=torch.int32
            )

            slopes = get_alibi_slopes(nh)
            alibi_slopes = torch.tensor(slopes, device=device, dtype=torch.float32)

            mod.context_attention_fwd_alibi(
                q, k_new, v_new, o,
                k_cache, v_cache, b_loc,
                b_start_loc, b_seq_len,
                max_input_len=q_len,
                alibi_slopes=alibi_slopes,
            )
            torch.cuda.synchronize()

            ref = reference_attention_alibi(
                q, k_new, v_new,
                full_k_ctx, full_v_ctx,
                b_start_loc, b_seq_len,
                alibi_slopes,
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

    for i, case in enumerate(ADDITIONAL_CORRECTNESS_CASES):
        name = case["name"]
        context_lens = case["context_lens"]
        query_lens = case["query_lens"]
        nh = case["num_heads"]
        nkv = case["num_kv_heads"]
        hd = case["head_dim"]
        blk_sz = case["block_size"]
        sm_scale = case["sm_scale"]
        bs = len(context_lens)

        try:
            assert len(query_lens) == bs
            torch.manual_seed(1000 + i)
            total_tokens = sum(query_lens)

            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v_new = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)

            k_cache, v_cache, b_loc, full_k_ctx, full_v_ctx = (
                setup_ragged_paged_kv_cache(
                    context_lens, nkv, hd, blk_sz, device, dtype
                )
            )

            b_start_loc = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            b_start_loc[1:] = torch.tensor(
                query_lens, device=device, dtype=torch.int32
            ).cumsum(0)
            b_seq_len = torch.tensor(
                [ctx + query for ctx, query in zip(context_lens, query_lens)],
                device=device,
                dtype=torch.int32,
            )

            slopes = get_alibi_slopes(nh)
            alibi_slopes = torch.tensor(
                slopes, device=device, dtype=torch.float32
            )

            mod.context_attention_fwd_alibi(
                q, k_new, v_new, o,
                k_cache, v_cache, b_loc,
                b_start_loc, b_seq_len,
                max_input_len=max(query_lens),
                alibi_slopes=alibi_slopes,
                sm_scale=sm_scale,
            )
            torch.cuda.synchronize()

            ref = reference_attention_alibi(
                q, k_new, v_new,
                full_k_ctx, full_v_ctx,
                b_start_loc, b_seq_len,
                alibi_slopes,
                bs, context_lens, query_lens,
                nh, nkv, hd,
                sm_scale=sm_scale,
            )

            if not torch.allclose(o, ref, atol=1e-2, rtol=1e-2):
                max_diff = (o - ref).abs().max().item()
                return False, f"Case {name}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Case {name}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (bs, ctx_len, q_len, nh, nkv, hd, blk_sz) in enumerate(TEST_SHAPES):
        try:
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

            slopes = get_alibi_slopes(nh)
            alibi_slopes = torch.tensor(slopes, device=device, dtype=torch.float32)

            def _bench_fn():
                mod.context_attention_fwd_alibi(
                    q, k_new, v_new, o,
                    k_cache, v_cache, b_loc,
                    b_start_loc, b_seq_len,
                    max_input_len=q_len,
                    alibi_slopes=alibi_slopes,
                )
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch_size": bs,
                    "ctx_len": ctx_len,
                    "query_len": q_len,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd,
                    "block_size": blk_sz
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": bs,
                    "ctx_len": ctx_len,
                    "query_len": q_len,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd,
                    "block_size": blk_sz
                }
            })
    return test_cases


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
            "num_shapes": len(TEST_SHAPES) + len(ADDITIONAL_CORRECTNESS_CASES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
