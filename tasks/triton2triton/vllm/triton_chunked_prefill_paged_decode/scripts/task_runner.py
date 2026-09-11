#!/usr/bin/env python3
"""Task runner for triton2triton/triton_chunked_prefill_paged_decode"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_chunked_prefill_paged_decode"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_chunked_prefill_paged_decode.py")

# Test configs: (num_seqs, seq_len_k, num_query_heads, num_kv_heads, head_size, block_size, x_factor)
TEST_SHAPES = [
    (4, 64, 8, 8, 64, 16, 8),
    (2, 128, 16, 4, 64, 16, 8),
    (8, 256, 32, 8, 128, 16, 8),
    (4, 128, 16, 16, 64, 32, 8),
    (2, 512, 8, 8, 128, 32, 8),
]

# These cases exercise correctness-only features without changing the scored
# performance shapes above. Sequence and query lengths are per-sequence.
EXTRA_CORRECTNESS_CASES = [
    {
        "name": "unaligned_shuffled_strided_x4_head80",
        "seq_lens": [1, 17, 31],
        "query_lens": [1, 1, 1],
        "num_query_heads": 12,
        "num_kv_heads": 3,
        "head_size": 80,
        "block_size": 16,
        "x_factor": 4,
        "page_mapping": "shuffled",
        "strided_layout": True,
        "filter_by_query_len": False,
    },
    {
        "name": "mixed_query_filter_sliding_x16_head96",
        "seq_lens": [33, 70, 47, 95],
        "query_lens": [1, 3, 1, 2],
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_size": 96,
        "block_size": 32,
        "x_factor": 16,
        "page_mapping": "reversed",
        "strided_layout": False,
        "filter_by_query_len": True,
        "sliding_window": 17,
    },
    {
        "name": "unaligned_alibi_mha",
        "seq_lens": [23, 49],
        "query_lens": [1, 1],
        "num_query_heads": 4,
        "num_kv_heads": 4,
        "head_size": 64,
        "block_size": 16,
        "x_factor": 8,
        "page_mapping": "shuffled",
        "strided_layout": False,
        "filter_by_query_len": False,
        "use_alibi": True,
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


def make_test_data(num_seqs, seq_len_k, num_query_heads, num_kv_heads,
                   head_size, block_size, x_factor, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.float16

    # Each seq has 1 query token (decode)
    total_tokens = num_seqs
    query = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4

    # 5D K cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    key_cache = torch.randn(total_blocks, num_kv_heads, head_size // x_factor,
                            block_size, x_factor, device=device, dtype=dtype)
    # 4D V cache: [num_blocks, num_kv_heads, head_size, block_size]
    value_cache = torch.randn(total_blocks, num_kv_heads, head_size, block_size,
                              device=device, dtype=dtype)

    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    seq_lens = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)

    # query_start_loc for decode: each seq has 1 token
    query_start_loc = torch.arange(0, num_seqs + 1, device=device, dtype=torch.int32)

    output = torch.zeros_like(query)
    scale = 1.0 / (head_size ** 0.5)

    return query, output, key_cache, value_cache, block_table, seq_lens, query_start_loc, scale


def reference_attention(query, key_cache, value_cache, block_table, seq_lens,
                        scale, block_size, x_factor):
    """CPU reference for paged attention with 5D K / 4D V cache."""
    import torch
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    output = torch.zeros_like(query, dtype=torch.float32)

    for s in range(num_seqs):
        k_len = seq_lens[s].item()

        # Gather K from 5D cache
        k_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=query.device, dtype=query.dtype)
        v_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=query.device, dtype=query.dtype)

        for t in range(k_len):
            bi = t // block_size
            bo = t % block_size
            pb = block_table[s, bi].item()
            # K: [num_blocks, num_kv_heads, head_size//x, block_size, x]
            for kv_h in range(num_kv_heads):
                for d in range(head_size):
                    d_outer = d // x_factor
                    d_inner = d % x_factor
                    k_gathered[t, kv_h, d] = key_cache[pb, kv_h, d_outer, bo, d_inner]
                # V: [num_blocks, num_kv_heads, head_size, block_size]
                v_gathered[t, kv_h, :] = value_cache[pb, kv_h, :, bo]

        for h in range(num_query_heads):
            kv_h = h // num_queries_per_kv
            Q_h = query[s, h, :].float()
            K_h = k_gathered[:, kv_h, :].float()
            V_h = v_gathered[:, kv_h, :].float()

            S = (Q_h @ K_h.T) * scale
            # Decode: no causal mask needed beyond seq_len (already bounded)
            # Just ensure positions beyond seq_len are masked
            S_max = S.max()
            P = torch.exp(S - S_max)
            P = P / P.sum()
            output[s, h, :] = P @ V_h

    return output.to(query.dtype)


def make_extended_test_data(case, device="cuda", dtype=None):
    """Build feature-focused inputs, including non-contiguous padded views."""
    import torch
    if dtype is None:
        dtype = torch.float16

    seq_lens_list = case["seq_lens"]
    query_lens_list = case["query_lens"]
    num_seqs = len(seq_lens_list)
    num_query_heads = case["num_query_heads"]
    num_kv_heads = case["num_kv_heads"]
    head_size = case["head_size"]
    block_size = case["block_size"]
    x_factor = case["x_factor"]
    strided_layout = case["strided_layout"]

    assert len(query_lens_list) == num_seqs
    assert num_query_heads % num_kv_heads == 0
    assert head_size % x_factor == 0
    if not case["filter_by_query_len"]:
        assert all(length == 1 for length in query_lens_list)

    total_tokens = sum(query_lens_list)
    if strided_layout:
        query_storage = torch.randn(
            total_tokens, num_query_heads, head_size + 3,
            device=device, dtype=dtype,
        )
        query = query_storage[..., :head_size]
        output_storage = torch.full(
            (total_tokens, num_query_heads, head_size + 5), -7.0,
            device=device, dtype=dtype,
        )
        output = output_storage[..., :head_size]
    else:
        query = torch.randn(
            total_tokens, num_query_heads, head_size,
            device=device, dtype=dtype,
        )
        output = torch.full_like(query, -7.0)

    blocks_per_seq = [
        (length + block_size - 1) // block_size for length in seq_lens_list
    ]
    logical_block_count = sum(blocks_per_seq)
    total_blocks = logical_block_count + 4

    if strided_layout:
        key_storage = torch.randn(
            total_blocks, num_kv_heads, head_size // x_factor,
            block_size, x_factor + 3, device=device, dtype=dtype,
        )
        key_cache = key_storage[..., :x_factor]
        value_storage = torch.randn(
            total_blocks, num_kv_heads, head_size, block_size + 3,
            device=device, dtype=dtype,
        )
        value_cache = value_storage[..., :block_size]
        assert not query.is_contiguous()
        assert not output.is_contiguous()
        assert not key_cache.is_contiguous()
        assert not value_cache.is_contiguous()
    else:
        key_cache = torch.randn(
            total_blocks, num_kv_heads, head_size // x_factor,
            block_size, x_factor, device=device, dtype=dtype,
        )
        value_cache = torch.randn(
            total_blocks, num_kv_heads, head_size, block_size,
            device=device, dtype=dtype,
        )

    max_blocks_per_seq = max(blocks_per_seq)
    table_storage = torch.zeros(
        num_seqs, max_blocks_per_seq + 2,
        device=device, dtype=torch.int32,
    )
    block_table = table_storage[:, :max_blocks_per_seq]

    if case["page_mapping"] == "shuffled":
        physical_blocks = torch.randperm(total_blocks, device=device)
    elif case["page_mapping"] == "reversed":
        physical_blocks = torch.arange(
            total_blocks - 1, -1, -1, device=device,
        )
    else:
        raise ValueError(f"Unknown page mapping: {case['page_mapping']}")

    block_offset = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = physical_blocks[
            block_offset:block_offset + block_count
        ]
        block_offset += block_count

    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    query_lens = torch.tensor(query_lens_list, device=device, dtype=torch.int32)
    query_start_loc = torch.cat((
        torch.zeros(1, device=device, dtype=torch.int32),
        query_lens.cumsum(dim=0),
    ))
    scale = 1.0 / (head_size ** 0.5)

    alibi_slopes = None
    if case.get("use_alibi", False):
        alibi_slopes = torch.linspace(
            0.02, 0.16, num_query_heads, device=device, dtype=torch.float32,
        )

    return (
        query, output, key_cache, value_cache, block_table, seq_lens,
        query_start_loc, scale, alibi_slopes,
    )


def reference_extended_attention(
    query, initial_output, key_cache, value_cache, block_table, seq_lens,
    query_start_loc, scale, block_size, x_factor, filter_by_query_len,
    sliding_window=0, alibi_slopes=None,
):
    """Reference for filtered decode, sliding-window, and ALiBi modes."""
    import torch
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert key_cache.shape[2] * x_factor == head_size
    output = initial_output.float().clone()

    for seq_idx in range(num_seqs):
        query_start = int(query_start_loc[seq_idx].item())
        query_stop = int(query_start_loc[seq_idx + 1].item())
        if filter_by_query_len and query_stop - query_start > 1:
            continue
        query_idx = query_start if filter_by_query_len else seq_idx

        seq_len = int(seq_lens[seq_idx].item())
        token_offsets = torch.arange(seq_len, device=query.device)
        logical_blocks = token_offsets // block_size
        physical_blocks = block_table[seq_idx, logical_blocks].long()
        block_offsets = token_offsets % block_size

        gathered_k = key_cache[
            physical_blocks, :, :, block_offsets, :
        ].reshape(seq_len, num_kv_heads, head_size)
        gathered_v = value_cache[
            physical_blocks, :, :, block_offsets
        ]

        for query_head in range(num_query_heads):
            kv_head = query_head // num_queries_per_kv
            scores = (
                query[query_idx, query_head].float()
                @ gathered_k[:, kv_head].float().T
            ) * scale

            context_len = seq_len - 1
            if sliding_window > 0:
                scores = scores.masked_fill(
                    (context_len - token_offsets) >= sliding_window,
                    float("-inf"),
                )
            if alibi_slopes is not None:
                scores += alibi_slopes[query_head] * (
                    token_offsets - context_len
                )

            probabilities = torch.softmax(scores, dim=0)
            output[query_idx, query_head] = (
                probabilities @ gathered_v[:, kv_head].float()
            )

    return output.to(query.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "chunked_prefill_paged_decode"), "Missing chunked_prefill_paged_decode"
        assert hasattr(mod, "kernel_paged_attention_2d"), "Missing kernel_paged_attention_2d"
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

    for i, (num_seqs, slk, nqh, nkvh, hs, bs, xf) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale = \
                make_test_data(num_seqs, slk, nqh, nkvh, hs, bs, xf, device, dtype)

            mod.chunked_prefill_paged_decode(
                query, output, key_cache, value_cache, block_table,
                seq_lens, qsl, scale, filter_by_query_len=False,
            )
            torch.cuda.synchronize()

            ref = reference_attention(
                query, key_cache, value_cache, block_table,
                seq_lens, scale, bs, xf,
            )

            if not torch.allclose(output.float(), ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(EXTRA_CORRECTNESS_CASES):
        case_name = case["name"]
        try:
            torch.manual_seed(142 + i)
            (
                query, output, key_cache, value_cache, block_table, seq_lens,
                qsl, scale, alibi_slopes,
            ) = make_extended_test_data(case, device, dtype)
            initial_output = output.clone()

            mod.chunked_prefill_paged_decode(
                query, output, key_cache, value_cache, block_table,
                seq_lens, qsl, scale,
                alibi_slopes=alibi_slopes,
                sliding_window=case.get("sliding_window", 0),
                filter_by_query_len=case["filter_by_query_len"],
            )
            torch.cuda.synchronize()

            ref = reference_extended_attention(
                query, initial_output, key_cache, value_cache, block_table,
                seq_lens, qsl, scale, case["block_size"], case["x_factor"],
                case["filter_by_query_len"],
                sliding_window=case.get("sliding_window", 0),
                alibi_slopes=alibi_slopes,
            )

            if not torch.allclose(
                output.float(), ref.float(), atol=1e-2, rtol=1e-2
            ):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return False, f"Case {case_name}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Case {case_name}: exception: {e}"

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

    for test_idx, (num_seqs, slk, nqh, nkvh, hs, bs, xf) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale = \
                make_test_data(num_seqs, slk, nqh, nkvh, hs, bs, xf, device, dtype)

            def _bench_fn():
                mod.chunked_prefill_paged_decode(
                    query, output, key_cache, value_cache, block_table,
                    seq_lens, qsl, scale, filter_by_query_len=False,
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
                    "num_seqs": num_seqs,
                    "seq_len_k": slk,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "x_factor": xf
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_seqs": num_seqs,
                    "seq_len_k": slk,
                    "num_query_heads": nqh,
                    "num_kv_heads": nkvh,
                    "head_size": hs,
                    "block_size": bs,
                    "x_factor": xf
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
            "num_shapes": len(TEST_SHAPES) + len(EXTRA_CORRECTNESS_CASES),
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
