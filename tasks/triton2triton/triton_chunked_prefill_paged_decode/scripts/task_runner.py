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
PERF_SHAPE_IDX = 2


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

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    num_seqs, slk, nqh, nkvh, hs, bs, xf = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale = \
        make_test_data(num_seqs, slk, nqh, nkvh, hs, bs, xf, device, dtype)

    for _ in range(5):
        mod.chunked_prefill_paged_decode(
            query, output, key_cache, value_cache, block_table,
            seq_lens, qsl, scale, filter_by_query_len=False,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.chunked_prefill_paged_decode(
            query, output, key_cache, value_cache, block_table,
            seq_lens, qsl, scale, filter_by_query_len=False,
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        elapsed_ms = run_performance()
        shape = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "num_seqs": shape[0], "seq_len_k": shape[1],
                "num_query_heads": shape[2], "num_kv_heads": shape[3],
                "head_size": shape[4], "block_size": shape[5],
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
