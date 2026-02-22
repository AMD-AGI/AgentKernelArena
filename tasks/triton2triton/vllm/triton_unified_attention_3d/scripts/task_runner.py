#!/usr/bin/env python3
"""Task runner for triton2triton/triton_unified_attention_3d"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_unified_attention_3d"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_unified_attention_3d.py")

# Test configs: (num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads, head_size, block_size, num_segments)
TEST_SHAPES = [
    (2, 1, 128, 8, 8, 64, 16, 2),
    (1, 1, 256, 16, 4, 64, 16, 4),
    (4, 1, 512, 32, 8, 128, 16, 4),
    (2, 1, 128, 16, 16, 64, 32, 2),
    (1, 1, 1024, 8, 8, 128, 32, 4),
]
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(num_seqs, seq_len_q, seq_len_k, num_query_heads, num_kv_heads,
                   head_size, block_size, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.float16

    total_tokens = num_seqs * seq_len_q
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)

    num_blocks_per_seq = (seq_len_k + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                            device=device, dtype=dtype)
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size,
                              device=device, dtype=dtype)

    block_table = torch.zeros(num_seqs, num_blocks_per_seq, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b

    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    for s in range(num_seqs):
        cu_seqlens_q[s + 1] = cu_seqlens_q[s] + seq_len_q

    seqused_k = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)
    scale = 1.0 / (head_size ** 0.5)

    return q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale


def reference_attention_3d(q, key_cache, value_cache, block_table, cu_seqlens_q,
                           seqused_k, scale, block_size, num_segments):
    """CPU reference: compute per-segment partial attention outputs."""
    import torch
    import triton
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    num_kv_heads = key_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    total_tokens = q.shape[0]

    segm_output = torch.zeros(total_tokens, num_query_heads, num_segments, head_size_padded,
                              device=q.device, dtype=torch.float32)
    segm_max = torch.full((total_tokens, num_query_heads, num_segments),
                          float("-inf"), device=q.device, dtype=torch.float32)
    segm_expsum = torch.zeros(total_tokens, num_query_heads, num_segments,
                              device=q.device, dtype=torch.float32)

    TILE_SIZE = 16

    for s in range(num_seqs):
        q_start = cu_seqlens_q[s].item()
        q_end = cu_seqlens_q[s + 1].item()
        q_len = q_end - q_start
        k_len = seqused_k[s].item()

        # Gather K, V
        k_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=q.device, dtype=q.dtype)
        v_gathered = torch.zeros(k_len, num_kv_heads, head_size, device=q.device, dtype=q.dtype)
        for t in range(k_len):
            bi = t // block_size
            bo = t % block_size
            pb = block_table[s, bi].item()
            k_gathered[t] = key_cache[pb, bo]
            v_gathered[t] = value_cache[pb, bo]

        tiles_per_segment = (k_len + num_segments * TILE_SIZE - 1) // (num_segments * TILE_SIZE)
        context_len = k_len - q_len

        for qi_local in range(q_len):
            qi_global = q_start + qi_local
            for h in range(num_query_heads):
                kv_h = h // num_queries_per_kv
                Q_h = q[qi_global, h, :].float()

                for seg in range(num_segments):
                    tile_lo = seg * tiles_per_segment * TILE_SIZE
                    tile_hi = min((seg + 1) * tiles_per_segment * TILE_SIZE, k_len)
                    if tile_lo >= k_len:
                        continue

                    K_seg = k_gathered[tile_lo:tile_hi, kv_h, :].float()
                    V_seg = v_gathered[tile_lo:tile_hi, kv_h, :].float()

                    scores = (Q_h @ K_seg.T) * scale

                    # Causal mask
                    for ki in range(scores.shape[0]):
                        abs_ki = tile_lo + ki
                        if abs_ki > context_len + qi_local:
                            scores[ki] = float("-inf")

                    max_s = scores.max().item()
                    if max_s == float("-inf"):
                        continue
                    exp_s = torch.exp(scores - max_s)
                    sum_exp = exp_s.sum().item()

                    out_seg = (exp_s @ V_seg)  # [head_size]

                    segm_output[qi_global, h, seg, :head_size] = out_seg
                    segm_max[qi_global, h, seg] = max_s
                    segm_expsum[qi_global, h, seg] = sum_exp

    return segm_output, segm_max, segm_expsum


def reduce_segments_ref(segm_output, segm_max, segm_expsum, head_size):
    """Reduce segment partials to final output using logsumexp."""
    import torch
    total_tokens = segm_output.shape[0]
    num_heads = segm_output.shape[1]
    output = torch.zeros(total_tokens, num_heads, head_size, device=segm_output.device, dtype=torch.float32)

    overall_max = segm_max.max(dim=-1).values  # [tokens, heads]
    rescaled_expsum = segm_expsum * torch.exp(segm_max - overall_max.unsqueeze(-1))
    overall_expsum = rescaled_expsum.sum(dim=-1)  # [tokens, heads]

    rescaled_output = segm_output * torch.exp(segm_max - overall_max.unsqueeze(-1)).unsqueeze(-1)
    summed = rescaled_output.sum(dim=2)  # [tokens, heads, head_size_padded]

    safe_denom = overall_expsum.unsqueeze(-1)
    safe_denom = torch.where(safe_denom == 0, torch.ones_like(safe_denom), safe_denom)
    output = summed[:, :, :head_size] / safe_denom

    return output


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "unified_attention_3d"), "Missing unified_attention_3d"
        assert hasattr(mod, "kernel_unified_attention_3d"), "Missing kernel_unified_attention_3d"
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

    for i, (num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, nseg) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale = \
                make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

            segm_out, segm_max_out, segm_expsum_out = mod.unified_attention_3d(
                q, key_cache, value_cache, block_table,
                cu_seqlens_q, seqused_k, scale, num_segments=nseg,
            )
            torch.cuda.synchronize()

            # Reduce to final output
            result = reduce_segments_ref(segm_out, segm_max_out, segm_expsum_out, hs)

            # Reference: full attention
            ref_segm_out, ref_segm_max, ref_segm_expsum = reference_attention_3d(
                q, key_cache, value_cache, block_table,
                cu_seqlens_q, seqused_k, scale, bs, nseg,
            )
            ref = reduce_segments_ref(ref_segm_out, ref_segm_max, ref_segm_expsum, hs)

            if not torch.allclose(result.float(), ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
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
    num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, nseg = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    q, key_cache, value_cache, block_table, cu_seqlens_q, seqused_k, scale = \
        make_test_data(num_seqs, seq_len_q, seq_len_k, nqh, nkvh, hs, bs, device, dtype)

    for _ in range(5):
        mod.unified_attention_3d(
            q, key_cache, value_cache, block_table,
            cu_seqlens_q, seqused_k, scale, num_segments=nseg,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.unified_attention_3d(
            q, key_cache, value_cache, block_table,
            cu_seqlens_q, seqused_k, scale, num_segments=nseg,
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
                "num_seqs": shape[0], "seq_len_q": shape[1], "seq_len_k": shape[2],
                "num_query_heads": shape[3], "num_kv_heads": shape[4],
                "head_size": shape[5], "block_size": shape[6], "num_segments": shape[7],
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
