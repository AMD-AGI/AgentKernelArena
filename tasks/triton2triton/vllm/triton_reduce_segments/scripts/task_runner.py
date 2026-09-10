#!/usr/bin/env python3
"""Task runner for triton2triton/triton_reduce_segments"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_reduce_segments"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_reduce_segments.py")

# Test configs: (num_seqs, num_query_heads, head_size, num_segments, seq_len_k)
TEST_SHAPES = [
    (4, 8, 64, 2, 128),
    (8, 16, 64, 4, 256),
    (16, 32, 128, 4, 512),
    (4, 8, 128, 2, 64),
    (32, 16, 64, 8, 1024),
]

# Additional cases are correctness-only so benchmark coverage and methodology
# remain unchanged. A query-length entry of zero represents a sequence with no
# query tokens in the packed batch.
CORRECTNESS_CASES = [
    {
        "name": f"decode_{i + 1}",
        "shape": shape,
    }
    for i, shape in enumerate(TEST_SHAPES)
] + [
    {
        "name": "packed_variable_partial_fp16",
        "shape": (4, 3, 80, 4, None),
        "query_lens": (2, 0, 3, 1),
        "seq_lens": (1, 127, 33, 65),
        "segment_dtype": "float16",
    },
    {
        "name": "zero_denom_extreme_max_float32_output",
        "shape": (2, 2, 48, 4, None),
        "query_lens": (1, 2),
        "seq_lens": (64, 128),
        "output_dtype": "float32",
        "special_values": "zero_denom_extreme_max",
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


def make_test_data(num_seqs, num_query_heads, head_size, num_segments, seq_len_k,
                   device="cuda", query_lens=None, seq_lens=None,
                   segment_dtype="float32", output_dtype="float16"):
    import torch
    import triton
    head_size_padded = triton.next_power_of_2(head_size)
    if query_lens is None:
        query_lens = (1,) * num_seqs
    if len(query_lens) != num_seqs or any(length < 0 for length in query_lens):
        raise ValueError("query_lens must contain one nonnegative length per sequence")
    total_tokens = sum(query_lens)

    segment_torch_dtype = getattr(torch, segment_dtype)
    output_torch_dtype = getattr(torch, output_dtype)

    # Simulate segment outputs with random values
    torch.manual_seed(42)
    segm_output = torch.randn(total_tokens, num_query_heads, num_segments, head_size_padded,
                              device=device, dtype=segment_torch_dtype)
    segm_max = torch.randn(total_tokens, num_query_heads, num_segments,
                           device=device, dtype=segment_torch_dtype)
    segm_expsum = torch.rand(total_tokens, num_query_heads, num_segments,
                             device=device, dtype=segment_torch_dtype) + 0.1

    output = torch.zeros(total_tokens, num_query_heads, head_size,
                         device=device, dtype=output_torch_dtype)

    if seq_lens is None:
        seq_lens = (seq_len_k,) * num_seqs
    if len(seq_lens) != num_seqs or any(length <= 0 for length in seq_lens):
        raise ValueError("seq_lens must contain one positive length per sequence")
    seqused_k = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(query_lens, device=device, dtype=torch.int32).cumsum(0)

    return segm_output, segm_max, segm_expsum, output, seqused_k, cu_seqlens_q


def reference_reduce(segm_output, segm_max, segm_expsum, head_size,
                     seqused_k, cu_seqlens_q, tile_size=16):
    """PyTorch reference for logsumexp reduction."""
    import torch
    total_tokens = segm_output.shape[0]
    num_segments = segm_output.shape[2]

    token_indices = torch.arange(total_tokens, device=segm_output.device, dtype=torch.int32)
    seq_indices = torch.searchsorted(cu_seqlens_q, token_indices, right=True) - 1
    token_seq_lens = seqused_k[seq_indices]
    tiles_per_segment = torch.div(
        token_seq_lens + num_segments * tile_size - 1,
        num_segments * tile_size,
        rounding_mode="floor",
    )
    active_segments = torch.div(
        token_seq_lens + tiles_per_segment * tile_size - 1,
        tiles_per_segment * tile_size,
        rounding_mode="floor",
    )
    segment_mask = (
        torch.arange(num_segments, device=segm_output.device)[None, :]
        < active_segments[:, None]
    )[:, None, :]

    segm_max_f32 = segm_max.float()
    segm_expsum_f32 = segm_expsum.float()
    segm_output_f32 = segm_output.float()
    masked_max = torch.where(segment_mask, segm_max_f32, -torch.inf)
    overall_max = masked_max.max(dim=-1).values  # [tokens, heads]
    max_scale = torch.exp(masked_max - overall_max.unsqueeze(-1))
    rescaled_expsum = torch.where(
        segment_mask,
        segm_expsum_f32 * max_scale,
        0.0,
    )
    overall_expsum = rescaled_expsum.sum(dim=-1)  # [tokens, heads]

    rescaled_output = torch.where(
        segment_mask.unsqueeze(-1),
        segm_output_f32 * max_scale.unsqueeze(-1),
        0.0,
    )
    summed = rescaled_output.sum(dim=2)  # [tokens, heads, head_size_padded]

    denominator = overall_expsum.unsqueeze(-1)
    safe_denom = torch.where(denominator == 0, 1.0, denominator)
    output = torch.where(
        denominator == 0,
        0.0,
        summed[:, :, :head_size] / safe_denom,
    )

    return output


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "reduce_attention_segments"), "Missing reduce_attention_segments"
        assert hasattr(mod, "reduce_segments"), "Missing reduce_segments"
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

    for i, case in enumerate(CORRECTNESS_CASES):
        name = case["name"]
        num_seqs, nqh, hs, nseg, slk = case["shape"]
        try:
            torch.manual_seed(42 + i)
            segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q = \
                make_test_data(
                    num_seqs, nqh, hs, nseg, slk, device,
                    query_lens=case.get("query_lens"),
                    seq_lens=case.get("seq_lens"),
                    segment_dtype=case.get("segment_dtype", "float32"),
                    output_dtype=case.get("output_dtype", "float16"),
                )

            if case.get("special_values") == "zero_denom_extreme_max":
                segm_expsum[0, 0, :] = 0
                segm_max_t[1, 0, :] = torch.tensor(
                    (-1.0e4, -1.0e3, 0.0, 1.0e4),
                    device=device,
                    dtype=segm_max_t.dtype,
                )

            mod.reduce_attention_segments(
                segm_output, segm_max_t, segm_expsum, output,
                seqused_k, cu_seqlens_q,
            )
            torch.cuda.synchronize()

            ref = reference_reduce(
                segm_output, segm_max_t, segm_expsum, hs,
                seqused_k, cu_seqlens_q,
            ).to(output.dtype)

            if not torch.allclose(output.float(), ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (output.float() - ref.float()).abs().max().item()
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
    test_cases = []

    for test_idx, (num_seqs, nqh, hs, nseg, slk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q = \
                make_test_data(num_seqs, nqh, hs, nseg, slk, device)

            def _bench_fn():
                mod.reduce_attention_segments(
                    segm_output, segm_max_t, segm_expsum, output,
                    seqused_k, cu_seqlens_q,
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
                    "num_query_heads": nqh,
                    "head_size": hs,
                    "num_segments": nseg,
                    "seq_len_k": slk
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_seqs": num_seqs,
                    "num_query_heads": nqh,
                    "head_size": hs,
                    "num_segments": nseg,
                    "seq_len_k": slk
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
            "num_shapes": len(CORRECTNESS_CASES),
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
