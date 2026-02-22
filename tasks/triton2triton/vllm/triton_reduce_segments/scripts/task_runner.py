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
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(num_seqs, num_query_heads, head_size, num_segments, seq_len_k,
                   device="cuda"):
    import torch
    import triton
    head_size_padded = triton.next_power_of_2(head_size)
    total_tokens = num_seqs  # 1 query token per seq (decode)

    # Simulate segment outputs with random values
    torch.manual_seed(42)
    segm_output = torch.randn(total_tokens, num_query_heads, num_segments, head_size_padded,
                              device=device, dtype=torch.float32)
    segm_max = torch.randn(total_tokens, num_query_heads, num_segments,
                           device=device, dtype=torch.float32)
    segm_expsum = torch.rand(total_tokens, num_query_heads, num_segments,
                             device=device, dtype=torch.float32) + 0.1

    output = torch.zeros(total_tokens, num_query_heads, head_size,
                         device=device, dtype=torch.float16)

    seqused_k = torch.full((num_seqs,), seq_len_k, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.arange(0, num_seqs + 1, device=device, dtype=torch.int32)

    return segm_output, segm_max, segm_expsum, output, seqused_k, cu_seqlens_q


def reference_reduce(segm_output, segm_max, segm_expsum, head_size):
    """CPU reference for logsumexp reduction."""
    import torch
    total_tokens = segm_output.shape[0]
    num_heads = segm_output.shape[1]
    output = torch.zeros(total_tokens, num_heads, head_size,
                         device=segm_output.device, dtype=torch.float32)

    overall_max = segm_max.max(dim=-1).values  # [tokens, heads]
    rescaled_expsum = segm_expsum * torch.exp(segm_max - overall_max.unsqueeze(-1))
    overall_expsum = rescaled_expsum.sum(dim=-1)  # [tokens, heads]

    rescaled_output = segm_output * torch.exp(segm_max - overall_max.unsqueeze(-1)).unsqueeze(-1)
    summed = rescaled_output.sum(dim=2)  # [tokens, heads, head_size_padded]

    safe_denom = overall_expsum.unsqueeze(-1)
    safe_denom = torch.where(safe_denom == 0, torch.ones_like(safe_denom), safe_denom)
    output = summed[:, :, :head_size] / safe_denom

    return output.half()


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

    for i, (num_seqs, nqh, hs, nseg, slk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q = \
                make_test_data(num_seqs, nqh, hs, nseg, slk, device)

            mod.reduce_attention_segments(
                segm_output, segm_max_t, segm_expsum, output,
                seqused_k, cu_seqlens_q,
            )
            torch.cuda.synchronize()

            ref = reference_reduce(segm_output, segm_max_t, segm_expsum, hs)

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
    num_seqs, nqh, hs, nseg, slk = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    segm_output, segm_max_t, segm_expsum, output, seqused_k, cu_seqlens_q = \
        make_test_data(num_seqs, nqh, hs, nseg, slk, device)

    for _ in range(5):
        mod.reduce_attention_segments(
            segm_output, segm_max_t, segm_expsum, output,
            seqused_k, cu_seqlens_q,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.reduce_attention_segments(
            segm_output, segm_max_t, segm_expsum, output,
            seqused_k, cu_seqlens_q,
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
                "num_seqs": shape[0], "num_query_heads": shape[1],
                "head_size": shape[2], "num_segments": shape[3],
                "seq_len_k": shape[4],
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
