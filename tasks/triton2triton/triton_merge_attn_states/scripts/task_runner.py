#!/usr/bin/env python3
"""Task runner for triton2triton/triton_merge_attn_states"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_merge_attn_states"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_merge_attn_states.py")

# Test configurations: (num_tokens, num_heads, head_size)
TEST_SHAPES = [
    (32, 8, 64),
    (128, 16, 64),
    (256, 32, 128),
    (64, 8, 128),
    (512, 16, 64),
]
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_merge(prefix_output, prefix_lse, suffix_output, suffix_lse):
    """CPU/PyTorch reference for merge_attn_states."""
    import torch
    # prefix_lse, suffix_lse: [num_heads, num_tokens]
    # prefix_output, suffix_output: [num_tokens, num_heads, head_size]
    num_tokens = prefix_output.shape[0]
    num_heads = prefix_output.shape[1]

    # Transpose lse to [num_tokens, num_heads] for broadcasting
    p_lse = prefix_lse.T.float()  # [num_tokens, num_heads]
    s_lse = suffix_lse.T.float()  # [num_tokens, num_heads]

    max_lse = torch.maximum(p_lse, s_lse)
    p_exp = torch.exp(p_lse - max_lse)
    s_exp = torch.exp(s_lse - max_lse)
    denom = p_exp + s_exp

    p_scale = (p_exp / denom).unsqueeze(-1)  # [num_tokens, num_heads, 1]
    s_scale = (s_exp / denom).unsqueeze(-1)

    out = prefix_output.float() * p_scale + suffix_output.float() * s_scale
    return out.to(prefix_output.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "merge_attn_states"), "Missing merge_attn_states"
        assert hasattr(mod, "merge_attn_states_kernel"), "Missing merge_attn_states_kernel"
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

    for i, (num_tokens, num_heads, head_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
            suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
            prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
            suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
            output = torch.empty_like(prefix_output)

            mod.merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
            torch.cuda.synchronize()

            ref = reference_merge(prefix_output, prefix_lse, suffix_output, suffix_lse)

            if not torch.allclose(output, ref, atol=1e-2, rtol=1e-2):
                max_diff = (output - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (tokens={num_tokens}, heads={num_heads}, hd={head_size}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (tokens={num_tokens}, heads={num_heads}, hd={head_size}): "
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
    num_tokens, num_heads, head_size = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
    suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
    output = torch.empty_like(prefix_output)

    for _ in range(5):
        mod.merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
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
        num_tokens, num_heads, head_size = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {"num_tokens": num_tokens, "num_heads": num_heads, "head_size": head_size},
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
