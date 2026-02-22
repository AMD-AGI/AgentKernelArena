#!/usr/bin/env python3
"""Task runner for triton2triton/triton_scale_swizzle"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_scale_swizzle"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_scale_swizzle.py")

# Test configs: (rows, cols) - must be multiples of (128, 4) for the kernel
TEST_SHAPES = [
    (128, 4),
    (256, 8),
    (128, 16),
    (384, 12),
    (512, 8),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cdiv(a, b):
    return (a + b - 1) // b


def reference_scale_swizzle(input_matrix):
    """CPU reference: PyTorch-based block rearrangement."""
    import torch
    rows, cols = input_matrix.shape
    n_row_blocks = cdiv(rows, 128)
    n_col_blocks = cdiv(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    assert (rows, cols) == (padded_rows, padded_cols), (
        f"Input must be padded to multiples of (128, 4), got ({rows}, {cols})"
    )

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten().view(padded_rows, padded_cols)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_mx_block_rearrange"), "Missing triton_mx_block_rearrange"
        assert hasattr(mod, "triton_scale_swizzle"), "Missing triton_scale_swizzle"
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

    for i, (rows, cols) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            # Create uint8 tensor (1-byte elements as required)
            data = torch.randint(0, 256, (rows, cols), device=device, dtype=torch.uint8)

            result = mod.triton_mx_block_rearrange(data)
            torch.cuda.synchronize()

            ref = reference_scale_swizzle(data.cpu())
            ref = ref.to(device)

            # Bit-exact comparison for uint8
            if not torch.equal(result, ref):
                diff_count = (result != ref).sum().item()
                return False, (
                    f"Shape {i+1} ({rows}, {cols}): {diff_count} mismatched elements"
                )
        except Exception as e:
            return False, f"Shape {i+1} ({rows}, {cols}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    rows, cols = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    data = torch.randint(0, 256, (rows, cols), device=device, dtype=torch.uint8)

    for _ in range(5):
        mod.triton_mx_block_rearrange(data)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.triton_mx_block_rearrange(data)
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
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
