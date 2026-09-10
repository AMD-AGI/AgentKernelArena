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

# Keep performance coverage stable; additional padding cases are correctness-only.
PERFORMANCE_SHAPES = [
    (128, 4),
    (256, 8),
    (128, 16),
    (384, 12),
    (512, 8),
]
CORRECTNESS_SHAPES = PERFORMANCE_SHAPES + [
    (129, 4),  # Row padding with packed columns.
    (128, 5),  # Column padding with aligned rows.
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

    padded = input_matrix.new_zeros((padded_rows, padded_cols))
    padded[:rows, :cols] = input_matrix

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

    for i, (rows, cols) in enumerate(CORRECTNESS_SHAPES):
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
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (rows, cols) in enumerate(PERFORMANCE_SHAPES):
        try:
            torch.manual_seed(0)
            data = torch.randint(0, 256, (rows, cols), device=device, dtype=torch.uint8)

            def _bench_fn():
                mod.triton_mx_block_rearrange(data)
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
                    "rows": rows,
                    "cols": cols
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "rows": rows,
                    "cols": cols
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
            "num_shapes": len(CORRECTNESS_SHAPES),
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
