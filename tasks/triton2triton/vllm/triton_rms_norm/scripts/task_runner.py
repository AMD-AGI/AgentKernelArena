#!/usr/bin/env python3
"""Task runner for triton2triton/triton_rms_norm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_rms_norm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rms_norm.py")

# Test configurations: (rows, hidden_size)
TEST_SHAPES = [
    (32, 128),
    (64, 512),
    (128, 1024),
    (256, 2048),
    (512, 4096),
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


def reference_rms_norm(x, weight, eps=1e-6):
    """CPU/PyTorch reference for RMS norm."""
    import torch
    x_f32 = x.float()
    w_f32 = weight.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * w_f32).to(x.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "rms_norm"), "Missing rms_norm"
        assert hasattr(mod, "_rms_norm_kernel"), "Missing _rms_norm_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness(*, case_index=None):
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    dtype = torch.float16
    eps = 1e-6

    for i, (rows, hidden) in enumerate(TEST_SHAPES):
        if case_index is not None and i != case_index:
            continue
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(rows, hidden, device=device, dtype=dtype)
            weight = torch.randn(hidden, device=device, dtype=dtype)

            result = mod.rms_norm(x, weight, eps=eps)
            torch.cuda.synchronize()

            ref = reference_rms_norm(x, weight, eps)

            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i + 1} (rows={rows}, hidden={hidden}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i + 1} (rows={rows}, hidden={hidden}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    eps = 1e-6
    test_cases = []

    for test_idx, (rows, hidden) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            x = torch.randn(rows, hidden, device=device, dtype=dtype)
            weight = torch.randn(hidden, device=device, dtype=dtype)

            def _bench_fn():
                mod.rms_norm(x, weight, eps=eps)
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
                    "hidden_size": hidden,
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "rows": rows,
                    "hidden_size": hidden,
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
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
