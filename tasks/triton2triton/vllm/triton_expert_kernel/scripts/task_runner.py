#!/usr/bin/env python3
"""Task runner for triton2triton/triton_expert_kernel"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_expert_kernel"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_expert_kernel.py")

TEST_SHAPES = [(32, 64, 32), (64, 128, 64), (128, 256, 128), (256, 512, 256), (512, 1024, 512)]
CORRECTNESS_CASES = [
    *[
        (f"aligned_{M}x{K}x{N}", M, K, N, "float16", "contiguous")
        for M, K, N in TEST_SHAPES
    ],
    ("sub_tile_tails", 7, 13, 11, "float16", "contiguous"),
    ("multi_tile_tails_wide", 65, 97, 129, "float16", "contiguous"),
    ("strided_bfloat16_tails", 73, 45, 19, "bfloat16", "strided"),
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


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "expert_gemm"), "Missing expert_gemm"
        assert hasattr(mod, "expert_triton_kernel"), "Missing expert_triton_kernel"
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
    for i, (case_name, M, K, N, dtype_name, layout) in enumerate(CORRECTNESS_CASES):
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, dtype_name)
            if layout == "strided":
                A_storage = torch.randn(M, 2 * K, device=device, dtype=dtype) * 0.1
                B_storage = torch.randn(2 * N, K, device=device, dtype=dtype) * 0.1
                A = A_storage[:, ::2]
                B = B_storage[::2, :].T
                assert not A.is_contiguous() and not B.is_contiguous()
            else:
                A = torch.randn(M, K, device=device, dtype=dtype) * 0.1
                B = torch.randn(K, N, device=device, dtype=dtype) * 0.1
            result = mod.expert_gemm(A, B)
            torch.cuda.synchronize()
            ref = (A.float() @ B.float()).to(dtype)
            if result.shape != ref.shape or result.dtype != ref.dtype:
                return False, (
                    f"{case_name}: expected shape/dtype {ref.shape}/{ref.dtype}, "
                    f"got {result.shape}/{result.dtype}"
                )
            if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return False, f"{case_name}: max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"{case_name}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []
    for test_idx, (M, K, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
            def _bench_fn():
                mod.expert_gemm(A, B)
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
                    "M": M,
                    "K": K,
                    "N": N
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "N": N
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
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(CORRECTNESS_CASES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
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
