#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fused_moe"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe.py")

# (M, K, E, N, topk)
TEST_SHAPES = [
    (16, 64, 4, 64, 2),
    (32, 128, 8, 128, 2),
    (64, 256, 8, 256, 2),
    (128, 512, 16, 512, 2),
    (256, 1024, 8, 1024, 2),
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


def reference_fused_moe(input, expert_weights, topk_ids, topk_weights, mul_routed_weight):
    """CPU reference: per-token expert GEMM with optional weight scaling."""
    import torch
    M, K = input.shape
    E, N, _ = expert_weights.shape
    topk = topk_ids.shape[1]
    num_valid = M * topk
    output = torch.zeros(num_valid, N, device=input.device, dtype=torch.float32)

    for token_idx in range(M):
        for k in range(topk):
            flat_idx = token_idx * topk + k
            expert_id = topk_ids[token_idx, k].item()
            if expert_id < 0 or expert_id >= E:
                continue
            # C = A @ B^T where B is [N, K]
            row = input[token_idx].float() @ expert_weights[expert_id].float().T
            if mul_routed_weight:
                row *= topk_weights[flat_idx].item()
            output[flat_idx] = row
    return output.to(input.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_moe"), "Missing fused_moe"
        assert hasattr(mod, "fused_moe_kernel"), "Missing fused_moe_kernel"
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
    for i, (M, K, E, N, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            expert_weights = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

            result = mod.fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, mul_routed_weight=True)
            torch.cuda.synchronize()

            ref = reference_fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, True).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1} (M={M},K={K},E={E},N={N}): max diff = {max_diff:.6f}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (M, K, E, N, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            expert_weights = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

            def _bench_fn():
                mod.fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, True)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                use_cuda_graph=False,
                fallback_reason="fused_moe_host_routing_and_dynamic_allocations",
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "M": M,
                    "K": K,
                    "E": E,
                    "N": N,
                    "topk": topk
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "E": E,
                    "N": N,
                    "topk": topk
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
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
