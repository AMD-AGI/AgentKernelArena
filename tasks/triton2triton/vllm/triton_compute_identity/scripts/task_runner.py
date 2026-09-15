#!/usr/bin/env python3
"""Task runner for triton2triton/triton_compute_identity"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_compute_identity"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_compute_identity.py")

# (num_tokens, hidden_dim, top_k)
TEST_SHAPES = [
    (32, 256, 2),
    (64, 512, 2),
    (128, 1024, 2),
    (256, 2048, 4),
    (512, 4096, 2),
]
CORRECTNESS_CASES = [
    (*shape, "positive") for shape in TEST_SHAPES
] + [
    # Keep hidden_dim divisible by the original kernel's 256-element block while
    # exercising both masked dimensions in candidates that tile multiple tokens.
    (33, 768, 2, "positive"),
    # Exercise a second hidden-dimension tail and signed, near-cancelling scales.
    (31, 1280, 4, "cancellation"),
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


def reference_compute_identity(hidden_states, expert_scales, top_k):
    """CPU reference: hidden_states * sum(expert_scales, dim=-1, keepdim=True)."""
    import torch
    scale_sum = expert_scales.sum(dim=-1, keepdim=True)
    return (hidden_states.float() * scale_sum.float()).to(hidden_states.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "compute_identity"), "Missing compute_identity"
        assert hasattr(mod, "compute_identity_kernel"), "Missing compute_identity_kernel"
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
    for i, (num_tokens, hidden_dim, top_k, scale_pattern) in enumerate(
        CORRECTNESS_CASES
    ):
        try:
            torch.manual_seed(42 + i)
            hidden_states = torch.randn(
                num_tokens, hidden_dim, device=device, dtype=torch.float16
            )
            if scale_pattern == "cancellation":
                magnitudes = torch.rand(num_tokens, 2, device=device, dtype=torch.float32) + 0.5
                residual = torch.linspace(-1e-3, 1e-3, num_tokens, device=device)
                expert_scales = torch.stack(
                    (
                        magnitudes[:, 0],
                        -magnitudes[:, 0],
                        magnitudes[:, 1],
                        -magnitudes[:, 1] + residual,
                    ),
                    dim=1,
                )
            else:
                expert_scales = torch.randn(
                    num_tokens, top_k, device=device, dtype=torch.float32
                ).abs() * 0.5

            result = mod.compute_identity(hidden_states, expert_scales, top_k)
            torch.cuda.synchronize()

            ref = reference_compute_identity(hidden_states, expert_scales, top_k).to(device)
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
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_tokens, hidden_dim, top_k) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
            expert_scales = torch.randn(num_tokens, top_k, device=device, dtype=torch.float32).abs() * 0.5

            def _bench_fn():
                mod.compute_identity(hidden_states, expert_scales, top_k)
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
                    "num_tokens": num_tokens,
                    "hidden_dim": hidden_dim,
                    "top_k": top_k
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_tokens": num_tokens,
                    "hidden_dim": hidden_dim,
                    "top_k": top_k
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
