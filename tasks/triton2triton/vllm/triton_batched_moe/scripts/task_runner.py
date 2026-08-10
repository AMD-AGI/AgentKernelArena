#!/usr/bin/env python3
"""Task runner for triton2triton/triton_batched_moe"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_batched_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_batched_moe.py")

# (E, max_tokens, K, N)
TEST_SHAPES = [
    (4, 16, 64, 64),
    (8, 32, 128, 128),
    (8, 64, 256, 256),
    (16, 64, 512, 512),
    (8, 128, 1024, 512),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract cases.  Keep these separate from TEST_SHAPES:
# TEST_SHAPES is the immutable performance/scoring workload.
BATCHED_MOE_CORRECTNESS_CASES = [
    {
        "name": "all_zero_token_experts_nonmultiple",
        "shape": (3, 7, 33, 35),
        "expert_num_tokens": (0, 0, 0),
        "a_last_dim_padding": 0,
        "b_last_dim_padding": 0,
    },
    {
        "name": "mixed_zero_tokens_noncompact_nonmultiple",
        "shape": (3, 17, 70, 73),
        "expert_num_tokens": (0, 7, 17),
        "a_last_dim_padding": 5,
        "b_last_dim_padding": 7,
    },
]


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


def _make_padded_random(torch, shape, last_dim_padding, *, device):
    """Return a logical tensor whose row/expert strides include optional padding."""
    storage_shape = (*shape[:-1], shape[-1] + last_dim_padding)
    storage = torch.randn(
        storage_shape,
        device=device,
        dtype=torch.float16,
    )
    storage.mul_(0.1)
    return storage[..., : shape[-1]], storage


def _check_batched_moe_case(
    torch,
    mod,
    A,
    B,
    expert_num_tokens,
    label,
    *,
    backing_storages=(),
):
    frozen_A = A.clone()
    frozen_B = B.clone()
    frozen_counts = expert_num_tokens.clone()
    frozen_backing = [
        (name, storage, storage.clone()) for name, storage in backing_storages
    ]

    result = mod.batched_moe_gemm(A, B, expert_num_tokens)
    torch.cuda.synchronize()

    if result.shape != (A.shape[0], A.shape[1], B.shape[1]):
        return f"{label}: wrong output shape {tuple(result.shape)}"
    if result.dtype != A.dtype:
        return f"{label}: wrong output dtype {result.dtype}, expected {A.dtype}"
    result_storage = result.untyped_storage().data_ptr()
    if result_storage in {
        A.untyped_storage().data_ptr(),
        B.untyped_storage().data_ptr(),
        expert_num_tokens.untyped_storage().data_ptr(),
    }:
        return f"{label}: output aliases an input"
    for name, observed, frozen in (
        ("A", A, frozen_A),
        ("B", B, frozen_B),
        ("expert_num_tokens", expert_num_tokens, frozen_counts),
    ):
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated protected input {name}"
    for name, observed, frozen in frozen_backing:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate wrote outside logical {name} view"

    ref = torch.zeros_like(result)
    for expert in range(A.shape[0]):
        num_tokens = int(expert_num_tokens[expert].item())
        if num_tokens > 0:
            ref[expert, :num_tokens] = (
                A[expert, :num_tokens].float() @ B[expert].float().T
            ).to(torch.float16)

    if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
        max_diff = (result.float() - ref.float()).abs().max().item()
        return f"{label}: max diff = {max_diff:.6f}"
    return None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "batched_moe_gemm"), "Missing batched_moe_gemm"
        assert hasattr(mod, "batched_triton_kernel"), "Missing batched_triton_kernel"
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
    for i, (E, max_tokens, K, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            expert_num_tokens = torch.randint(1, max_tokens + 1, (E,), device=device, dtype=torch.int32)
            error = _check_batched_moe_case(
                torch,
                mod,
                A,
                B,
                expert_num_tokens,
                f"Shape {i+1}",
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for case_index, case in enumerate(BATCHED_MOE_CORRECTNESS_CASES):
        try:
            torch.manual_seed(142 + case_index)
            E, max_tokens, K, N = case["shape"]
            A, A_storage = _make_padded_random(
                torch,
                (E, max_tokens, K),
                case["a_last_dim_padding"],
                device=device,
            )
            B, B_storage = _make_padded_random(
                torch,
                (E, N, K),
                case["b_last_dim_padding"],
                device=device,
            )
            expert_num_tokens = torch.tensor(
                case["expert_num_tokens"],
                device=device,
                dtype=torch.int32,
            )
            if case["a_last_dim_padding"] and A.is_contiguous():
                return False, f"{case['name']}: A padding did not produce a view"
            if case["b_last_dim_padding"] and B.is_contiguous():
                return False, f"{case['name']}: B padding did not produce a view"
            error = _check_batched_moe_case(
                torch,
                mod,
                A,
                B,
                expert_num_tokens,
                case["name"],
                backing_storages=(
                    ("A", A_storage),
                    ("B", B_storage),
                ),
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"{case['name']}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (E, max_tokens, K, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            A = torch.randn(E, max_tokens, K, device=device, dtype=torch.float16) * 0.1
            B = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            expert_num_tokens = torch.full((E,), max_tokens, device=device, dtype=torch.int32)

            def _bench_fn():
                mod.batched_moe_gemm(A, B, expert_num_tokens)
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
                    "E": E,
                    "max_tokens": max_tokens,
                    "K": K,
                    "N": N
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "E": E,
                    "max_tokens": max_tokens,
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(BATCHED_MOE_CORRECTNESS_CASES),
        }
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
