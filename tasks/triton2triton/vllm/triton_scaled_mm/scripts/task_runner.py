#!/usr/bin/env python3
"""Task runner for triton2triton/triton_scaled_mm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_scaled_mm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_scaled_mm.py")

# Test configs: (M, K, N, per_token_scale_a, per_channel_scale_b, has_bias)
TEST_SHAPES = [
    (32, 64, 64, True, True, False),
    (64, 128, 128, True, True, True),
    (128, 256, 256, False, False, False),
    (256, 512, 512, True, True, True),
    (64, 256, 128, True, False, False),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract coverage. Keep TEST_SHAPES reserved for performance.
CORRECTNESS_CASES = [
    {
        "name": "bias_after_output_cast",
        "shape": (5, 16, 7),
        "input_dtype": "float16",
        "scale_dtype": "float32",
        "out_dtype": "float16",
        "layout": "contiguous",
        "data": "bias_rounding",
        "per_token_scale_a": False,
        "per_channel_scale_b": False,
        "has_bias": True,
        "tiles": (16, 16, 16),
        "atol": 5e-5,
        "rtol": 0.0,
    },
    {
        "name": "sequential_low_precision_scales",
        "shape": (5, 64, 7),
        "input_dtype": "float16",
        "scale_dtype": "float16",
        "out_dtype": "float32",
        "layout": "contiguous",
        "data": "scale_product_underflow",
        "per_token_scale_a": False,
        "per_channel_scale_b": False,
        "has_bias": False,
        "tiles": (16, 16, 32),
        "atol": 1e-8,
        "rtol": 1e-4,
    },
    {
        "name": "weak_contiguous_custom_tiles",
        "shape": (37, 96, 45),
        "input_dtype": "float16",
        "scale_dtype": "float32",
        "out_dtype": "float16",
        "layout": "column_input_row_weight",
        "data": "random",
        "per_token_scale_a": True,
        "per_channel_scale_b": True,
        "has_bias": True,
        "tiles": (16, 32, 32),
        "atol": 1e-2,
        "rtol": 1e-2,
    },
    {
        "name": "weak_contiguous_column_weight",
        "shape": (33, 96, 47),
        "input_dtype": "float16",
        "scale_dtype": "float32",
        "out_dtype": "float16",
        "layout": "row_input_column_weight",
        "data": "random",
        "per_token_scale_a": True,
        "per_channel_scale_b": True,
        "has_bias": False,
        "tiles": (32, 16, 32),
        "atol": 1e-2,
        "rtol": 1e-2,
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


class _KernelLaunchRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.tiles = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record_and_launch(*args, **kwargs):
            self.tiles.append(
                (
                    kwargs.get("BLOCK_SIZE_M"),
                    kwargs.get("BLOCK_SIZE_N"),
                    kwargs.get("BLOCK_SIZE_K"),
                )
            )
            return launch(*args, **kwargs)

        return record_and_launch


def reference_scaled_mm(input_t, weight, scale_a, scale_b, out_dtype, bias=None):
    """Mirror vLLM's dot, sequential scales, output cast, then bias order."""
    a = input_t.float()
    b = weight.float()
    sa = scale_a.float()
    sb = scale_b.float()

    result = a @ b
    result = sa * result
    result = result * sb.reshape(1, -1)
    result = result.to(out_dtype)

    if bias is not None:
        result = result + bias

    return result


def _make_scaled_correctness_inputs(torch, case, device):
    M, K, N = case["shape"]
    input_dtype = getattr(torch, case["input_dtype"])
    scale_dtype = getattr(torch, case["scale_dtype"])
    out_dtype = getattr(torch, case["out_dtype"])

    if case["data"] == "bias_rounding":
        input_t = torch.full(
            (M, K), 1.0 / 3.0, device=device, dtype=input_dtype
        )
        weight = torch.full((K, N), 0.1875, device=device, dtype=input_dtype)
        scale_a = torch.ones((1, 1), device=device, dtype=scale_dtype)
        scale_b = torch.ones((1, 1), device=device, dtype=scale_dtype)
        bias = -torch.ones(N, device=device, dtype=out_dtype)
        backing_storages = ()
    elif case["data"] == "scale_product_underflow":
        input_t = torch.ones((M, K), device=device, dtype=input_dtype)
        weight = torch.ones((K, N), device=device, dtype=input_dtype)
        scale_a = torch.full(
            (1, 1), 2.0**-13, device=device, dtype=scale_dtype
        )
        scale_b = torch.full(
            (1, 1), 2.0**-13, device=device, dtype=scale_dtype
        )
        bias = None
        backing_storages = ()
    elif case["layout"] == "column_input_row_weight":
        input_storage = torch.randn(
            K, M + 3, device=device, dtype=input_dtype
        )
        input_storage.mul_(0.1)
        input_t = input_storage[:, :M].T
        weight_storage = torch.randn(
            K, N + 5, device=device, dtype=input_dtype
        )
        weight_storage.mul_(0.1)
        weight = weight_storage[:, :N]
        assert not input_t.is_contiguous() and input_t.stride(0) == 1
        assert not weight.is_contiguous() and weight.stride(1) == 1
        backing_storages = (
            ("input", input_storage),
            ("weight", weight_storage),
        )
    else:
        input_storage = torch.randn(
            M, K + 3, device=device, dtype=input_dtype
        )
        input_storage.mul_(0.1)
        input_t = input_storage[:, :K]
        weight_storage = torch.randn(
            N, K + 5, device=device, dtype=input_dtype
        )
        weight_storage.mul_(0.1)
        weight = weight_storage[:, :K].T
        assert not input_t.is_contiguous() and input_t.stride(1) == 1
        assert not weight.is_contiguous() and weight.stride(0) == 1
        backing_storages = (
            ("input", input_storage),
            ("weight", weight_storage),
        )

    if case["data"] == "random":
        scale_a_rows = M if case["per_token_scale_a"] else 1
        scale_b_rows = N if case["per_channel_scale_b"] else 1
        scale_a = torch.rand(
            scale_a_rows, 1, device=device, dtype=scale_dtype
        ) + 0.5
        scale_b = torch.rand(
            scale_b_rows, 1, device=device, dtype=scale_dtype
        ) + 0.5
        bias = (
            torch.randn(N, device=device, dtype=out_dtype) * 0.1
            if case["has_bias"]
            else None
        )

    return input_t, weight, scale_a, scale_b, out_dtype, bias, backing_storages


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_scaled_mm"), "Missing triton_scaled_mm"
        assert hasattr(mod, "scaled_mm_kernel"), "Missing scaled_mm_kernel"
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

    for i, (M, K, N, per_tok_a, per_ch_b, has_bias) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            input_t = torch.randn(M, K, device=device, dtype=dtype) * 0.1
            weight = torch.randn(K, N, device=device, dtype=dtype) * 0.1

            if per_tok_a:
                scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_a = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5

            if per_ch_b:
                scale_b = torch.rand(N, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_b = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5

            bias = torch.randn(N, device=device, dtype=dtype) * 0.1 if has_bias else None

            result = mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)
            torch.cuda.synchronize()

            ref = reference_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)

            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, K={K}, N={N}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, K={K}, N={N}): exception: {e}"

    for i, case in enumerate(CORRECTNESS_CASES):
        name = case["name"]
        try:
            torch.manual_seed(142 + i)
            inputs = _make_scaled_correctness_inputs(torch, case, device)
            (
                input_t,
                weight,
                scale_a,
                scale_b,
                out_dtype,
                bias,
                backing_storages,
            ) = inputs
            block_m, block_n, block_k = case["tiles"]
            protected_inputs = [
                ("input", input_t, input_t.clone()),
                ("weight", weight, weight.clone()),
                ("scale_a", scale_a, scale_a.clone()),
                ("scale_b", scale_b, scale_b.clone()),
            ]
            if bias is not None:
                protected_inputs.append(("bias", bias, bias.clone()))
            protected_backing = [
                (storage_name, storage, storage.clone())
                for storage_name, storage in backing_storages
            ]
            ref = reference_scaled_mm(
                protected_inputs[0][2],
                protected_inputs[1][2],
                protected_inputs[2][2],
                protected_inputs[3][2],
                out_dtype,
                bias=protected_inputs[4][2] if bias is not None else None,
            )

            original_kernel = mod.scaled_mm_kernel
            recorder = _KernelLaunchRecorder(original_kernel)
            mod.scaled_mm_kernel = recorder
            try:
                result = mod.triton_scaled_mm(
                    input_t,
                    weight,
                    scale_a,
                    scale_b,
                    out_dtype,
                    bias=bias,
                    block_size_m=block_m,
                    block_size_n=block_n,
                    block_size_k=block_k,
                    use_heuristic=False,
                )
            finally:
                mod.scaled_mm_kernel = original_kernel
            torch.cuda.synchronize()

            for input_name, observed, frozen in protected_inputs:
                if not torch.equal(observed, frozen):
                    return False, (
                        f"Contract case {name}: candidate mutated protected "
                        f"{input_name}"
                    )
            for storage_name, observed, frozen in protected_backing:
                if not torch.equal(observed, frozen):
                    return False, (
                        f"Contract case {name}: write outside logical "
                        f"{storage_name} view"
                    )

            if (block_m, block_n, block_k) not in recorder.tiles:
                return False, (
                    f"Contract case {name}: custom tile request was not honored; "
                    f"launches={recorder.tiles}"
                )
            if result.shape != ref.shape or result.dtype != out_dtype:
                return False, (
                    f"Contract case {name}: expected shape/dtype {ref.shape}/{out_dtype}, "
                    f"got {result.shape}/{result.dtype}"
                )
            result_storage = result.untyped_storage().data_ptr()
            if any(
                result_storage == observed.untyped_storage().data_ptr()
                for _input_name, observed, _frozen in protected_inputs
            ):
                return False, f"Contract case {name}: output aliases an input"
            if not torch.allclose(
                result, ref, atol=case["atol"], rtol=case["rtol"]
            ):
                max_diff = (result - ref).abs().max().item()
                return False, f"Contract case {name}: max diff = {max_diff:.8f}"
        except Exception as e:
            return False, f"Contract case {name}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (M, K, N, per_tok_a, per_ch_b, has_bias) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            input_t = torch.randn(M, K, device=device, dtype=dtype)
            weight = torch.randn(K, N, device=device, dtype=dtype)
            if per_tok_a:
                scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) + 0.5
            else:
                scale_a = torch.rand(1, 1, device=device, dtype=torch.float32) + 0.5
            if per_ch_b:
                scale_b = torch.rand(N, 1, device=device, dtype=torch.float32) + 0.5
            else:
                scale_b = torch.rand(1, 1, device=device, dtype=torch.float32) + 0.5
            bias = torch.randn(N, device=device, dtype=dtype) if has_bias else None

            def _bench_fn():
                mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)
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
                    "N": N,
                    "per_token_scale_a": per_tok_a,
                    "per_channel_scale_b": per_ch_b,
                    "has_bias": has_bias
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "N": N,
                    "per_token_scale_a": per_tok_a,
                    "per_channel_scale_b": per_ch_b,
                    "has_bias": has_bias
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
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_CASES),
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
