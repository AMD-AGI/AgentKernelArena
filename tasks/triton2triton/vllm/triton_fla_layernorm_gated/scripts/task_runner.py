#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fla_layernorm_gated"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_fla_layernorm_gated"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fla_layernorm_gated.py")

TEST_CASES = [
    # (T, D, activation, is_rms_norm, has_weight, has_bias)
    (256, 128, "swish", True, True, False),
    (192, 96, "sigmoid", True, True, True),
    (128, 64, "swish", False, True, True),
    (160, 80, "swish", True, False, False),
    (64, 48, "sigmoid", False, False, True),
]

# Keep the scored performance workload above unchanged.  Correctness also covers
# tail tiles, a wider range of feature sizes, supported low-precision dtypes,
# non-default eps values, and the SiLU spelling accepted by the public wrapper.
# (T, D, activation, is_rms_norm, has_weight, has_bias, dtype, eps)
CORRECTNESS_TEST_CASES = [
    (*case, "float32", 1e-5) for case in TEST_CASES
] + [
    (1, 1, "silu", False, True, True, "float16", 1e-6),
    (7, 31, "sigmoid", True, False, False, "bfloat16", 1e-4),
    (9, 129, "swish", False, False, True, "float16", 1e-3),
    (33, 1025, "sigmoid", True, True, False, "bfloat16", 1e-2),
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


def reference(x, g, weight=None, bias=None, activation='swish', eps=1e-5, is_rms_norm=True):
    import torch
    x_f = x.float().cpu()
    g_f = g.float().cpu()
    mean = None
    if is_rms_norm:
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        x_hat = x_f * rstd
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        x_hat = (x_f - mean) * rstd
    if weight is not None:
        x_hat = x_hat * weight.float().cpu()
    if bias is not None:
        x_hat = x_hat + bias.float().cpu()
    if activation in ('swish', 'silu'):
        y = x_hat * g_f * torch.sigmoid(g_f)
    elif activation == 'sigmoid':
        y = x_hat * torch.sigmoid(g_f)
    else:
        y = x_hat
    mean = mean.squeeze(-1) if mean is not None else None
    return y.to(x.dtype), mean, rstd.squeeze(-1)


def gen_inputs(seed, test_case, device):
    import torch
    torch.manual_seed(seed)
    T, D, activation, is_rms_norm, has_weight, has_bias, dtype_name, eps = test_case
    dtype = getattr(torch, dtype_name)
    x = torch.randn(T, D, device=device, dtype=dtype)
    g = torch.randn(T, D, device=device, dtype=dtype)
    w = torch.randn(D, device=device, dtype=dtype) if has_weight else None
    b = torch.randn(D, device=device, dtype=dtype) if has_bias else None
    kwargs = {
        "weight": w,
        "bias": b,
        "activation": activation,
        "eps": eps,
        "is_rms_norm": is_rms_norm,
    }
    return (x, g), kwargs


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "layer_norm_gated_fwd"), "Missing layer_norm_gated_fwd"
        assert hasattr(mod, "layer_norm_gated_fwd_kernel"), "Missing layer_norm_gated_fwd_kernel"
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
    for i, test_case in enumerate(CORRECTNESS_TEST_CASES):
        try:
            args, kwargs = gen_inputs(42 + i, test_case, device)
            args_cpu = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result_tuple = mod.layer_norm_gated_fwd(*args, **kwargs)
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 3:
                return False, f"Case {i+1} {test_case}: expected (y, mean, rstd) tuple"
            result, mean, rstd = result_tuple
            ref_result, ref_mean, ref_rstd = reference(
                args_cpu[0], args_cpu[1],
                weight=kwargs["weight"], bias=kwargs["bias"],
                activation=kwargs["activation"], eps=kwargs["eps"],
                is_rms_norm=kwargs["is_rms_norm"]
            )
            for name, actual, expected, expected_dtype in (
                ("y", result, ref_result, args[0].dtype),
                ("mean", mean, ref_mean, torch.float32),
                ("rstd", rstd, ref_rstd, torch.float32),
            ):
                if expected is None:
                    if actual is not None:
                        return False, f"Case {i+1} {test_case}: expected {name}=None"
                    continue
                if actual is None:
                    return False, f"Case {i+1} {test_case}: {name} is None"
                if actual.dtype != expected_dtype:
                    return False, (
                        f"Case {i+1} {test_case}: {name} dtype "
                        f"{actual.dtype} != {expected_dtype}"
                    )
                actual_f = actual.float().cpu()
                expected_f = expected.float()
                if actual_f.shape != expected_f.shape:
                    return False, (
                        f"Case {i+1} {test_case}: {name} shape "
                        f"{tuple(actual_f.shape)} != {tuple(expected_f.shape)}"
                    )
                if not torch.allclose(actual_f, expected_f, atol=1e-3, rtol=1e-3):
                    max_diff = (actual_f - expected_f).abs().max().item()
                    return False, (
                        f"Case {i+1} {test_case}: {name} max diff = {max_diff:.6f}"
                    )

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Case {i+1} {test_case}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, test_case in enumerate(TEST_CASES):
        try:
            T, D, activation, is_rms_norm, has_weight, has_bias = test_case
            args, kwargs = gen_inputs(
                42 + test_idx, (*test_case, "float32", 1e-5), device
            )
            kwargs.pop("eps")  # Preserve the original default-eps benchmark call.

            def _bench_fn():
                mod.layer_norm_gated_fwd(*args, **kwargs)
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
                    "T": T,
                    "D": D,
                    "activation": activation,
                    "is_rms_norm": is_rms_norm,
                    "has_weight": has_weight,
                    "has_bias": has_bias
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "T": T,
                    "D": D,
                    "activation": activation,
                    "is_rms_norm": is_rms_norm,
                    "has_weight": has_weight,
                    "has_bias": has_bias
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(CORRECTNESS_TEST_CASES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
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
