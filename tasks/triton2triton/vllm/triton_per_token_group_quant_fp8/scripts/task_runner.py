#!/usr/bin/env python3
"""Task runner for triton2triton/triton_per_token_group_quant_fp8"""
import sys
import os
import json
import math
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_per_token_group_quant_fp8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_per_token_group_quant_fp8.py")

# Performance configs: (M, N, group_size)
TEST_SHAPES = [
    (32, 128, 128),
    (64, 256, 128),
    (128, 512, 128),
    (256, 1024, 128),
    (64, 512, 64),
]

# Targeted correctness-only cases. Keep these separate from TEST_SHAPES so
# hardening correctness coverage does not alter the performance workload.
CORRECTNESS_CASES = [
    {
        "name": "zeros",
        "shape": (1, 128),
        "group_size": 32,
        "input_kind": "zeros",
        "input_dtype": "float16",
    },
    {
        "name": "tiny_custom_eps",
        "shape": (2, 96),
        "group_size": 48,
        "input_kind": "patterned",
        "amplitudes": (5e-5, 1e-8),
        "input_dtype": "float32",
        "eps": 1e-4,
    },
    {
        "name": "saturation_boundary_explicit_dtype",
        "shape": (2, 128),
        "group_size": 64,
        "input_kind": "saturation_boundary",
        "input_dtype": "float16",
        "output_dtype": "wider_fp8",
    },
    {
        "name": "mixed_dynamic_range_3d",
        "shape": (2, 3, 192),
        "group_size": 48,
        "input_kind": "patterned",
        "amplitudes": (2**-12, 2**-4, 1.0, 64.0),
        "input_dtype": "bfloat16",
    },
    {
        "name": "ue8m0_non_power_of_two_group",
        "shape": (3, 120),
        "group_size": 40,
        "input_kind": "patterned",
        "amplitudes": (0.75, 17.0, 93.0),
        "input_dtype": "float32",
        "use_ue8m0": True,
    },
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


def reference_per_token_group_quant_fp8(
    x,
    group_size,
    fp8_dtype,
    fp8_min,
    fp8_max,
    eps=1e-10,
    use_ue8m0=False,
):
    """CPU reference for per-token-group FP8 quantization."""
    import torch

    N = x.shape[-1]
    leading_shape = x.shape[:-1]
    x_cpu = x.cpu().float().reshape(-1, N)
    M = x_cpu.shape[0]
    num_groups = N // group_size
    x_q = torch.zeros_like(x_cpu)
    x_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = x_cpu[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale_raw = absmax / fp8_max
            scale = (
                2.0 ** math.ceil(math.log2(scale_raw))
                if use_ue8m0
                else scale_raw
            )
            x_s[row, g] = scale
            x_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)

    scale_shape = leading_shape + (num_groups,)
    return x_q.reshape(x.shape).to(fp8_dtype), x_s.reshape(scale_shape)


def _get_reference_fp8_min_max(torch, fp8_dtype):
    """Derive platform clamp limits independently of the implementation."""
    if (
        hasattr(torch, "float8_e4m3fnuz")
        and fp8_dtype == torch.float8_e4m3fnuz
    ):
        return -240.0, 240.0
    finfo = torch.finfo(fp8_dtype)
    return finfo.min, finfo.max


def _get_wider_fp8_dtype(torch, default_dtype, fp8_max):
    """Return a supported explicit FP8 dtype that can represent the clamp range."""
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        if dtype == default_dtype or torch.finfo(dtype).max < fp8_max:
            continue
        try:
            torch.empty(1, device="cuda", dtype=dtype)
            return dtype
        except Exception:
            continue
    return default_dtype


def _make_patterned_input(torch, shape, group_size, amplitudes, dtype):
    """Build deterministic groups whose maxima span the requested amplitudes."""
    rows = math.prod(shape[:-1])
    num_groups = shape[-1] // group_size
    base = torch.linspace(-1.0, 1.0, group_size, dtype=torch.float32)
    x = torch.empty((rows, shape[-1]), dtype=torch.float32)

    for row in range(rows):
        for group in range(num_groups):
            amplitude = amplitudes[(row * num_groups + group) % len(amplitudes)]
            start = group * group_size
            x[row, start : start + group_size] = base * amplitude

    return x.reshape(shape).to(device="cuda", dtype=dtype)


def _make_correctness_input(torch, case, fp8_min, fp8_max):
    dtype = getattr(torch, case["input_dtype"])
    if case["input_kind"] == "zeros":
        return torch.zeros(case["shape"], device="cuda", dtype=dtype)
    if case["input_kind"] == "saturation_boundary":
        return _make_patterned_input(
            torch,
            case["shape"],
            case["group_size"],
            (fp8_max * 0.25, abs(fp8_min) * 0.25),
            dtype,
        )
    return _make_patterned_input(
        torch,
        case["shape"],
        case["group_size"],
        case["amplitudes"],
        dtype,
    )


def _check_outputs(
    torch,
    case_name,
    x_q,
    x_s,
    ref_q,
    ref_s,
    group_size,
    *,
    exact_quantized=False,
    scale_atol=1e-5,
    scale_rtol=1e-3,
):
    if not torch.allclose(x_s, ref_s, atol=scale_atol, rtol=scale_rtol):
        max_diff = (x_s - ref_s).abs().max().item()
        return f"{case_name}: scale max diff = {max_diff:.6g}"

    if exact_quantized and not torch.equal(x_q.float(), ref_q.float()):
        max_diff = (x_q.float() - ref_q.float()).abs().max().item()
        return f"{case_name}: quantized max diff = {max_diff:.6g}"

    x_dq = x_q.float() * x_s.repeat_interleave(group_size, dim=-1)
    ref_dq = ref_q.float() * ref_s.repeat_interleave(group_size, dim=-1)
    if not torch.allclose(x_dq, ref_dq, atol=1e-1, rtol=1e-1):
        max_diff = (x_dq - ref_dq).abs().max().item()
        return f"{case_name}: dequant max diff = {max_diff:.6f}"

    return None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "per_token_group_quant_fp8"), "Missing per_token_group_quant_fp8"
        assert hasattr(mod, "_per_token_group_quant_fp8"), "Missing _per_token_group_quant_fp8"
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
    fp8_dtype = mod._get_fp8_dtype()
    fp8_min, fp8_max = _get_reference_fp8_min_max(torch, fp8_dtype)

    for i, (M, N, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            x_q, x_s = mod.per_token_group_quant_fp8(x, group_size)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_per_token_group_quant_fp8(
                x, group_size, fp8_dtype, fp8_min, fp8_max
            )
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            case_name = f"Shape {i+1} (M={M}, N={N}, G={group_size})"
            error = _check_outputs(
                torch, case_name, x_q, x_s, ref_q, ref_s, group_size
            )
            if error:
                return False, error
        except Exception as e:
            return False, (
                f"Shape {i+1} (M={M}, N={N}, G={group_size}): exception: {e}"
            )

    wider_fp8_dtype = _get_wider_fp8_dtype(torch, fp8_dtype, fp8_max)
    for case in CORRECTNESS_CASES:
        case_name = case["name"]
        try:
            x = _make_correctness_input(torch, case, fp8_min, fp8_max)
            output_dtype = (
                wider_fp8_dtype
                if case.get("output_dtype") == "wider_fp8"
                else fp8_dtype
            )
            eps = case.get("eps", 1e-10)
            use_ue8m0 = case.get("use_ue8m0", False)
            group_size = case["group_size"]

            x_q, x_s = mod.per_token_group_quant_fp8(
                x,
                group_size,
                eps=eps,
                dtype=output_dtype,
                use_ue8m0=use_ue8m0,
            )
            torch.cuda.synchronize()

            if x_q.dtype != output_dtype:
                return False, (
                    f"{case_name}: output dtype {x_q.dtype} does not match "
                    f"requested dtype {output_dtype}"
                )

            ref_q, ref_s = reference_per_token_group_quant_fp8(
                x,
                group_size,
                output_dtype,
                fp8_min,
                fp8_max,
                eps=eps,
                use_ue8m0=use_ue8m0,
            )
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            error = _check_outputs(
                torch,
                case_name,
                x_q,
                x_s,
                ref_q,
                ref_s,
                group_size,
                exact_quantized=True,
                scale_atol=0.0,
                scale_rtol=1e-4,
            )
            if error:
                return False, error
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

    for test_idx, (M, N, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            def _bench_fn():
                mod.per_token_group_quant_fp8(x, group_size)
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
                    "N": N,
                    "group_size": group_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "N": N,
                    "group_size": group_size
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
