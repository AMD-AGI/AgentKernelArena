#!/usr/bin/env python3
"""Task runner for triton2triton/triton_silu_mul_quant_fp8"""
import sys
import os
import json
import math
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_silu_mul_quant_fp8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_silu_mul_quant_fp8.py")

# Test configs: (M, N) where N must be divisible by 256 (GROUP_SIZE*2), M by 128
TEST_SHAPES = [
    (128, 256),
    (128, 512),
    (256, 512),
    (256, 1024),
    (512, 1024),
]

# Targeted correctness-only cases. Keep these separate from TEST_SHAPES so
# hardening correctness coverage does not alter the performance workload.
CORRECTNESS_CASES = [
    {
        "name": "bfloat16_ue8m0",
        "shape": (128, 256),
        "input_dtype": "bfloat16",
        "input_kind": "random",
        "inject_wide_value": True,
        "eps": 1e-6,
        "use_ue8m0": True,
    },
    {
        "name": "float32_edges_preallocated_output",
        "shape": (128, 512),
        "input_dtype": "float32",
        "input_kind": "edge_values",
        "eps": 1e-4,
        "preallocate_output": True,
        "exact_quantized": True,
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


def reference_silu_mul_quant_fp8(
    input_t,
    fp8_dtype,
    *,
    eps=1e-10,
    use_ue8m0=False,
):
    """CPU reference: silu(x[:,:N/2]) * x[:,N/2:], then quantize per group."""
    import torch
    GROUP_SIZE = 128
    M, N = input_t.shape
    N_2 = N // 2

    x = input_t.cpu().float()
    gate = x[:, :N_2]
    up = x[:, N_2:]

    # SiLU + mul
    silu_out = gate / (1.0 + torch.exp(-gate))
    y = silu_out * up

    if fp8_dtype == torch.float8_e4m3fnuz:
        fp8_min, fp8_max = -240.0, 240.0
    else:
        finfo = torch.finfo(fp8_dtype)
        fp8_min, fp8_max = finfo.min, finfo.max

    num_groups = N_2 // GROUP_SIZE
    y_q = torch.zeros_like(y)
    # Column-major scales: shape [M, num_groups]
    y_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * GROUP_SIZE
            end = start + GROUP_SIZE
            group = y[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale_raw = absmax / fp8_max
            scale = (
                2.0 ** math.ceil(math.log2(scale_raw))
                if use_ue8m0
                else scale_raw
            )
            y_s[row, g] = scale
            y_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)

    return y_q.to(fp8_dtype), y_s


def _make_correctness_input(torch, case):
    """Create deterministic inputs for the targeted correctness cases."""
    dtype = getattr(torch, case["input_dtype"])
    shape = case["shape"]
    if case["input_kind"] == "random":
        torch.manual_seed(123)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        if case.get("inject_wide_value", False):
            N_2 = shape[1] // 2
            x[:, 0] = 1e5
            x[:, N_2] = 1e-4
        return x

    # Group zero mixes tiny values with large finite products, while group one
    # remains exactly zero so the non-default epsilon determines its scale.
    x = torch.zeros(shape, device="cuda", dtype=dtype)
    N_2 = shape[1] // 2
    x[:, 0] = 1e-20
    x[:, N_2] = 1.0
    x[:, 1] = -1e-20
    x[:, N_2 + 1] = 1.0
    x[:, 2] = 1e5
    x[:, N_2 + 2] = 0.064
    x[:, 3] = 1e5
    x[:, N_2 + 3] = -0.064
    return x


def _check_outputs(
    torch,
    case_name,
    y_q,
    y_s,
    ref_q,
    ref_s,
    *,
    exact_quantized=False,
    scale_atol=1e-2,
    scale_rtol=1e-1,
):
    """Compare scales and quantized results without changing task tolerances."""
    if y_q.shape != ref_q.shape or y_s.shape != ref_s.shape:
        return (
            f"{case_name}: output shapes {(tuple(y_q.shape), tuple(y_s.shape))} "
            f"do not match expected {(tuple(ref_q.shape), tuple(ref_s.shape))}"
        )
    if y_q.dtype != ref_q.dtype or y_s.dtype != ref_s.dtype:
        return (
            f"{case_name}: output dtypes {(y_q.dtype, y_s.dtype)} do not match "
            f"expected {(ref_q.dtype, ref_s.dtype)}"
        )

    if not torch.allclose(y_s, ref_s, atol=scale_atol, rtol=scale_rtol):
        max_diff = (y_s - ref_s).abs().max().item()
        return f"{case_name}: scale max diff = {max_diff:.6g}"

    if exact_quantized and not torch.equal(y_q.float(), ref_q.float()):
        max_diff = (y_q.float() - ref_q.float()).abs().max().item()
        return f"{case_name}: quantized max diff = {max_diff:.6g}"

    GROUP_SIZE = 128
    y_dq = y_q.float() * y_s.repeat_interleave(GROUP_SIZE, dim=-1)
    ref_dq = ref_q.float() * ref_s.repeat_interleave(GROUP_SIZE, dim=-1)
    if not torch.allclose(y_dq, ref_dq, atol=5e-1, rtol=1e-1):
        max_diff = (y_dq - ref_dq).abs().max().item()
        return f"{case_name}: dequant max diff = {max_diff:.6f}"

    return None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing silu_mul_per_token_group_quant_fp8_colmajor"
        assert hasattr(mod, "_silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing _silu_mul_per_token_group_quant_fp8_colmajor"
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

    for i, (M, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            y_q, y_s = mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_silu_mul_quant_fp8(x, fp8_dtype)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            case_name = f"Shape {i+1} (M={M}, N={N})"
            error = _check_outputs(
                torch, case_name, y_q, y_s, ref_q, ref_s
            )
            if error:
                return False, error
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, N={N}): exception: {e}"

    for case in CORRECTNESS_CASES:
        case_name = case["name"]
        try:
            x = _make_correctness_input(torch, case)
            M, N = x.shape
            eps = case.get("eps", 1e-10)
            use_ue8m0 = case.get("use_ue8m0", False)

            output = None
            if case.get("preallocate_output", False):
                output = torch.full(
                    (M, N // 2),
                    240.0,
                    device=device,
                    dtype=fp8_dtype,
                )

            y_q, y_s = mod.silu_mul_per_token_group_quant_fp8_colmajor(
                x,
                output=output,
                use_ue8m0=use_ue8m0,
                eps=eps,
            )
            torch.cuda.synchronize()

            if output is not None and y_q.data_ptr() != output.data_ptr():
                return False, f"{case_name}: returned a different output buffer"

            ref_q, ref_s = reference_silu_mul_quant_fp8(
                x,
                fp8_dtype,
                eps=eps,
                use_ue8m0=use_ue8m0,
            )
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            error = _check_outputs(
                torch,
                case_name,
                y_q,
                y_s,
                ref_q,
                ref_s,
                exact_quantized=case.get("exact_quantized", False),
                scale_atol=1e-8,
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

    for test_idx, (M, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            def _bench_fn():
                mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
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
                    "N": N
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
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
