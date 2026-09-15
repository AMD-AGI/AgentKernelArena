#!/usr/bin/env python3
"""Task runner for triton2triton/triton_per_token_group_quant_int8"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_per_token_group_quant_int8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_per_token_group_quant_int8.py")

# Performance configs and seeded-Gaussian correctness cases: (M, N, group_size).
# Keep boundary-only correctness coverage separate so benchmark methodology and
# reported performance cases remain unchanged.
TEST_SHAPES = [
    (32, 128, 128),
    (64, 256, 128),
    (128, 512, 128),
    (256, 1024, 128),
    (64, 512, 64),
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


def reference_per_token_group_quant_int8(x, group_size, eps=1e-10):
    """CPU reference for per-token-group INT8 quantization."""
    import torch
    M, N = x.shape
    x_cpu = x.cpu().float()

    int8_max = 127
    int8_min = -128

    num_groups = N // group_size
    x_q = torch.zeros((M, N), dtype=torch.int8)
    x_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = x_cpu[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale = absmax / int8_max
            x_s[row, g] = scale
            x_q[row, start:end] = (group / scale).clamp(int8_min, int8_max).to(torch.int8)

    return x_q, x_s


def boundary_correctness_cases(torch, device):
    """Small deterministic cases for values Gaussian inputs do not cover."""
    group_size = 96

    all_zero = torch.zeros(
        (1, group_size), device=device, dtype=torch.float16
    )

    # Every magnitude is strictly below eps, including the largest FP16 value
    # below 1.0, so eps rather than the observed absmax determines the scale.
    eps_pattern = torch.tensor(
        [
            -0.99951171875,
            -0.5,
            -0.25,
            -0.003937007874015748,
            -0.0,
            0.0,
            0.003937007874015748,
            0.25,
            0.5,
            0.99951171875,
            -0.125,
            0.125,
        ],
        device=device,
        dtype=torch.float16,
    )
    eps_dominated = eps_pattern.repeat(8).reshape(1, group_size)

    # The first group includes both finite FP16 extrema and its smallest normal
    # and subnormal magnitudes. The second anchors absmax at 1.0 and samples
    # integer and half-step INT8 quantization boundaries on both signs.
    extrema_pattern = torch.tensor(
        [
            -65504.0,
            65504.0,
            -32752.0,
            32752.0,
            -1024.0,
            1024.0,
            -6.103515625e-05,
            6.103515625e-05,
            -5.960464477539063e-08,
            5.960464477539063e-08,
            -0.0,
            0.0,
        ],
        device=device,
        dtype=torch.float16,
    )
    boundary_pattern = torch.tensor(
        [
            -1.0,
            -126.5 / 127.0,
            -126.0 / 127.0,
            -64.5 / 127.0,
            -64.0 / 127.0,
            -1.5 / 127.0,
            -1.0 / 127.0,
            -0.5 / 127.0,
            0.5 / 127.0,
            1.0 / 127.0,
            1.5 / 127.0,
            64.0 / 127.0,
            64.5 / 127.0,
            126.0 / 127.0,
            126.5 / 127.0,
            1.0,
        ],
        device=device,
        dtype=torch.float16,
    )
    extrema_and_boundaries = torch.cat(
        (extrema_pattern.repeat(8), boundary_pattern.repeat(6))
    ).reshape(1, 2 * group_size)

    return [
        ("all_zero_g96", all_zero, group_size, 1.0),
        ("eps_dominated_g96", eps_dominated, group_size, 1.0),
        (
            "fp16_extrema_and_quant_boundaries_g96",
            extrema_and_boundaries,
            group_size,
            1e-10,
        ),
    ]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "per_token_group_quant_int8"), "Missing per_token_group_quant_int8"
        assert hasattr(mod, "_per_token_group_quant_int8"), "Missing _per_token_group_quant_int8"
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

    cases = []
    for i, (M, N, group_size) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        x = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()
        cases.append((f"gaussian_{i + 1}", x, group_size, 1e-10))
    cases.extend(boundary_correctness_cases(torch, device))

    for case_name, x, group_size, eps in cases:
        M, N = x.shape
        try:
            x_q, x_s = mod.per_token_group_quant_int8(x, group_size, eps=eps)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_per_token_group_quant_int8(
                x, group_size, eps=eps
            )
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            # Check scales
            if not torch.allclose(x_s, ref_s, atol=1e-4, rtol=1e-3):
                max_diff = (x_s - ref_s).abs().max().item()
                return False, (
                    f"Case {case_name} (M={M}, N={N}, G={group_size}): "
                    f"scale max diff = {max_diff:.6f}"
                )

            # Check quantized values
            if not torch.allclose(x_q.float(), ref_q.float(), atol=1.0, rtol=0.0):
                max_diff = (x_q.float() - ref_q.float()).abs().max().item()
                return False, (
                    f"Case {case_name} (M={M}, N={N}, G={group_size}): "
                    f"quant max diff = {max_diff:.1f}"
                )
        except Exception as e:
            return False, (
                f"Case {case_name} (M={M}, N={N}, G={group_size}): exception: {e}"
            )

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
            x = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()

            def _bench_fn():
                mod.per_token_group_quant_int8(x, group_size)
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
            "num_shapes": len(TEST_SHAPES) + 3,
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
