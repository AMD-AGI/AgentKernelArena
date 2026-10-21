#!/usr/bin/env python3
"""Task runner for triton2triton/triton_awq_gemm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_awq_gemm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_awq_gemm.py")

# Test configs: (M, K, N_packed, group_size, split_k)
TEST_SHAPES = [
    (1, 8192, 1280, 128, 8),
    (1, 8192, 1024, 128, 8),
    (1, 8192, 7168, 128, 8),
    (1, 28672, 1024, 128, 8),
    (32, 28672, 1024, 128, 8),
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


def reference_awq_gemm(input_tensor, qweight, scales, qzeros, group_size):
    """CPU reference: dequantize then matmul (the original loops, vectorized)."""
    import torch
    torch.set_num_threads(min(32, torch.get_num_threads()))  # shared host
    K, N_packed = qweight.shape
    N = N_packed * 8
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    shifts = torch.tensor([awq_order[b] * 4 for b in range(8)], dtype=torch.int32)

    qweight_cpu = qweight.cpu().to(torch.int32)
    qzeros_cpu = qzeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()

    dequant = torch.empty((K, N), dtype=torch.float32, device="cpu")
    rows = max(group_size, (1 << 26) // max(N, 1) // group_size * group_size)
    for start in range(0, K, rows):
        end = min(start + rows, K)
        groups = torch.arange(start, end) // group_size
        weight_val = ((qweight_cpu[start:end].unsqueeze(-1) >> shifts) & 0xF).reshape(end - start, N)
        zero_val = ((qzeros_cpu[groups].unsqueeze(-1) >> shifts) & 0xF).reshape(end - start, N)
        dequant[start:end] = (weight_val - zero_val).float() * scales_cpu[groups]

    input_cpu = input_tensor.cpu().float()
    result = input_cpu @ dequant
    return result.to(scales.dtype)


def make_awq_weights(K, N, group_size, device):
    """AWQ-quantize a Gaussian weight with std 1/sqrt(K): per group of rows, asymmetric
    4-bit with scale (max - min) / 15, packed in AWQ nibble order."""
    import torch
    groups = K // group_size
    weight = torch.randn(K, N, device=device, dtype=torch.float32) * K ** -0.5
    grouped = weight.reshape(groups, group_size, N)
    wmin, wmax = grouped.amin(dim=1), grouped.amax(dim=1)
    scales = ((wmax - wmin) / 15).clamp_min(1e-4)
    zeros = (-wmin / scales).round().clamp(0, 15)
    q = (grouped / scales[:, None, :] + zeros[:, None, :]).round().clamp(0, 15).reshape(K, N)
    del weight, grouped
    shifts = torch.tensor([o * 4 for o in (0, 4, 1, 5, 2, 6, 3, 7)], device=device, dtype=torch.int64)

    def pack(values):
        packed = (values.to(torch.int64).reshape(values.shape[0], -1, 8) << shifts).sum(-1)
        return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)

    return pack(q), scales.to(torch.float16), pack(zeros)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "awq_gemm_triton"), "Missing awq_gemm_triton"
        assert hasattr(mod, "awq_gemm_kernel"), "Missing awq_gemm_kernel"
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

    for i, (M, K, N_packed, group_size, split_k) in enumerate(TEST_SHAPES):
        if case_index is not None and i != case_index:
            continue
        try:
            torch.manual_seed(42 + i)
            N = N_packed * 8
            num_groups = K // group_size

            input_tensor = torch.randn(M, K, device=device, dtype=dtype)
            qweight, scales, qzeros = make_awq_weights(K, N, group_size, device)

            result = mod.awq_gemm_triton(input_tensor, qweight, scales, qzeros, split_k)
            torch.cuda.synchronize()

            ref = reference_awq_gemm(input_tensor, qweight, scales, qzeros, group_size).to(device)

            if not torch.allclose(result, ref, atol=1e-1, rtol=1e-1):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, K={K}, N_packed={N_packed}, G={group_size}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (M={M}, K={K}, N_packed={N_packed}, G={group_size}): "
                f"exception: {e}"
            )

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

    for test_idx, (M, K, N_packed, group_size, split_k) in enumerate(TEST_SHAPES):
        try:
            N = N_packed * 8
            num_groups = K // group_size

            torch.manual_seed(0)
            input_tensor = torch.randn(M, K, device=device, dtype=dtype)
            qweight, scales, qzeros = make_awq_weights(K, N, group_size, device)

            def _bench_fn():
                mod.awq_gemm_triton(input_tensor, qweight, scales, qzeros, split_k)
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
                    "N_packed": N_packed,
                    "group_size": group_size,
                    "split_k": split_k
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "N_packed": N_packed,
                    "group_size": group_size,
                    "split_k": split_k
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
