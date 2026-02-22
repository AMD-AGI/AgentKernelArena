#!/usr/bin/env python3
"""Task runner for triton2triton/triton_awq_dequantize"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_awq_dequantize"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_awq_dequantize.py")

# Test configurations: (K, N_packed, group_size)
# qweight shape: [K, N_packed], scales shape: [K//G, N_packed*8], zeros shape: [K//G, N_packed]
TEST_SHAPES = [
    (64, 8, 32),
    (128, 16, 32),
    (128, 16, 64),
    (256, 32, 128),
    (256, 32, 64),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_awq_dequantize(qweight, scales, zeros, group_size):
    """CPU reference: unpack 4-bit AWQ weights and dequantize."""
    import torch
    K, N_packed = qweight.shape
    N = N_packed * 8

    # AWQ order: [0, 4, 1, 5, 2, 6, 3, 7]
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]

    result = torch.zeros((K, N), dtype=scales.dtype, device="cpu")
    qweight_cpu = qweight.cpu().to(torch.int32)
    zeros_cpu = zeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()

    for row in range(K):
        group_idx = row // group_size
        for col_packed in range(N_packed):
            packed_val = qweight_cpu[row, col_packed].item()
            zero_packed = zeros_cpu[group_idx, col_packed].item()
            for bit_idx in range(8):
                awq_idx = awq_order[bit_idx]
                weight_val = (packed_val >> (awq_idx * 4)) & 0xF
                zero_val = (zero_packed >> (awq_idx * 4)) & 0xF
                out_col = col_packed * 8 + bit_idx
                scale_val = scales_cpu[group_idx, out_col].item()
                result[row, out_col] = (weight_val - zero_val) * scale_val

    return result.to(scales.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "awq_dequantize_triton"), "Missing awq_dequantize_triton"
        assert hasattr(mod, "awq_dequantize_kernel"), "Missing awq_dequantize_kernel"
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

    for i, (K, N_packed, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            N = N_packed * 8
            num_groups = K // group_size

            # Create random packed weights (int32, each packing 8x 4-bit values)
            qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            zeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)

            # Run Triton kernel
            result = mod.awq_dequantize_triton(qweight, scales, zeros)
            torch.cuda.synchronize()

            # Run reference
            ref = reference_awq_dequantize(qweight, scales, zeros, group_size)
            ref = ref.to(device)

            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (K={K}, N_packed={N_packed}, G={group_size}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (K={K}, N_packed={N_packed}, G={group_size}): "
                f"exception: {e}"
            )

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    K, N_packed, group_size = TEST_SHAPES[PERF_SHAPE_IDX]
    N = N_packed * 8
    num_groups = K // group_size

    torch.manual_seed(0)
    qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
    scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
    zeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)

    for _ in range(5):
        mod.awq_dequantize_triton(qweight, scales, zeros)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.awq_dequantize_triton(qweight, scales, zeros)
        end_events[j].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


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
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
