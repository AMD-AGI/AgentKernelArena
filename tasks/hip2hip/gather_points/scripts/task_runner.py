#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/gather_points"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/gather_points"
TOLERANCE = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}

# 5 test shapes: (B, C, N, M)
TEST_SHAPES = [
    (2, 16, 128, 32),
    (4, 32, 512, 64),
    (8, 64, 1024, 128),
    (2, 128, 2048, 256),
    (16, 3, 4096, 512),
]
PERF_SHAPE_IDX = 2


def cpu_reference(features, idx):
    """Pure PyTorch CPU reference for gather_points.
    output[b, :, m] = features[b, :, idx[b, m]]
    """
    B, C, N = features.shape
    M = idx.shape[1]
    idx_long = idx.long().unsqueeze(1).expand(B, C, M)
    return torch.gather(features, 2, idx_long)


def run_compile():
    try:
        from kernel_loader import gather_points_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from gather_points_wrapper import gather_points

    for i, (B, C, N, M) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        features = torch.randn(B, C, N, device="cuda", dtype=torch.float32)
        idx = torch.randint(0, N, (B, M), device="cuda", dtype=torch.int32)

        gpu_out = gather_points(features, idx)
        cpu_out = cpu_reference(features.cpu(), idx.cpu())

        tol = TOLERANCE[torch.float32]
        if not torch.allclose(gpu_out.cpu(), cpu_out, atol=tol["atol"], rtol=tol["rtol"]):
            return False, f"Shape {i+1} failed (B={B},C={C},N={N},M={M}): max_diff={torch.max(torch.abs(gpu_out.cpu() - cpu_out)).item():.6e}"

        # Also test fp16
        features_half = features.half()
        gpu_out_half = gather_points(features_half, idx)
        cpu_out_half = cpu_reference(features_half.cpu(), idx.cpu())

        tol_fp16 = TOLERANCE[torch.float16]
        if not torch.allclose(gpu_out_half.cpu(), cpu_out_half, atol=tol_fp16["atol"], rtol=tol_fp16["rtol"]):
            return False, f"Shape {i+1} fp16 failed (B={B},C={C},N={N},M={M})"

    return True, None


def _time_kernel(fn, n_warmup=10, n_iter=100):
    """Time a kernel function using CUDA events."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def run_performance():
    from gather_points_wrapper import gather_points

    B, C, N, M = TEST_SHAPES[PERF_SHAPE_IDX]
    features_fp32 = torch.randn(B, C, N, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, N, (B, M), device="cuda", dtype=torch.int32)
    features_fp16 = features_fp32.half()

    # Perf1: float32 forward
    ms_fp32 = _time_kernel(lambda: gather_points(features_fp32, idx))
    # Perf2: float16 forward
    ms_fp16 = _time_kernel(lambda: gather_points(features_fp16, idx))

    return ms_fp32, ms_fp16


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
        ms_fp32, ms_fp16 = run_performance()
        report = {
            "shape": list(TEST_SHAPES[PERF_SHAPE_IDX]),
            "perf1_fp32_ms": ms_fp32,
            "perf2_fp16_ms": ms_fp16,
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Perf: {ms_fp32:.4f} ms")
        print(f"Perf: {ms_fp16:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
