#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/three_interpolate"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/three_interpolate"
ATOL, RTOL = 1e-4, 1e-4

# 5 test shapes: (B, C, M_source, N_query)
TEST_SHAPES = [
    (2, 8, 32, 16),
    (4, 32, 256, 128),
    (8, 64, 8192, 2048),
    (2, 128, 1024, 512),
    (4, 16, 4096, 1024),
]
PERF_SHAPE_IDX = 2


def cpu_reference(features, idx, weight):
    """output[b,c,n] = sum_k weight[b,n,k] * features[b,c,idx[b,n,k]]"""
    B, C, M = features.shape
    N = idx.shape[1]
    idx_long = idx.long()
    gathered = torch.stack([
        torch.gather(features, 2, idx_long[:, :, k].unsqueeze(1).expand(B, C, N))
        for k in range(3)
    ], dim=-1)  # (B, C, N, 3)
    return (gathered * weight.unsqueeze(1)).sum(dim=-1)


def run_compile():
    try:
        from kernel_loader import interpolate_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from three_interpolate_wrapper import three_interpolate

    for i, (B, C, M, N) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        features = torch.randn(B, C, M, device="cuda", dtype=torch.float32)
        idx = torch.randint(0, M, (B, N, 3), device="cuda", dtype=torch.int32)
        weight = torch.rand(B, N, 3, device="cuda", dtype=torch.float32)
        weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize

        gpu_out = three_interpolate(features, idx, weight)
        cpu_out = cpu_reference(features.cpu(), idx.cpu(), weight.cpu())

        if not torch.allclose(gpu_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL):
            diff = torch.max(torch.abs(gpu_out.cpu() - cpu_out)).item()
            return False, f"Shape {i+1} failed (B={B},C={C},M={M},N={N}): max_diff={diff:.6e}"

    return True, None


def run_performance():
    from three_interpolate_wrapper import three_interpolate

    B, C, M, N = TEST_SHAPES[PERF_SHAPE_IDX]
    features = torch.randn(B, C, M, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, M, (B, N, 3), device="cuda", dtype=torch.int32)
    weight = torch.rand(B, N, 3, device="cuda", dtype=torch.float32)
    weight = weight / weight.sum(dim=-1, keepdim=True)

    for _ in range(10):
        three_interpolate(features, idx, weight)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_iter = 100
    start.record()
    for _ in range(n_iter):
        three_interpolate(features, idx, weight)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


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
        report = {"execution_time_ms": elapsed_ms, "shape": list(TEST_SHAPES[PERF_SHAPE_IDX])}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
