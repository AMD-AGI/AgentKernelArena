#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/three_nn"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/three_nn"
ATOL, RTOL = 1e-4, 1e-4

# 5 test shapes: (B, N_target, M_source)
TEST_SHAPES = [
    (2, 32, 64),
    (4, 128, 256),
    (8, 1024, 2048),
    (2, 512, 4096),
    (4, 2048, 128),
]
PERF_SHAPE_IDX = 2


def cpu_reference(target, source):
    """Find 3 nearest neighbors and return sqrt(squared distances) and indices."""
    dist_sq = torch.cdist(target.float(), source.float()).pow(2)  # (B, N, M)
    dists_sq, idx = dist_sq.topk(3, dim=2, largest=False, sorted=True)
    return torch.sqrt(dists_sq), idx.int()


def run_compile():
    try:
        from kernel_loader import interpolate_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from three_nn_wrapper import three_nn

    for i, (B, N, M) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        target = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        source = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        gpu_dist, gpu_idx = three_nn(target, source)
        cpu_dist, cpu_idx = cpu_reference(target.cpu(), source.cpu())

        # Compare distances
        if not torch.allclose(gpu_dist.cpu(), cpu_dist, atol=ATOL, rtol=RTOL):
            diff = torch.max(torch.abs(gpu_dist.cpu() - cpu_dist)).item()
            return False, f"Shape {i+1} distances failed (B={B},N={N},M={M}): max_diff={diff:.6e}"

        # Compare indices - if indices differ, verify the distances at those indices match
        if not torch.all(gpu_idx.cpu() == cpu_idx):
            # Fallback: check that gpu indices produce same distances as cpu indices
            gpu_dists_check = torch.sqrt(
                torch.cdist(target.cpu().float(), source.cpu().float()).pow(2)
            )
            for b in range(B):
                for n in range(N):
                    for k in range(3):
                        g_idx = gpu_idx[b, n, k].cpu().item()
                        c_idx = cpu_idx[b, n, k].item()
                        if g_idx != c_idx:
                            g_dist = gpu_dists_check[b, n, g_idx]
                            c_dist = gpu_dists_check[b, n, c_idx]
                            if abs(g_dist - c_dist) > ATOL:
                                return False, f"Shape {i+1} indices differ with different distances at (b={b},n={n},k={k})"

    return True, None


def run_performance():
    from three_nn_wrapper import three_nn

    B, N, M = TEST_SHAPES[PERF_SHAPE_IDX]
    target = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
    source = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

    for _ in range(10):
        three_nn(target, source)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_iter = 100
    start.record()
    for _ in range(n_iter):
        three_nn(target, source)
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
