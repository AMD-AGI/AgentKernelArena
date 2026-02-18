#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/furthest_point_sample"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/furthest_point_sample"

# 5 test shapes: (B, N, npoints)
TEST_SHAPES = [
    (2, 32, 4),
    (4, 128, 16),
    (2, 512, 32),
    (8, 1024, 64),
    (2, 2048, 128),
]
PERF_SHAPE_IDX = 3


def cpu_fps(points_xyz, num_points):
    """Greedy farthest point sampling starting from index 0."""
    B, N, _ = points_xyz.shape
    idx = torch.zeros(B, num_points, dtype=torch.int32)
    for b in range(B):
        distances = torch.full((N,), 1e10, dtype=torch.float32)
        farthest = 0
        for i in range(num_points):
            idx[b, i] = farthest
            centroid = points_xyz[b, farthest]
            dist = ((points_xyz[b] - centroid) ** 2).sum(dim=1)
            distances = torch.min(distances, dist)
            farthest = distances.argmax().item()
    return idx


def cpu_fps_with_dist(dist_matrix, num_points):
    """FPS using precomputed distance matrix."""
    B, N, _ = dist_matrix.shape
    idx = torch.zeros(B, num_points, dtype=torch.int32)
    for b in range(B):
        distances = torch.full((N,), 1e10, dtype=torch.float32)
        farthest = 0
        for i in range(num_points):
            idx[b, i] = farthest
            dist = dist_matrix[b, farthest]
            distances = torch.min(distances, dist)
            farthest = distances.argmax().item()
    return idx


def run_compile():
    try:
        from kernel_loader import furthest_point_sample_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from furthest_point_sample_wrapper import furthest_point_sample, furthest_point_sample_with_dist

    for i, (B, N, npoints) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)

        # Test basic FPS
        gpu_idx = furthest_point_sample(xyz, npoints)
        cpu_idx = cpu_fps(xyz.cpu(), npoints)

        if not torch.all(gpu_idx.cpu() == cpu_idx):
            mismatch = (gpu_idx.cpu() != cpu_idx).sum().item()
            return False, f"FPS shape {i+1} failed (B={B},N={N},npts={npoints}): {mismatch} mismatches"

        # Test FPS with distance matrix
        dist_matrix = torch.cdist(xyz, xyz).pow(2)
        gpu_idx_d = furthest_point_sample_with_dist(dist_matrix, npoints)
        cpu_idx_d = cpu_fps_with_dist(dist_matrix.cpu(), npoints)

        if not torch.all(gpu_idx_d.cpu() == cpu_idx_d):
            mismatch = (gpu_idx_d.cpu() != cpu_idx_d).sum().item()
            return False, f"FPS-dist shape {i+1} failed (B={B},N={N},npts={npoints}): {mismatch} mismatches"

    return True, None


def _time_kernel(fn, n_warmup=10, n_iter=100):
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
    from furthest_point_sample_wrapper import furthest_point_sample, furthest_point_sample_with_dist

    B, N, npoints = TEST_SHAPES[PERF_SHAPE_IDX]
    xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)

    # Perf1: FPS on raw point coordinates
    ms_coords = _time_kernel(lambda: furthest_point_sample(xyz, npoints))

    # Perf2: FPS with pre-computed distance matrix
    dist_matrix = torch.cdist(xyz, xyz).pow(2)
    ms_dist = _time_kernel(lambda: furthest_point_sample_with_dist(dist_matrix, npoints))

    return ms_coords, ms_dist


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
        ms_coords, ms_dist = run_performance()
        report = {
            "shape": list(TEST_SHAPES[PERF_SHAPE_IDX]),
            "perf1_fps_coords_ms": ms_coords,
            "perf2_fps_dist_ms": ms_dist,
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Perf: {ms_coords:.4f} ms")
        print(f"Perf: {ms_dist:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
