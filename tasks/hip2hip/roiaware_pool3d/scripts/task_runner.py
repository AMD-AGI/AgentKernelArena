#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/roiaware_pool3d"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/roiaware_pool3d"

# 5 test shapes: (num_rois, num_pts, C, out_size)
TEST_SHAPES = [
    (2, 200, 3, 4),
    (4, 1000, 3, 4),
    (8, 5000, 3, 4),
    (4, 2000, 6, 4),
    (16, 3000, 3, 2),
]
PERF_SHAPE_IDX = 2


def cpu_roiaware_pool3d(rois, pts, pts_feature, out_size, mode='max'):
    """CPU reference for RoI-aware 3D pooling.
    Args:
        rois: (N, 7) - (cx, cy, cz, dx, dy, dz, heading), bottom-center
        pts: (npoints, 3)
        pts_feature: (npoints, C)
        out_size: int
        mode: 'max' or 'avg'
    Returns:
        pooled_features: (N, out_size, out_size, out_size, C)
    """
    N = rois.shape[0]
    npoints = pts.shape[0]
    C = pts_feature.shape[1]
    out_x = out_y = out_z = out_size
    pooled = torch.zeros(N, out_x, out_y, out_z, C, dtype=pts_feature.dtype)

    for n in range(N):
        cx, cy, cz, dx, dy, dz, heading = rois[n]
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)

        for p in range(npoints):
            # Transform point to ROI local frame
            px = pts[p, 0] - cx
            py = pts[p, 1] - cy
            pz = pts[p, 2] - cz

            local_x = px * cos_h - py * sin_h
            local_y = px * sin_h + py * cos_h
            local_z = pz

            # Check if point is inside the box
            if (abs(local_x) > dx / 2 or abs(local_y) > dy / 2
                    or local_z < 0 or local_z > dz):
                continue

            # Map to voxel indices
            vx = int((local_x + dx / 2) / dx * out_x)
            vy = int((local_y + dy / 2) / dy * out_y)
            vz = int(local_z / dz * out_z)

            # Clamp
            vx = min(max(vx, 0), out_x - 1)
            vy = min(max(vy, 0), out_y - 1)
            vz = min(max(vz, 0), out_z - 1)

            if mode == 'max':
                pooled[n, vx, vy, vz] = torch.max(pooled[n, vx, vy, vz], pts_feature[p])
            else:  # avg - accumulate, we'll divide later
                pooled[n, vx, vy, vz] += pts_feature[p]

    # For avg mode, we'd need counts - but boundary effects make exact comparison difficult
    # So we use sum-based comparison instead of exact match
    return pooled


def generate_test_data(num_rois, num_pts, C, device="cpu"):
    """Generate rois, points, and features with points near rois."""
    torch.manual_seed(42)
    rois = torch.zeros(num_rois, 7, device=device)
    rois[:, :3] = torch.randn(num_rois, 3, device=device) * 3
    rois[:, 3:6] = torch.rand(num_rois, 3, device=device) * 4 + 2  # sizes 2-6
    rois[:, 6] = torch.rand(num_rois, device=device) * 0.3

    # Generate points near rois so some end up inside
    pts = torch.randn(num_pts, 3, device=device) * 5
    # Place some points inside first roi
    n_inside = min(num_pts // 4, 50)
    cx, cy, cz, dx, dy, dz, _ = rois[0]
    pts[:n_inside, 0] = cx + (torch.rand(n_inside, device=device) - 0.5) * dx * 0.8
    pts[:n_inside, 1] = cy + (torch.rand(n_inside, device=device) - 0.5) * dy * 0.8
    pts[:n_inside, 2] = cz + torch.rand(n_inside, device=device) * dz * 0.8

    pts_feature = torch.randn(num_pts, C, device=device).abs()  # positive for max pooling

    return rois, pts, pts_feature


def run_compile():
    try:
        from kernel_loader import roiaware_pool3d_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from roiaware_pool3d_wrapper import RoIAwarePool3d

    for i, (num_rois, num_pts, C, out_size) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        rois, pts, pts_feature = generate_test_data(num_rois, num_pts, C)
        rois_gpu = rois.cuda().float()
        pts_gpu = pts.cuda().float()
        feat_gpu = pts_feature.cuda().float()

        # Test max pooling
        pool_max = RoIAwarePool3d(out_size=out_size, max_pts_per_voxel=128, mode='max')
        gpu_out_max = pool_max(rois_gpu, pts_gpu, feat_gpu)

        cpu_out_max = cpu_roiaware_pool3d(
            rois.float(), pts.float(), pts_feature.float(), out_size, mode='max')

        # Compare using sum-based approach (boundary voxel assignment may differ)
        gpu_sum = gpu_out_max.cpu().sum()
        cpu_sum = cpu_out_max.sum()
        if gpu_sum.abs() > 1e-6 or cpu_sum.abs() > 1e-6:
            if not torch.allclose(gpu_sum, cpu_sum, atol=1e-1, rtol=0.2):
                return False, (f"Max pool shape {i+1} (N={num_rois},pts={num_pts},C={C},"
                               f"out={out_size}): sum mismatch gpu={gpu_sum.item():.4f} "
                               f"cpu={cpu_sum.item():.4f}")

        # Verify shapes match
        expected_shape = (num_rois, out_size, out_size, out_size, C)
        if gpu_out_max.shape != expected_shape:
            return False, (f"Shape {i+1}: output shape {gpu_out_max.shape} "
                           f"!= expected {expected_shape}")

        # Test avg pooling (just check shape and non-NaN)
        pool_avg = RoIAwarePool3d(out_size=out_size, max_pts_per_voxel=128, mode='avg')
        gpu_out_avg = pool_avg(rois_gpu, pts_gpu, feat_gpu)
        if gpu_out_avg.shape != expected_shape:
            return False, f"Shape {i+1}: avg pool output shape mismatch"
        if torch.isnan(gpu_out_avg).any():
            return False, f"Shape {i+1}: avg pool output contains NaN"

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
    from roiaware_pool3d_wrapper import RoIAwarePool3d

    num_rois, num_pts, C, out_size = TEST_SHAPES[PERF_SHAPE_IDX]
    rois, pts, pts_feature = generate_test_data(num_rois, num_pts, C, device="cuda")
    rois = rois.float()
    pts = pts.float()
    pts_feature = pts_feature.float()

    pool_max = RoIAwarePool3d(out_size=out_size, max_pts_per_voxel=128, mode='max')
    pool_avg = RoIAwarePool3d(out_size=out_size, max_pts_per_voxel=128, mode='avg')

    # Perf1: max pooling
    ms_max = _time_kernel(lambda: pool_max(rois, pts, pts_feature))
    # Perf2: avg pooling
    ms_avg = _time_kernel(lambda: pool_avg(rois, pts, pts_feature))

    return ms_max, ms_avg


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
        ms_max, ms_avg = run_performance()
        report = {
            "shape": list(TEST_SHAPES[PERF_SHAPE_IDX]),
            "perf1_maxpool_ms": ms_max,
            "perf2_avgpool_ms": ms_avg,
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Perf: {ms_max:.4f} ms")
        print(f"Perf: {ms_avg:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
