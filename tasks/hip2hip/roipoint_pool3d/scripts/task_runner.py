#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/roipoint_pool3d"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/roipoint_pool3d"

# 5 test shapes: (B, N, C, M, nsample)
TEST_SHAPES = [
    (1, 100, 3, 2, 4),
    (2, 500, 6, 4, 16),
    (2, 5000, 6, 8, 4),
    (4, 2000, 3, 8, 32),
    (2, 10000, 6, 16, 8),
]
PERF_SHAPE_IDX = 2


def check_point_in_box(point, box):
    """Check if a 3D point is inside a rotated 3D box.
    Box format: (cx, cy, cz, dx, dy, dz, heading). (cx,cy,cz) is bottom center."""
    cx, cy, cz, dx, dy, dz, heading = box
    px = point[0] - cx
    py = point[1] - cy
    pz = point[2] - cz
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    local_x = px * cos_h - py * sin_h
    local_y = px * sin_h + py * cos_h
    return (abs(local_x) <= dx / 2 and abs(local_y) <= dy / 2 and 0 <= pz <= dz)


def cpu_roipoint_pool3d(points, point_features, boxes3d, nsample):
    """CPU reference: for each box, find contained points and sample nsample of them.
    Args:
        points: (B, N, 3)
        point_features: (B, N, C)
        boxes3d: (B, M, 7)
    Returns:
        pooled_features: (B, M, nsample, 3+C)
        pooled_empty_flag: (B, M) int
    """
    B, N, C = point_features.shape
    M = boxes3d.shape[1]
    pooled_features = torch.zeros(B, M, nsample, 3 + C, dtype=points.dtype)
    pooled_empty_flag = torch.zeros(B, M, dtype=torch.int32)

    for b in range(B):
        for m in range(M):
            box = boxes3d[b, m]
            # Find points inside this box
            inside_indices = []
            for n_idx in range(N):
                if check_point_in_box(points[b, n_idx], box):
                    inside_indices.append(n_idx)

            if len(inside_indices) == 0:
                pooled_empty_flag[b, m] = 1
            else:
                # Sample nsample points (repeat if fewer)
                sampled = []
                for s in range(nsample):
                    sampled.append(inside_indices[s % len(inside_indices)])
                for s, idx in enumerate(sampled):
                    pooled_features[b, m, s, :3] = points[b, idx]
                    pooled_features[b, m, s, 3:] = point_features[b, idx]

    return pooled_features, pooled_empty_flag


def generate_test_data(B, N, C, M, nsample, device="cpu"):
    """Generate boxes and points with some points guaranteed inside boxes."""
    torch.manual_seed(42)
    boxes3d = torch.zeros(B, M, 7, device=device)
    boxes3d[:, :, :3] = torch.randn(B, M, 3, device=device) * 3
    boxes3d[:, :, 3:6] = torch.rand(B, M, 3, device=device) * 4 + 2  # sizes 2-6
    boxes3d[:, :, 6] = torch.rand(B, M, device=device) * 0.3  # small rotation

    points = torch.randn(B, N, 3, device=device) * 10
    point_features = torch.randn(B, N, C, device=device)

    # Place some points inside the first box of each batch
    for b in range(B):
        cx, cy, cz, dx, dy, dz, heading = boxes3d[b, 0]
        n_inside = min(N // 4, nsample * 2)
        points[b, :n_inside, 0] = cx + (torch.rand(n_inside, device=device) - 0.5) * dx * 0.4
        points[b, :n_inside, 1] = cy + (torch.rand(n_inside, device=device) - 0.5) * dy * 0.4
        points[b, :n_inside, 2] = cz + torch.rand(n_inside, device=device) * dz * 0.8

    return points, point_features, boxes3d


def run_compile():
    try:
        from kernel_loader import roipoint_pool3d_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from roipoint_pool3d_wrapper import RoIPointPool3d

    for i, (B, N, C, M, nsample) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        points, point_features, boxes3d = generate_test_data(B, N, C, M, nsample)
        points_gpu = points.cuda().float()
        feats_gpu = point_features.cuda().float()
        boxes_gpu = boxes3d.cuda().float()

        pool = RoIPointPool3d(num_sampled_points=nsample)
        gpu_feat, gpu_flag = pool(points_gpu, feats_gpu, boxes_gpu)

        cpu_feat, cpu_flag = cpu_roipoint_pool3d(
            points.float(), point_features.float(), boxes3d.float(), nsample)

        # Compare empty flags exactly
        if not torch.all(gpu_flag.cpu() == cpu_flag):
            mismatch = (gpu_flag.cpu() != cpu_flag).sum().item()
            return False, f"Shape {i+1} (B={B},N={N},C={C},M={M}): empty_flag {mismatch} mismatches"

        # For non-empty boxes, compare pooled features
        # Due to potential ordering differences in point containment checks,
        # compare using sorted feature sums per box
        for b in range(B):
            for m_idx in range(M):
                if cpu_flag[b, m_idx] == 1:
                    continue  # empty box, skip
                gpu_sum = gpu_feat[b, m_idx].cpu().sum()
                cpu_sum = cpu_feat[b, m_idx].sum()
                if not torch.allclose(gpu_sum, cpu_sum, atol=1e-3, rtol=1e-3):
                    return False, (f"Shape {i+1} (B={B},N={N},C={C},M={M}): "
                                   f"box ({b},{m_idx}) feature sum mismatch "
                                   f"gpu={gpu_sum.item():.4f} cpu={cpu_sum.item():.4f}")

    return True, None


def run_performance():
    from roipoint_pool3d_wrapper import RoIPointPool3d

    B, N, C, M, nsample = TEST_SHAPES[PERF_SHAPE_IDX]
    points, point_features, boxes3d = generate_test_data(B, N, C, M, nsample, device="cuda")
    points = points.float()
    point_features = point_features.float()
    boxes3d = boxes3d.float()

    pool = RoIPointPool3d(num_sampled_points=nsample)

    for _ in range(10):
        pool(points, point_features, boxes3d)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_iter = 100
    start.record()
    for _ in range(n_iter):
        pool(points, point_features, boxes3d)
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
