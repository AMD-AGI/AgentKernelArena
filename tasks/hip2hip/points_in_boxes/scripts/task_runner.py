#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/points_in_boxes"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/points_in_boxes"

# 5 test shapes: (B, T_boxes, M_points)
TEST_SHAPES = [
    (1, 2, 16),
    (2, 4, 64),
    (4, 8, 256),
    (2, 16, 1024),
    (8, 4, 512),
]
PERF_SHAPE_IDX = 3


def generate_test_data(B, T, M, device="cpu"):
    """Generate boxes and points with some points guaranteed inside boxes."""
    torch.manual_seed(42)
    # Generate boxes: (cx, cy, cz, dx, dy, dz, rz)
    boxes = torch.zeros(B, T, 7, device=device)
    boxes[:, :, :3] = torch.randn(B, T, 3, device=device) * 5  # centers
    boxes[:, :, 3:6] = torch.rand(B, T, 3, device=device) * 4 + 1  # sizes 1-5
    boxes[:, :, 6] = torch.rand(B, T, device=device) * 0.5  # small rotation

    # Generate points: mix of random + some inside boxes
    points = torch.randn(B, M, 3, device=device) * 10
    # Place first few points inside first box of each batch
    for b in range(B):
        cx, cy, cz, dx, dy, dz, rz = boxes[b, 0]
        n_inside = min(M // 4, 10)
        points[b, :n_inside, 0] = cx + (torch.rand(n_inside, device=device) - 0.5) * dx * 0.5
        points[b, :n_inside, 1] = cy + (torch.rand(n_inside, device=device) - 0.5) * dy * 0.5
        points[b, :n_inside, 2] = cz + torch.rand(n_inside, device=device) * dz * 0.5

    return boxes, points


def cpu_points_in_boxes_part(points, boxes):
    """CPU: for each point, find first box containing it. Returns (B, M) int, -1 if no box."""
    B, M, _ = points.shape
    T = boxes.shape[1]
    result = torch.full((B, M), -1, dtype=torch.int32)
    for b in range(B):
        for m in range(M):
            for t in range(T):
                cx, cy, cz, dx, dy, dz, rz = boxes[b, t]
                px = points[b, m, 0] - cx
                py = points[b, m, 1] - cy
                pz = points[b, m, 2] - cz
                cos_r = torch.cos(-rz)
                sin_r = torch.sin(-rz)
                local_x = px * cos_r - py * sin_r
                local_y = px * sin_r + py * cos_r
                if (abs(local_x) <= dx / 2 and abs(local_y) <= dy / 2
                        and 0 <= pz <= dz):
                    result[b, m] = t
                    break
    return result


def cpu_points_in_boxes_all(points, boxes):
    """CPU: for each point-box pair, return 1 if inside. Returns (B, M, T) int."""
    B, M, _ = points.shape
    T = boxes.shape[1]
    result = torch.zeros(B, M, T, dtype=torch.int32)
    for b in range(B):
        for m in range(M):
            for t in range(T):
                cx, cy, cz, dx, dy, dz, rz = boxes[b, t]
                px = points[b, m, 0] - cx
                py = points[b, m, 1] - cy
                pz = points[b, m, 2] - cz
                cos_r = torch.cos(-rz)
                sin_r = torch.sin(-rz)
                local_x = px * cos_r - py * sin_r
                local_y = px * sin_r + py * cos_r
                if (abs(local_x) <= dx / 2 and abs(local_y) <= dy / 2
                        and 0 <= pz <= dz):
                    result[b, m, t] = 1
    return result


def run_compile():
    try:
        from kernel_loader import points_in_boxes_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from points_in_boxes_wrapper import points_in_boxes_part, points_in_boxes_all

    for i, (B, T, M) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        boxes, points = generate_test_data(B, T, M)
        boxes_gpu = boxes.cuda().float()
        points_gpu = points.cuda().float()

        # Test points_in_boxes_part
        gpu_part = points_in_boxes_part(points_gpu, boxes_gpu)
        cpu_part = cpu_points_in_boxes_part(points.float(), boxes.float())
        if not torch.all(gpu_part.cpu() == cpu_part):
            mismatch = (gpu_part.cpu() != cpu_part).sum().item()
            return False, f"points_in_boxes_part shape {i+1} (B={B},T={T},M={M}): {mismatch} mismatches"

        # Test points_in_boxes_all
        gpu_all = points_in_boxes_all(points_gpu, boxes_gpu)
        cpu_all = cpu_points_in_boxes_all(points.float(), boxes.float())
        if not torch.all(gpu_all.cpu() == cpu_all):
            mismatch = (gpu_all.cpu() != cpu_all).sum().item()
            return False, f"points_in_boxes_all shape {i+1} (B={B},T={T},M={M}): {mismatch} mismatches"

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
    from points_in_boxes_wrapper import points_in_boxes_part, points_in_boxes_all

    B, T, M = TEST_SHAPES[PERF_SHAPE_IDX]
    boxes, points = generate_test_data(B, T, M, device="cuda")
    boxes = boxes.float()
    points = points.float()

    # Generate rotated boxes (larger rotation angles)
    boxes_rotated = boxes.clone()
    boxes_rotated[:, :, 6] = torch.rand(B, T, device=boxes.device) * 3.14

    # Perf1: points_in_boxes_part with standard boxes
    ms_part_std = _time_kernel(lambda: points_in_boxes_part(points, boxes))

    # Perf2: points_in_boxes_part with rotated boxes
    ms_part_rot = _time_kernel(lambda: points_in_boxes_part(points, boxes_rotated))

    # Perf3: points_in_boxes_all with standard boxes
    ms_all_std = _time_kernel(lambda: points_in_boxes_all(points, boxes))

    # Perf4: points_in_boxes_all with rotated boxes
    ms_all_rot = _time_kernel(lambda: points_in_boxes_all(points, boxes_rotated))

    return ms_part_std, ms_part_rot, ms_all_std, ms_all_rot


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
        ms_part_std, ms_part_rot, ms_all_std, ms_all_rot = run_performance()
        report = {
            "shape": list(TEST_SHAPES[PERF_SHAPE_IDX]),
            "perf1_part_standard_ms": ms_part_std,
            "perf2_part_rotated_ms": ms_part_rot,
            "perf3_all_standard_ms": ms_all_std,
            "perf4_all_rotated_ms": ms_all_rot,
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Perf: {ms_part_std:.4f} ms")
        print(f"Perf: {ms_part_rot:.4f} ms")
        print(f"Perf: {ms_all_std:.4f} ms")
        print(f"Perf: {ms_all_rot:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
