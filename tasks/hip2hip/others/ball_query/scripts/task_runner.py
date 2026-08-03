#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/ball_query"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch
from _aka_benchmark import (
    benchmark_cuda_graph_or_events,
    hip_source_graph_capture_policy,
)

TASK_NAME = "hip2hip/ball_query"
HIP_GRAPH_ENABLED, HIP_GRAPH_FALLBACK_REASON = hip_source_graph_capture_policy(
    os.path.join(TASK_DIR, "src", "ball_query.cpp"),
    os.path.join(TASK_DIR, "src", "ball_query_cuda.hip"),
)

# 5 test shapes: (B, N, M, max_radius, nsample)
TEST_SHAPES = [
    (2, 256, 32, 1.0, 5),
    (4, 1024, 128, 0.5, 10),
    (4, 16384, 2048, 0.2, 5),
    (2, 4096, 512, 2.0, 16),
    (8, 2048, 256, 0.3, 8),
]


def cpu_reference(min_radius, max_radius, nsample, xyz, center_xyz):
    """CPU ball query: find points within [min_radius, max_radius) for each center."""
    B, N, _ = xyz.shape
    M = center_xyz.shape[1]
    idx = torch.zeros(B, M, nsample, dtype=torch.int32)

    for b in range(B):
        for m in range(M):
            dists = torch.norm(xyz[b] - center_xyz[b, m], dim=1)
            mask = (dists >= min_radius) & (dists < max_radius)
            valid = torch.where(mask)[0]
            if len(valid) == 0:
                idx[b, m, :] = 0
            else:
                n_valid = min(len(valid), nsample)
                idx[b, m, :n_valid] = valid[:n_valid].int()
                if n_valid < nsample:
                    idx[b, m, n_valid:] = valid[0].int()
    return idx


def validate_ball_query(gpu_idx, xyz, center_xyz, min_radius, max_radius):
    """Validate that all returned GPU indices point to valid in-range neighbors."""
    B, M, nsample = gpu_idx.shape
    for b in range(B):
        for m in range(M):
            dists = torch.norm(xyz[b] - center_xyz[b, m], dim=1)
            has_valid = ((dists >= min_radius) & (dists < max_radius)).any()
            for s in range(nsample):
                pt_idx = gpu_idx[b, m, s].item()
                if has_valid:
                    d = dists[pt_idx].item()
                    if not (min_radius <= d < max_radius or d < 1e-6):
                        # Allow index 0 as padding when no valid neighbors
                        if pt_idx != 0:
                            return False
    return True


def run_compile():
    try:
        from kernel_loader import ball_query_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from ball_query_wrapper import ball_query

    for i, (B, N, M, max_r, nsample) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        min_r = 0.0
        xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        center_xyz = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        gpu_idx = ball_query(min_r, max_r, nsample, xyz, center_xyz).cpu().int()

        # Compare against the CPU ball-query reference (mmcv semantics: the first
        # `nsample` in-range points in index order, with any remaining slots padded
        # by the first match). Exact comparison — the kernel is deterministic and
        # scans points in index order, so this matches element-for-element. The
        # previous within-radius-only check tolerated pt_idx==0 and would pass a
        # kernel that wrote nothing (output is zero-initialized).
        ref_idx = cpu_reference(min_r, max_r, nsample, xyz.cpu(), center_xyz.cpu())

        if not torch.equal(gpu_idx, ref_idx):
            mism = int((gpu_idx != ref_idx).sum().item())
            return False, (f"Shape {i+1} mismatch (B={B},N={N},M={M},r={max_r},"
                           f"nsample={nsample}): {mism} index/indices differ from CPU reference")

    return True, None


def _time_kernel(fn, n_warmup=10, n_iter=100):
    return benchmark_cuda_graph_or_events(
        fn, warmup=n_warmup, repetition=n_iter,
        use_cuda_graph=HIP_GRAPH_ENABLED,
        fallback_reason=HIP_GRAPH_FALLBACK_REASON,
    )


def run_performance():
    from ball_query_wrapper import ball_query

    test_cases = []
    
    for shape_idx, (B, N, M, max_r, nsample) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + shape_idx)
        xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        center_xyz = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        # Perf1: fixed radius ball query (min=0, max=max_r)
        ms_fixed, meta_fixed = _time_kernel(lambda: ball_query(0.0, max_r, nsample, xyz, center_xyz))
        # Perf2: dilated/annular ball query (min=max_r, max=max_r*2)
        ms_dilated, meta_dilated = _time_kernel(lambda: ball_query(max_r, max_r * 2, nsample, xyz, center_xyz))

        test_cases.append({
            "test_case_id": f"shape_{shape_idx}_fixed_radius",
            "execution_time_ms": ms_fixed,
            **meta_fixed,
            "params": {
                "B": B,
                "N": N,
                "M": M,
                "max_radius": max_r,
                "nsample": nsample,
                "query_type": "fixed_radius"
            }
        })
        test_cases.append({
            "test_case_id": f"shape_{shape_idx}_dilated_radius",
            "execution_time_ms": ms_dilated,
            **meta_dilated,
            "params": {
                "B": B,
                "N": N,
                "M": M,
                "max_radius": max_r,
                "nsample": nsample,
                "query_type": "dilated_radius"
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
        report = {"test_cases": test_cases}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        for case in test_cases:
            print(f"Perf: {case['execution_time_ms']:.4f} ms ({case['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
