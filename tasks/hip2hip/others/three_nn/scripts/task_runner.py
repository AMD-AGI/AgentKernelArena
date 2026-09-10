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
from _aka_benchmark import (
    benchmark_cuda_graph_or_events,
    hip_source_graph_capture_policy,
)

TASK_NAME = "hip2hip/three_nn"
HIP_GRAPH_ENABLED, HIP_GRAPH_FALLBACK_REASON = hip_source_graph_capture_policy(
    os.path.join(TASK_DIR, "src", "three_nn.cpp"),
    os.path.join(TASK_DIR, "src", "three_nn_cuda.hip"),
)
ATOL, RTOL = 1e-4, 1e-4

# 5 test shapes: (B, N_target, M_source)
TEST_SHAPES = [
    (2, 32, 64),
    (4, 128, 256),
    (8, 1024, 2048),
    (2, 512, 4096),
    (4, 2048, 128),
]

# Correctness-only cases around the optimized launcher's two dispatch limits.
# Keeping these separate ensures correctness coverage cannot alter benchmark
# case identities or timing methodology.
TARGETED_CASES = [
    ("m255_ties_duplicates", 1, 5, 255),
    ("bn2048_m257", 2, 1024, 257),
    ("bn2049_m257", 3, 683, 257),
]


def cpu_reference(target, source):
    """Find 3 nearest neighbors and return sqrt(squared distances) and indices."""
    dist_sq = torch.cdist(target.float(), source.float()).pow(2)  # (B, N, M)
    dists_sq, idx = dist_sq.topk(3, dim=2, largest=False, sorted=True)
    return torch.sqrt(dists_sq), idx.int()


def make_targeted_case(case_idx, B, N, M):
    """Create float32 inputs with duplicate and equal-distance neighbors."""
    generator = torch.Generator().manual_seed(1000 + case_idx)
    target = torch.randn(B, N, 3, generator=generator, dtype=torch.float32)
    # Keep the background points away from the engineered query at the origin.
    source = torch.rand(B, M, 3, generator=generator, dtype=torch.float32) * 4.0 + 4.0

    target[:, 0] = 0.0
    source[:, 0] = 0.0
    source[:, 65] = 0.0
    source[:, 2] = torch.tensor([1.0, 0.0, 0.0])
    source[:, 67] = torch.tensor([-1.0, 0.0, 0.0])
    source[:, M - 1] = torch.tensor([0.0, 1.0, 0.0])
    return target.contiguous(), source.contiguous()


def check_case(name, target, source, three_nn):
    """Compare one case with the CPU reference, allowing valid tie choices."""
    gpu_dist, gpu_idx = three_nn(target.cuda(), source.cuda())
    cpu_dist, cpu_idx = cpu_reference(target, source)
    expected_shape = (target.shape[0], target.shape[1], 3)

    if gpu_dist.shape != expected_shape or gpu_idx.shape != expected_shape:
        return False, (f"{name} output shape mismatch: distances={tuple(gpu_dist.shape)}, "
                       f"indices={tuple(gpu_idx.shape)}, expected={expected_shape}")
    if gpu_dist.dtype != torch.float32 or gpu_idx.dtype != torch.int32:
        return False, (f"{name} output dtype mismatch: distances={gpu_dist.dtype}, "
                       f"indices={gpu_idx.dtype}")

    gpu_dist_cpu = gpu_dist.cpu()
    gpu_idx_cpu = gpu_idx.cpu()
    if not torch.allclose(gpu_dist_cpu, cpu_dist, atol=ATOL, rtol=RTOL):
        diff = torch.max(torch.abs(gpu_dist_cpu - cpu_dist)).item()
        return False, f"{name} distances failed: max_diff={diff:.6e}"

    # Equal-distance neighbors do not have a unique valid index ordering. If
    # indices differ, require every selected point to have the reference
    # distance for that rank.
    if not torch.equal(gpu_idx_cpu, cpu_idx):
        all_dists = torch.sqrt(torch.cdist(target.float(), source.float()).pow(2))
        if torch.any(gpu_idx_cpu < 0) or torch.any(gpu_idx_cpu >= source.shape[1]):
            return False, f"{name} returned an out-of-range source index"
        selected_dists = torch.gather(all_dists, 2, gpu_idx_cpu.long())
        if not torch.allclose(selected_dists, cpu_dist, atol=ATOL, rtol=RTOL):
            diff = torch.max(torch.abs(selected_dists - cpu_dist)).item()
            return False, f"{name} indices select incorrect distances: max_diff={diff:.6e}"

    return True, None


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

        ok, err = check_case(f"Shape {i+1} (B={B},N={N},M={M})",
                             target.cpu(), source.cpu(), three_nn)
        if not ok:
            return ok, err

    for case_idx, (name, B, N, M) in enumerate(TARGETED_CASES):
        target, source = make_targeted_case(case_idx, B, N, M)
        ok, err = check_case(name, target, source, three_nn)
        if not ok:
            return ok, err

    return True, None


def run_performance():
    from three_nn_wrapper import three_nn

    test_cases = []
    
    for shape_idx, (B, N, M) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + shape_idx)
        target = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        source = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        elapsed_ms, benchmark_meta = benchmark_cuda_graph_or_events(
            lambda: three_nn(target, source), warmup=10, repetition=100,
            use_cuda_graph=HIP_GRAPH_ENABLED,
            fallback_reason=HIP_GRAPH_FALLBACK_REASON,
        )
        
        test_cases.append({
            "test_case_id": f"shape_{shape_idx}",
            "execution_time_ms": elapsed_ms,
            **benchmark_meta,
            "params": {
                "B": B,
                "N_target": N,
                "M_source": M
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(TARGETED_CASES),
            "num_targeted_cases": len(TARGETED_CASES),
        }
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
            print(f"Performance: {case['execution_time_ms']:.4f} ms ({case['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
