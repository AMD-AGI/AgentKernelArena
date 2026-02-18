#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/assign_score_withk"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/assign_score_withk"

# 5 test shapes: (B, N0, N1, M, K, O)
# N0 = num source points, N1 = num query points (npoint), M = weight matrices, K = neighbors, O = out_dim
TEST_SHAPES = [
    (2, 8, 4, 2, 4, 4),
    (2, 32, 16, 4, 8, 8),
    (2, 128, 64, 8, 16, 16),
    (4, 256, 128, 4, 8, 32),
    (2, 512, 256, 8, 16, 8),
]
PERF_SHAPE_IDX = 2


def cpu_assign_score_withk_forward(scores, point_features, center_features, knn_idx):
    """CPU reference for assign_score_withk forward (sum aggregation).
    Args:
        scores: (B, N1, K, M)
        point_features: (B, N0, M, O)
        center_features: (B, N0, M, O)
        knn_idx: (B, N1, K) - indices into N0
    Returns:
        output: (B, O, N1, K)
    """
    B, N1, K, M = scores.shape
    N0 = point_features.shape[1]
    O = point_features.shape[3]

    output = torch.zeros(B, O, N1, K, dtype=scores.dtype)
    for b in range(B):
        for n in range(N1):
            center_idx = knn_idx[b, n, 0]  # first idx is center
            for k in range(K):
                nb_idx = knn_idx[b, n, k]
                for o in range(O):
                    val = 0.0
                    for m in range(M):
                        # neighbor kernel: (point - center, point)
                        diff = point_features[b, nb_idx, m, o] - center_features[b, center_idx, m, o]
                        feat = point_features[b, nb_idx, m, o]
                        val += scores[b, n, k, m] * (diff + feat)
                    output[b, o, n, k] = val
    return output


def cpu_assign_score_withk_forward_vectorized(scores, point_features, center_features, knn_idx):
    """Vectorized CPU reference for assign_score_withk forward (sum aggregation).
    Args:
        scores: (B, N1, K, M)
        point_features: (B, N0, M, O)
        center_features: (B, N0, M, O)
        knn_idx: (B, N1, K)
    Returns:
        output: (B, O, N1, K)
    """
    B, N1, K, M = scores.shape
    O = point_features.shape[3]

    # Gather neighbor and center features using knn_idx
    # knn_idx: (B, N1, K) -> expand for gathering from (B, N0, M, O)
    center_idx = knn_idx[:, :, 0:1].expand(-1, -1, K)  # (B, N1, K) - center for all k

    # Expand indices for gathering: (B, N1, K) -> (B, N1, K, M, O)
    nb_idx_exp = knn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, M, O)  # (B, N1, K, M, O)
    ct_idx_exp = center_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, M, O)

    # Gather: point_features (B, N0, M, O) -> (B, N1, K, M, O)
    pf_exp = point_features.unsqueeze(2).expand(-1, -1, K, -1, -1)  # broadcast won't work, use gather
    pf_gathered = torch.gather(
        point_features.unsqueeze(1).expand(-1, N1, -1, -1, -1),
        2, nb_idx_exp.long()
    )  # (B, N1, K, M, O)
    cf_gathered = torch.gather(
        center_features.unsqueeze(1).expand(-1, N1, -1, -1, -1),
        2, ct_idx_exp.long()
    )  # (B, N1, K, M, O)

    # neighbor kernel: point - center
    kernel_feat = pf_gathered - cf_gathered  # (B, N1, K, M, O)

    # Weighted sum over M: scores (B, N1, K, M) * kernel_feat (B, N1, K, M, O) -> sum over M
    weighted = scores.unsqueeze(-1) * kernel_feat  # (B, N1, K, M, O)
    output = weighted.sum(dim=3)  # (B, N1, K, O)

    # Reshape to (B, O, N1, K)
    output = output.permute(0, 3, 1, 2).contiguous()
    return output


def run_compile():
    try:
        from kernel_loader import assign_score_withk_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from assign_score_withk_wrapper import assign_score_withk

    for i, (B, N0, N1, M, K, O) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        scores = torch.randn(B, N1, K, M, device="cuda", dtype=torch.float32)
        point_features = torch.randn(B, N0, M, O, device="cuda", dtype=torch.float32)
        center_features = torch.randn(B, N0, M, O, device="cuda", dtype=torch.float32)
        knn_idx = torch.randint(0, N0, (B, N1, K), device="cuda", dtype=torch.int64)

        gpu_out = assign_score_withk(scores, point_features, center_features, knn_idx, 'sum')
        cpu_out = cpu_assign_score_withk_forward_vectorized(
            scores.cpu(), point_features.cpu(), center_features.cpu(), knn_idx.cpu())

        if not torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-3, rtol=1e-3):
            max_diff = (gpu_out.cpu() - cpu_out).abs().max().item()
            return False, (f"Forward shape {i+1} (B={B},N0={N0},N1={N1},M={M},K={K},O={O}): "
                           f"max_diff={max_diff:.6f}")

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
    from assign_score_withk_wrapper import assign_score_withk

    B, N0, N1, M, K, O = TEST_SHAPES[PERF_SHAPE_IDX]
    scores = torch.randn(B, N1, K, M, device="cuda", dtype=torch.float32, requires_grad=True)
    point_features = torch.randn(B, N0, M, O, device="cuda", dtype=torch.float32, requires_grad=True)
    center_features = torch.randn(B, N0, M, O, device="cuda", dtype=torch.float32, requires_grad=True)
    knn_idx = torch.randint(0, N0, (B, N1, K), device="cuda", dtype=torch.int64)

    # Perf1: forward pass
    ms_fwd = _time_kernel(lambda: assign_score_withk(scores, point_features, center_features, knn_idx, 'sum'))

    # Perf2: backward pass
    def fwd_bwd():
        out = assign_score_withk(scores, point_features, center_features, knn_idx, 'sum')
        loss = out.sum()
        loss.backward()

    ms_fwd_bwd = _time_kernel(fwd_bwd)

    return ms_fwd, ms_fwd_bwd


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
        ms_fwd, ms_fwd_bwd = run_performance()
        report = {
            "shape": list(TEST_SHAPES[PERF_SHAPE_IDX]),
            "perf1_forward_ms": ms_fwd,
            "perf2_forward_backward_ms": ms_fwd_bwd,
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Perf: {ms_fwd:.4f} ms")
        print(f"Perf: {ms_fwd_bwd:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
