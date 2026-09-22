# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Independent CPU reference controls and checks for every measured native variant.

The original task_runner gates remain in force. These checks execute outside
formal timing; none reduce its shapes, tolerances, samples, or state resets.
"""
import torch


def contract(actual, expected, *, gpu=False):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Output shape/dtype does not match the reference contract")
    if gpu and actual.device.type != "cuda":
        raise ValueError("Native output must remain on the GPU")
    if not torch.isfinite(actual).all():
        raise ValueError("Output contains NaN/Inf")


def close(actual, expected, *, atol=0, rtol=0, gpu=False):
    contract(actual, expected, gpu=gpu)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)


def known_answer(actual, expected):
    close(actual, expected)
    # A nontrivial known answer must reject a fabricated all-zero result.
    if not torch.count_nonzero(expected):
        raise ValueError("Reference control must contain a nonzero known answer")
    try:
        close(torch.zeros_like(actual), expected)
    except AssertionError:
        return
    raise AssertionError("Reference comparison accepted a zero-output negative control")


def self_test(h):
    xyz = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]]])
    expected = torch.tensor([[0, 2, 1]], dtype=torch.int32)
    known_answer(h.cpu_fps(xyz, 3), expected)
    known_answer(h.cpu_fps_with_dist(torch.cdist(xyz, xyz).square(), 3), expected)


def check_additional_paths(h):
    # Original correctness already covers both coordinate and distance variants.
    # Re-run their output contracts as well as the unchanged exact-index gate.
    from furthest_point_sample_wrapper import furthest_point_sample, furthest_point_sample_with_dist
    for i, (B, N, npoint) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda")
        distance = torch.cdist(xyz, xyz).square()
        close(furthest_point_sample(xyz, npoint), h.cpu_fps(xyz.cpu(), npoint), gpu=True)
        close(furthest_point_sample_with_dist(distance, npoint), h.cpu_fps_with_dist(distance.cpu(), npoint), gpu=True)
