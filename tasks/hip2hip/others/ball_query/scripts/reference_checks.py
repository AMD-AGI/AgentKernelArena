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
    xyz = torch.tensor([[[0., 0., 0.], [.5, 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]])
    center = torch.zeros(1, 1, 3)
    known_answer(h.cpu_reference(0., 1., 3, xyz, center), torch.tensor([[[0, 1, 0]]], dtype=torch.int32))
    known_answer(h.cpu_reference(1., 2., 3, xyz, center), torch.tensor([[[2, 2, 2]]], dtype=torch.int32))


def check_additional_paths(h):
    from ball_query_wrapper import ball_query
    for i, (B, N, M, radius, nsample) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda")
        center = torch.randn(B, M, 3, device="cuda")
        for lower, upper in ((0., radius), (radius, radius * 2)):
            actual = ball_query(lower, upper, nsample, xyz, center)
            expected = h.cpu_reference(lower, upper, nsample, xyz.cpu(), center.cpu())
            close(actual, expected, gpu=True)
