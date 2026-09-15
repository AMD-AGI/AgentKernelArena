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
    xyz = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [5., 0., 0.]]])
    center = torch.tensor([[[.25, 0., 0.]]])
    known_answer(h.cpu_reference(2, xyz, center), torch.tensor([[[0], [1]]], dtype=torch.int32))


def check_additional_paths(h):
    from knn_wrapper import knn
    for i, (B, N, M, k) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda")
        center = torch.randn(B, M, 3, device="cuda")
        for query, transposed in ((center, False), (center, True), (xyz, False)):
            actual = (knn(k, xyz.transpose(1, 2).contiguous(), query.transpose(1, 2).contiguous(), True)
                      if transposed else knn(k, xyz, query))
            expected = h.cpu_reference(k, xyz.cpu(), query.cpu())
            contract(actual, expected, gpu=True)
            close(actual.sort(dim=1).values, expected.sort(dim=1).values)
