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
    scores = torch.tensor([[[[2.], [2.]]]], requires_grad=True)
    points = torch.tensor([[[[3.]], [[5.]]]], requires_grad=True)
    centers = torch.tensor([[[[1.]], [[7.]]]], requires_grad=True)
    idx = torch.tensor([[[0, 1]]])
    result = h.cpu_assign_score_withk_forward_vectorized(scores, points, centers, idx)
    known_answer(result, torch.tensor([[[[4., 8.]]]]))
    result.sum().backward()
    known_answer(scores.grad, torch.tensor([[[[2.], [4.]]]]))
    known_answer(points.grad, torch.tensor([[[[2.]], [[2.]]]]))
    known_answer(centers.grad, torch.tensor([[[[-4.]], [[0.]]]]))


def check_additional_paths(h):
    from assign_score_withk_wrapper import assign_score_withk
    for i, (B, N0, N1, M, K, O) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        scores = torch.randn(B, N1, K, M, device="cuda", requires_grad=True)
        points = torch.randn(B, N0, M, O, device="cuda", requires_grad=True)
        centers = torch.randn(B, N0, M, O, device="cuda", requires_grad=True)
        idx = torch.randint(0, N0, (B, N1, K), device="cuda", dtype=torch.int64)
        cpu = [x.detach().cpu().requires_grad_() for x in (scores, points, centers)]
        expected = h.cpu_assign_score_withk_forward_vectorized(*cpu, idx.cpu())
        actual = assign_score_withk(scores, points, centers, idx, 'sum')
        close(actual, expected, atol=1e-3, rtol=1e-3, gpu=True)
        expected.sum().backward()
        actual.sum().backward()
        for value, reference in zip((scores, points, centers), cpu):
            close(value.grad, reference.grad, atol=1e-3, rtol=1e-3, gpu=True)
