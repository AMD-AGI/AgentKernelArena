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


def boundary_controls():
    # Exact binary radii make the inclusive/exclusive boundaries unambiguous.
    # A near-zero nonzero point must not use MMCV's exact d2 == 0 exception.
    center = torch.zeros(1, 1, 3)
    xyz = torch.tensor([[[2., 0., 0.], [0., 0., 0.], [1e-4, 0., 0.],
                         [1.5, 0., 0.], [1., 0., 0.], [-1.5, 0., 0.],
                         [.5, 0., 0.], [0., 0., 0.]]])
    for lower, upper, count, indices in (
        (1., 2., 7, [1, 3, 4, 5, 7, 1, 1]),
        (1., 2., 3, [1, 3, 4]),
        (0., 1., 6, [1, 2, 6, 7, 1, 1]),
    ):
        yield lower, upper, count, xyz, center, torch.tensor([[indices]], dtype=torch.int32)
    near = torch.tensor([[[1e-4, 0., 0.], [0., 0., 0.], [-1e-4, 0., 0.]]])
    yield 1., 2., 3, near, center, torch.tensor([[[1, 1, 1]]], dtype=torch.int32)
    one, two = torch.tensor(1.), torch.tensor(2.)
    edges = torch.stack((torch.nextafter(one, torch.tensor(0.)), one,
                         torch.nextafter(one, two), torch.nextafter(two, one),
                         two, torch.nextafter(two, torch.tensor(3.))))
    boundary_xyz = torch.zeros(1, 6, 3)
    boundary_xyz[0, :, 0] = edges
    yield 1., 2., 5, boundary_xyz, center, torch.tensor([[[1, 2, 3, 1, 1]]], dtype=torch.int32)


def self_test(h):
    for lower, upper, count, xyz, center, expected in boundary_controls():
        known_answer(h.cpu_reference(lower, upper, count, xyz, center), expected)
    # No match retains the wrapper's zero-initialized output, independently of
    # the nonzero known answers above which reject a do-nothing implementation.
    close(h.cpu_reference(1., 2., 3, torch.full((1, 2, 3), 10.), torch.zeros(1, 1, 3)),
          torch.zeros(1, 1, 3, dtype=torch.int32))


def check_additional_paths(h):
    from ball_query_wrapper import ball_query
    # Same small exact-index controls exercise the real native implementation.
    for lower, upper, count, xyz, center, expected in boundary_controls():
        close(ball_query(lower, upper, count, xyz.cuda(), center.cuda()), expected, gpu=True)
    for i, (B, N, M, radius, nsample) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda")
        center = torch.randn(B, M, 3, device="cuda")
        for lower, upper in ((0., radius), (radius, radius * 2)):
            actual = ball_query(lower, upper, nsample, xyz, center)
            expected = h.cpu_reference(lower, upper, nsample, xyz.cpu(), center.cpu())
            close(actual, expected, gpu=True)
