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



def full_output(actual, expected, *, gpu=False):
    # Preserve the task's original numerical tolerance while checking every
    # voxel/channel or sampled point/feature, not just a conserved total.
    close(actual, expected, atol=1e-3, rtol=1e-3, gpu=gpu)


def check_timed_output(actual, expected, *, gpu=True):
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise ValueError('ROI point pooling must expose features and empty flags')
    output, flag = actual
    expected_output, expected_flag = expected
    full_output(output, expected_output, gpu=gpu)
    close(flag, expected_flag, gpu=gpu)
    for b in range(expected_flag.shape[0]):
        for m in range(expected_flag.shape[1]):
            if expected_flag[b, m] != 1:
                close(output[b, m].sum(), expected_output[b, m].sum(), atol=1e-3, rtol=1e-3)


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
    points = torch.tensor([[[0., 0., 1.], [.5, 0., 1.]]])
    features = torch.tensor([[[2.], [4.]]])
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2., 0.], [10., 0., 0., 2., 2., 2., 0.]]])
    actual, flag = h.cpu_roipoint_pool3d(points, features, boxes, 3)
    expected = torch.zeros(1, 2, 3, 4)
    expected[0, 0] = torch.tensor([[0., 0., 1., 2.], [.5, 0., 1., 4.], [0., 0., 1., 2.]])
    known_answer(actual, expected)
    known_answer(flag, torch.tensor([[0, 1]], dtype=torch.int32))


def check_additional_paths(h):
    from kernel_loader import roipoint_pool3d_ext
    for i, (B, N, C, M, S) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        points, features, boxes = h.generate_test_data(B, N, C, M, S, device="cuda")
        points, features, boxes = points.float(), features.float(), boxes.float()
        output = torch.zeros((B, M, S, 3 + C), device="cuda")
        flag = torch.zeros((B, M), device="cuda", dtype=torch.int)
        assignments = torch.empty((B, N, M), device="cuda", dtype=torch.int)
        indices = torch.empty((B, M, S), device="cuda", dtype=torch.int)
        roipoint_pool3d_ext.forward(points.contiguous(), boxes.view(B, -1, 7).contiguous(), features.contiguous(), output, flag, assignments, indices)
        expected, expected_flag = h.cpu_roipoint_pool3d(points.cpu(), features.cpu(), boxes.cpu(), S)
        contract(output, expected, gpu=True)
        close(flag, expected_flag, gpu=True)
        full_output(output, expected, gpu=True)
        for b in range(B):
            for m in range(M):
                if expected_flag[b, m] != 1:
                    close(output[b, m].sum(), expected[b, m].sum(), atol=1e-3, rtol=1e-3)
