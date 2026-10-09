# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Independent CPU reference controls and checks for every measured native variant.

The original task_runner gates remain in force. These checks execute outside
formal timing; none reduce its shapes, tolerances, samples, or state resets.
"""
import math

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
    check_face_reference(h)


def face_control_inputs():
    """Exact membership and ordered pooled outputs for the native box predicate."""
    points = torch.tensor([[[
        0., 0., 1.,       # interior
    ], [1., 0., 1.],     # +x face: excluded
        [-1., 0., 1.],   # -x face: excluded
        [0., 1., 1.],    # +y face: excluded
        [0., -1., 1.],   # -y face: excluded
        [.5, 0., 0.],    # bottom z face: included
        [-.5, 0., 2.],   # top z face: included
        [0., 0., -.25],  # below bottom
        [0., 0., 2.25],  # above top
        [.25, .25, 1.],  # second interior point
        [4., 1.5, 1.],   # rotated box interior, unrotated exterior
        [4.75, 0., 1.],  # rotated box exterior, unrotated interior
    ]], dtype=torch.float32)
    features = torch.arange(1, 13, dtype=torch.float32).reshape(1, 12, 1)
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2., 0.],
                           [4., 0., 0., 4., 1., 2., math.pi / 2],
                           [10., 0., 0., 2., 2., 2., 0.]]], dtype=torch.float32)
    expected = torch.zeros((1, 3, 6, 4), dtype=torch.float32)
    for box_index, selected in ((0, (0, 5, 6, 9, 0, 5)),
                                (1, (10, 10, 10, 10, 10, 10))):
        for sample_index, point_index in enumerate(selected):
            expected[0, box_index, sample_index] = torch.cat(
                (points[0, point_index], features[0, point_index]))
    flags = torch.tensor([[0, 0, 1]], dtype=torch.int32)
    return points, features, boxes, expected, flags


def check_face_reference(h):
    points, features, boxes, expected, flags = face_control_inputs()
    axis_membership = (True, False, False, False, False, True, True,
                       False, False, True)
    for point_index, inside in enumerate(axis_membership):
        if bool(h.check_point_in_box(points[0, point_index], boxes[0, 0])) != inside:
            raise AssertionError(f"CPU box reference has wrong axis-face membership at point {point_index}")
    for point_index, inside in ((10, True), (11, False)):
        if bool(h.check_point_in_box(points[0, point_index], boxes[0, 1])) != inside:
            raise AssertionError(f"CPU box reference has wrong rotated membership at point {point_index}")
    actual, actual_flags = h.cpu_roipoint_pool3d(points, features, boxes, 6)
    close(actual, expected)
    close(actual_flags, flags)
    return points, features, boxes, expected, flags


def check_face_controls(h, native):
    points, features, boxes, expected, flags = check_face_reference(h)
    points_gpu, features_gpu, boxes_gpu = points.cuda(), features.cuda(), boxes.cuda()
    output = torch.zeros_like(expected, device="cuda")
    empty_flags = torch.zeros_like(flags, device="cuda")
    assignments = torch.empty((1, points.shape[1], boxes.shape[1]), device="cuda", dtype=torch.int)
    indices = torch.empty((1, boxes.shape[1], 6), device="cuda", dtype=torch.int)
    native.forward(points_gpu.contiguous(), boxes_gpu.contiguous(), features_gpu.contiguous(),
                   output, empty_flags, assignments, indices)
    close(output, expected, gpu=True)
    close(empty_flags, flags, gpu=True)


def check_additional_paths(h):
    from kernel_loader import roipoint_pool3d_ext
    check_face_controls(h, roipoint_pool3d_ext)
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
