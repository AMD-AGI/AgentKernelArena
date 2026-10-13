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
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2., 0.]]])
    points = torch.tensor([[[0., 0., 1.], [1., 0., 1.], [-1., 0., 1.],
                            [0., 1., 1.], [0., 0., 0.], [0., 0., 2.],
                            [0., 0., -1.]]])
    known_answer(h.cpu_points_in_boxes_part(points, boxes),
                 torch.tensor([[0, -1, -1, -1, 0, 0, -1]], dtype=torch.int32))
    known_answer(h.cpu_points_in_boxes_all(points, boxes),
                 torch.tensor([[[1], [0], [0], [0], [1], [1], [0]]], dtype=torch.int32))
    boxes = torch.tensor([[[0., 0., 0., 4., 1., 2., torch.pi / 2]]])
    points = torch.tensor([[[0., 1.5, 1.], [1.5, 0., 1.]]])
    known_answer(h.cpu_points_in_boxes_all(points, boxes), torch.tensor([[[1], [0]]], dtype=torch.int32))


def heading_control_inputs():
    """Independent ordered answers for both signs of broad box rotations."""
    headings = (-.7, -math.pi / 2, -2.2, -math.pi,
                .7, math.pi / 2, 2.2, math.pi)
    local = ((0., 0., 1.), (1.5, 0., 1.), (2.5, 0., 1.),
             (0., .75, 1.), (0., 0., 0.), (0., 0., 2.),
             (0., 0., -.5), (0., 0., 2.5))
    boxes, points = [], []
    for index, heading in enumerate(headings):
        center_x = float(index * 10)
        boxes.append(((center_x, 0., 0., 4., 1., 2., heading),))
        cosine, sine = math.cos(heading), math.sin(heading)
        points.append(tuple((center_x + x * cosine - y * sine,
                             x * sine + y * cosine, z) for x, y, z in local))
    boxes = torch.tensor(boxes, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)
    expected_part = torch.tensor([[0, 0, -1, -1, 0, 0, -1, -1]] * len(headings),
                                 dtype=torch.int32)
    expected_all = (expected_part == 0).to(torch.int32).unsqueeze(-1)
    return points, boxes, expected_part, expected_all


def unchanged_tensor_bytes(before, after):
    if (before.shape != after.shape or before.dtype != after.dtype
            or before.device != after.device or before.stride() != after.stride()
            or not torch.equal(before.contiguous().view(torch.uint8),
                               after.contiguous().view(torch.uint8))):
        raise ValueError("Points-in-boxes control changed a caller-owned input")


def check_heading_reference(h):
    points, boxes, expected_part, expected_all = heading_control_inputs()
    original_points, original_boxes = points.clone(), boxes.clone()
    try:
        known_answer(h.cpu_points_in_boxes_part(points, boxes), expected_part)
        unchanged_tensor_bytes(original_points, points)
        unchanged_tensor_bytes(original_boxes, boxes)
        known_answer(h.cpu_points_in_boxes_all(points, boxes), expected_all)
        unchanged_tensor_bytes(original_points, points)
        unchanged_tensor_bytes(original_boxes, boxes)
    finally:
        points.copy_(original_points)
        boxes.copy_(original_boxes)
    return points, boxes, expected_part, expected_all


def check_heading_controls(h, native_part, native_all, device="cuda"):
    points, boxes, expected_part, expected_all = check_heading_reference(h)
    points, boxes = points.to(device), boxes.to(device)
    original_points, original_boxes = points.clone(), boxes.clone()
    try:
        close(native_part(points, boxes), expected_part, gpu=device == "cuda")
        unchanged_tensor_bytes(original_points, points)
        unchanged_tensor_bytes(original_boxes, boxes)
        close(native_all(points, boxes), expected_all, gpu=device == "cuda")
        unchanged_tensor_bytes(original_points, points)
        unchanged_tensor_bytes(original_boxes, boxes)
    finally:
        points.copy_(original_points)
        boxes.copy_(original_boxes)


def check_additional_paths(h):
    from points_in_boxes_wrapper import points_in_boxes_part, points_in_boxes_all
    check_heading_controls(h, points_in_boxes_part, points_in_boxes_all)
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2., 0.]]], device="cuda")
    points = torch.tensor([[[0., 0., 1.], [1., 0., 1.], [-1., 0., 1.],
                            [0., 1., 1.], [0., 0., 0.], [0., 0., 2.],
                            [0., 0., -1.]]], device="cuda")
    close(points_in_boxes_part(points, boxes),
          torch.tensor([[0, -1, -1, -1, 0, 0, -1]], dtype=torch.int32), gpu=True)
    close(points_in_boxes_all(points, boxes),
          torch.tensor([[[1], [0], [0], [0], [1], [1], [0]]], dtype=torch.int32), gpu=True)
    for i, (B, T, M) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        boxes, points = h.generate_test_data(B, T, M, device="cuda")
        boxes, points = boxes.float(), points.float()
        rotated = boxes.clone()
        rotated[:, :, 6] = torch.rand(B, T, device=boxes.device) * 3.14
        for selected in (boxes, rotated):
            close(points_in_boxes_part(points, selected), h.cpu_points_in_boxes_part(points.cpu(), selected.cpu()), gpu=True)
            close(points_in_boxes_all(points, selected), h.cpu_points_in_boxes_all(points.cpu(), selected.cpu()), gpu=True)
