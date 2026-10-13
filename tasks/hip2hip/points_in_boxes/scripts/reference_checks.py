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
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2., 0.]]])
    points = torch.tensor([[[0., 0., 1.], [2., 0., 1.], [0., 0., -1.]]])
    known_answer(h.cpu_points_in_boxes_part(points, boxes), torch.tensor([[0, -1, -1]], dtype=torch.int32))
    known_answer(h.cpu_points_in_boxes_all(points, boxes), torch.tensor([[[1], [0], [0]]], dtype=torch.int32))
    boxes = torch.tensor([[[0., 0., 0., 4., 1., 2., torch.pi / 2]]])
    points = torch.tensor([[[0., 1.5, 1.], [1.5, 0., 1.]]])
    known_answer(h.cpu_points_in_boxes_all(points, boxes), torch.tensor([[[1], [0]]], dtype=torch.int32))


def check_additional_paths(h):
    from points_in_boxes_wrapper import points_in_boxes_part, points_in_boxes_all
    for i, (B, T, M) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        boxes, points = h.generate_test_data(B, T, M, device="cuda")
        boxes, points = boxes.float(), points.float()
        rotated = boxes.clone()
        rotated[:, :, 6] = torch.rand(B, T, device=boxes.device) * 3.14
        for selected in (boxes, rotated):
            close(points_in_boxes_part(points, selected), h.cpu_points_in_boxes_part(points.cpu(), selected.cpu()), gpu=True)
            close(points_in_boxes_all(points, selected), h.cpu_points_in_boxes_all(points.cpu(), selected.cpu()), gpu=True)
