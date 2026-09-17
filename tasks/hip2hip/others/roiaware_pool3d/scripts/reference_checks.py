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
    close(actual, expected, atol=1e-1, rtol=.2, gpu=gpu)


def check_timed_output(actual, expected, mode, *, gpu=True):
    if mode == 'max':
        full_output(actual, expected, gpu=gpu)
        close(actual.sum(), expected.sum(), atol=1e-1, rtol=.2)
    elif mode == 'avg':
        close(actual, expected, atol=1e-4, rtol=1e-3, gpu=gpu)
    else:
        raise ValueError('Unknown ROI pooling mode')


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
    rois = torch.tensor([[0., 0., 0., 2., 2., 2., 0.]])
    points = torch.tensor([[0., 0., 1.], [.5, 0., 1.], [4., 0., 1.]])
    features = torch.tensor([[2.], [4.], [100.]])
    for mode, value in (("max", 4.), ("avg", 3.)):
        known_answer(h.cpu_roiaware_pool3d(rois, points, features, 1, mode), torch.full((1, 1, 1, 1, 1), value))


def check_additional_paths(h):
    from kernel_loader import roiaware_pool3d_ext
    for i, (R, N, C, S) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        rois, points, features = h.generate_test_data(R, N, C, device="cuda")
        rois, points, features = rois.float(), points.float(), features.float()
        output = torch.empty((R, S, S, S, C), device="cuda")
        argmax = torch.empty_like(output, dtype=torch.int)
        indices = torch.empty((R, S, S, S, 128), device="cuda", dtype=torch.int)
        mask = torch.empty((R, N), device="cuda", dtype=torch.int)
        for mode_id, mode in enumerate(("max", "avg")):
            output.zero_(); argmax.zero_(); indices.zero_()
            roiaware_pool3d_ext.forward(rois, points, features, argmax, indices, output, mask, mode_id)
            expected = h.cpu_roiaware_pool3d(rois.cpu(), points.cpu(), features.cpu(), S, mode)
            contract(output, expected, gpu=True)
            # Preserve the original max-pool sum gate and avg pointwise gate.
            if mode == "max":
                close(output.sum(), expected.sum(), atol=1e-1, rtol=.2)
                full_output(output, expected, gpu=True)
            else:
                close(output, expected, atol=1e-4, rtol=1e-3)
