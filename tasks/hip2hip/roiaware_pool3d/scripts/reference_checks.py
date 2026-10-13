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


def strict_max_output(actual, expected, *, gpu=False):
    # Every scored voxel/channel must reject a systematic output scaling.
    # Retain the historical full_output and sum gates alongside this control.
    close(actual, expected, atol=1e-4, rtol=1e-3, gpu=gpu)


def check_timed_output(actual, expected, mode, *, gpu=True):
    if mode == 'max':
        full_output(actual, expected, gpu=gpu)
        close(actual.sum(), expected.sum(), atol=1e-1, rtol=.2)
        strict_max_output(actual, expected, gpu=gpu)
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
    rois, points, features, expected = boundary_negative_control()
    for mode in ("max", "avg"):
        known_answer(h.cpu_roiaware_pool3d(rois, points, features, 1, mode),
                     expected[mode])
    rois, points, features, expected = all_negative_faces_control()
    for mode in ("max", "avg"):
        known_answer(h.cpu_roiaware_pool3d(rois, points, features, 1, mode),
                     expected[mode])


def boundary_negative_control(device="cpu"):
    """A horizontal face is excluded; bottom/top faces and negative features count."""
    rois = torch.tensor([[0., 0., 0., 2., 2., 2., 0.]], device=device)
    points = torch.tensor([[0., 0., 1.], [.5, 0., 1.], [1., 0., 1.],
                           [0., 0., 0.], [0., 0., 2.], [4., 0., 1.]], device=device)
    features = torch.tensor([[-2., 3.], [-4., -5.], [100., 100.],
                             [1., -7.], [2., -8.], [999., 999.]], device=device)
    return rois, points, features, {
        'max': torch.tensor([2., 3.]).reshape(1, 1, 1, 1, 2),
        'avg': torch.tensor([-.75, -4.25]).reshape(1, 1, 1, 1, 2),
    }


def all_negative_faces_control(device="cpu"):
    """Four excluded horizontal faces must not mask all-negative pooled values."""
    rois = torch.tensor([[0., 0., 0., 2., 2., 2., 0.]], device=device)
    points = torch.tensor([[0., 0., 0.], [.5, 0., 1.], [0., -.5, 2.],
                           [1., 0., 1.], [-1., 0., 1.], [0., 1., 1.],
                           [0., -1., 1.]], device=device)
    features = torch.tensor([[-7., -2.], [-5., -6.], [-3., -4.],
                             [101., 101.], [102., 102.], [103., 103.],
                             [104., 104.]], device=device)
    return rois, points, features, {
        'max': torch.tensor([-3., -2.]).reshape(1, 1, 1, 1, 2),
        'avg': torch.tensor([-5., -4.]).reshape(1, 1, 1, 1, 2),
    }


def negative_multivoxel_control(out_size, channels, device="cpu"):
    """Known negative answers at each scored voxel width, including its last bin."""
    if (out_size, channels) not in ((2, 3), (4, 3), (4, 6)):
        raise ValueError("The unscored control covers the scored size/channel branches")
    rois = torch.tensor([[0., 0., 0., 2., 2., 2., 0.]], device=device)
    near = -1. + 1. / out_size
    far = 1. - 1. / out_size
    points = torch.tensor([
        [near, near, 1. / out_size],
        [near + .1 / out_size, near, 1. / out_size],
        [far, far, 2. - 1. / out_size],
        [0., 1., 1. / out_size],  # excluded horizontal face
        [1., 0., 1. / out_size],
    ], device=device)
    features = torch.tensor([
        [-7., -2.], [-5., -6.], [-3., -4.], [100., 100.], [101., 101.],
    ], device=device).repeat(1, (channels + 1) // 2)[:, :channels].contiguous()
    max_expected = torch.zeros((1, out_size, out_size, out_size, channels))
    avg_expected = torch.zeros_like(max_expected)
    max_expected[0, 0, 0, 0] = torch.tensor([-5., -2.]).repeat((channels + 1) // 2)[:channels]
    avg_expected[0, 0, 0, 0] = torch.tensor([-6., -4.]).repeat((channels + 1) // 2)[:channels]
    for expected in (max_expected, avg_expected):
        expected[0, -1, -1, -1] = torch.tensor([-3., -4.]).repeat((channels + 1) // 2)[:channels]
    return rois, points, features, {'max': max_expected, 'avg': avg_expected}


def check_negative_multivoxel_paths(h, device="cuda"):
    """Exercise the native size-2/4 max and avg paths without adding score rows."""
    from kernel_loader import roiaware_pool3d_ext
    checked = []
    for out_size, channels in ((2, 3), (4, 3), (4, 6)):
        rois, points, features, known = negative_multivoxel_control(
            out_size, channels, device=device)
        for mode_id, mode in enumerate(("max", "avg")):
            # The manually constructed tensor is independent of the CPU voxel
            # implementation, so a shared reference bug cannot make it pass.
            expected = known[mode]
            known_answer(h.cpu_roiaware_pool3d(rois.cpu(), points.cpu(),
                                               features.cpu(), out_size, mode), expected)
            output = torch.zeros((1, out_size, out_size, out_size, channels), device=device)
            argmax = torch.zeros_like(output, dtype=torch.int)
            indices = torch.zeros((1, out_size, out_size, out_size, 128),
                                  device=device, dtype=torch.int)
            mask = torch.empty((1, len(points)), device=device, dtype=torch.int)
            before = tuple(value.clone() for value in (rois, points, features))
            roiaware_pool3d_ext.forward(rois, points, features, argmax, indices,
                                        output, mask, mode_id)
            for name, current, saved in zip(("rois", "points", "features"),
                                            (rois, points, features), before):
                if (current.shape != saved.shape or current.dtype != saved.dtype or
                        current.device != saved.device or current.stride() != saved.stride() or
                        not torch.equal(current.contiguous().view(torch.uint8),
                                        saved.contiguous().view(torch.uint8))):
                    raise ValueError(f"Native size-{out_size}/C{channels} {mode} changed caller-owned {name}")
            close(output, expected, atol=1e-4, rtol=1e-3, gpu=device == "cuda")
            checked.append((out_size, channels, mode))
    return checked


def check_additional_paths(h):
    from kernel_loader import roiaware_pool3d_ext
    check_negative_multivoxel_paths(h)
    for control in (boundary_negative_control, all_negative_faces_control):
        rois, points, features, known = control(device="cuda")
        output = torch.zeros((1, 1, 1, 1, 2), device="cuda")
        argmax = torch.zeros_like(output, dtype=torch.int)
        indices = torch.zeros((1, 1, 1, 1, 128), device="cuda", dtype=torch.int)
        mask = torch.empty((1, len(points)), device="cuda", dtype=torch.int)
        for mode_id, mode in enumerate(("max", "avg")):
            output.zero_(); argmax.zero_(); indices.zero_()
            roiaware_pool3d_ext.forward(rois, points, features, argmax, indices, output, mask, mode_id)
            close(output, known[mode], atol=1e-4, rtol=1e-3, gpu=True)
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
                strict_max_output(output, expected, gpu=True)
            else:
                close(output, expected, atol=1e-4, rtol=1e-3)
