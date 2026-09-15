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
    target = torch.zeros(1, 1, 3)
    source = torch.tensor([[[1., 0., 0.], [3., 0., 0.], [5., 0., 0.]]])
    distance, indices = h.cpu_reference(target, source)
    known_answer(distance, torch.tensor([[[1., 3., 5.]]]))
    known_answer(indices, torch.tensor([[[0, 1, 2]]], dtype=torch.int32))


def check_timed_output(actual, expected, target, source, *, gpu=True):
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise ValueError('Three-NN must return distances and indices')
    distance, indices = actual
    expected_distance, expected_indices = expected
    close(distance, expected_distance, atol=1e-4, rtol=1e-4, gpu=gpu)
    contract(indices, expected_indices, gpu=gpu)
    actual_indices = indices.cpu()
    if ((actual_indices < 0) | (actual_indices >= source.shape[1])).any():
        raise ValueError('Three-NN returned an out-of-range index')
    different = actual_indices != expected_indices
    if different.any():
        # Preserve the original tie allowance: a different selected point must
        # have the same distance as the reference index within absolute 1e-4.
        distances = torch.sqrt(torch.cdist(target.float(), source.float()).pow(2))
        selected = distances.gather(2, actual_indices.long())
        reference = distances.gather(2, expected_indices.long())
        if ((selected - reference).abs()[different] > 1e-4).any():
            raise ValueError('Three-NN index selects a point with the wrong distance')


def check_additional_paths(h):
    # The original gate allows alternate tied indices; retain that numerical
    # comparison and additionally reject invalid indices and output contracts.
    from three_nn_wrapper import three_nn
    for i, (B, N, M) in enumerate(h.TEST_SHAPES):
        torch.manual_seed(42 + i)
        target = torch.randn(B, N, 3, device="cuda")
        source = torch.randn(B, M, 3, device="cuda")
        distance, indices = three_nn(target, source)
        expected, expected_indices = h.cpu_reference(target.cpu(), source.cpu())
        close(distance, expected, atol=1e-4, rtol=1e-4, gpu=True)
        contract(indices, expected_indices, gpu=True)
        if ((indices < 0) | (indices >= M)).any():
            raise ValueError("Three-NN returned an out-of-range index")
