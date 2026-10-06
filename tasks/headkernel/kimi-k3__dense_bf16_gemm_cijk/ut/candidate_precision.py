"""Candidate-only BF16 accuracy checks; native calibration remains separate."""
import math


def candidate_error_metrics(actual, expected):
    """Measure CPU FP64 Frobenius error without an absolute output-scale floor."""
    import torch
    if actual.device.type != 'cpu' or expected.device.type != 'cpu':
        raise AssertionError('Candidate accuracy requires frozen CPU observations')
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise AssertionError('Unexpected Kimi dense output dtype')
    if actual.shape != expected.shape:
        raise AssertionError('Candidate/reference shape differs')
    # BF16 unit roundoff is 2^-8. Two final BF16 roundings give the
    # output-precision scale 2u/(1-u) relative to the rounded reference.
    # This independent accuracy requirement is not fitted to probe errors.
    unit_roundoff = 2.0**-8
    limit = 2 * unit_roundoff / (1 - unit_roundoff)
    reference_norm = float(torch.linalg.vector_norm(expected.double()))
    error_norm = float(torch.linalg.vector_norm(actual.double() - expected.double()))
    if not math.isfinite(reference_norm) or not math.isfinite(error_norm):
        raise AssertionError('Nonfinite candidate/reference norm')
    ratio = error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0.0 else None)
    return {'reference_l2': reference_norm, 'error_l2': error_norm,
            'normalized_l2': ratio, 'normalized_l2_limit': limit,
            'scale_relative_pass': error_norm <= limit * reference_norm}


def candidate_close(actual, expected):
    import torch
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    metrics = candidate_error_metrics(actual, expected)
    if not metrics['scale_relative_pass']:
        raise AssertionError('Scale-relative candidate error exceeds BF16 accuracy requirement: ' + str(metrics))
    return metrics
