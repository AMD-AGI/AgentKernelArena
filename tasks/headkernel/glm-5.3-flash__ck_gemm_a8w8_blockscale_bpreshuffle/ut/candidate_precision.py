"""Additional candidate-only output-scale accuracy, evaluated on CPU FP64."""
import math


def candidate_error_metrics(actual,expected):
    import torch
    if actual.device.type!='cpu' or expected.device.type!='cpu':raise AssertionError('Candidate norms must be evaluated on CPU')
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or expected.dtype!=torch.bfloat16:raise AssertionError('Unexpected FP8 GEMM output ABI')
    # BF16 output unit roundoff u=2^-8. Use the same fixture-independent
    # global requirement as the reviewed dense checker: 2u/(1-u)=2/255.
    limit=2*(2.0**-8)/(1-2.0**-8)
    reference_norm=float(torch.linalg.vector_norm(expected.double()))
    error_norm=float(torch.linalg.vector_norm(actual.double()-expected.double()))
    if not math.isfinite(reference_norm) or not math.isfinite(error_norm):raise AssertionError('Nonfinite candidate/reference norm')
    ratio=error_norm/reference_norm if reference_norm else (0.0 if error_norm==0.0 else None)
    return {'reference_l2':reference_norm,'error_l2':error_norm,'normalized_l2':ratio,
        'normalized_l2_limit':limit,'scale_relative_pass':error_norm<=limit*reference_norm}


def require_candidate_accuracy(actual,expected):
    metrics=candidate_error_metrics(actual,expected)
    if not metrics['scale_relative_pass']:
        raise AssertionError('Scale-relative candidate error exceeds BF16 accuracy requirement: '+str(metrics))
    return metrics
