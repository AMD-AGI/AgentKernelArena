# SPDX-License-Identifier: MIT
"""Protected FP32-accumulation oracle for the BF16 partial projection."""


def reference(x, weight):
    import torch

    # Never use TF32 to construct the independent numerical reference.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return (x.float() @ weight.float().T).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def check_output(actual, expected, tolerance):
    import torch

    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != torch.bfloat16 or actual.device != expected.device):
        raise AssertionError("GEMM output shape, dtype, or device differs")
    if not torch.isfinite(actual).all():
        raise AssertionError("GEMM output contains nonfinite values")
    torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"])
    difference = (actual.float() - expected.float()).abs()
    return {"max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item()}
