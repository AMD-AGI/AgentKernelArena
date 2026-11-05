"""Independent FP32 dequantized GEMM, following AITER's upstream test math."""


def reference(x, canonical_weight, x_scale, weight_scale):
    import torch

    m, k = x.shape
    n = canonical_weight.shape[0]
    a = (x.float().reshape(m, k // 128, 128) * x_scale.float().unsqueeze(-1)).reshape(m, k)
    b = canonical_weight.float() * weight_scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    if tuple(b.shape) != (n, k):
        raise ValueError("Weight scales do not cover the declared 128x128 blocks")
    return (a @ b.t()).to(torch.bfloat16)


def check_output(actual, expected):
    import torch

    if not isinstance(actual, torch.Tensor):
        raise AssertionError("The complete call must return a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError("Output shape, dtype or device differs")
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise AssertionError("Nonfinite reference or output")
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    error = actual.float() - expected.float()
    norm = torch.linalg.vector_norm(expected.float()).item()
    error_norm = torch.linalg.vector_norm(error).item()
    relative = error_norm / norm if norm else (0.0 if error_norm == 0 else float("inf"))
    # The absolute allclose floor alone can accept zero at small scale.
    if relative > 0.01:
        raise AssertionError(f"Relative Frobenius error {relative} exceeds 0.01")
    return {"max_abs_error": error.abs().max().item(), "relative_frobenius_error": relative}
