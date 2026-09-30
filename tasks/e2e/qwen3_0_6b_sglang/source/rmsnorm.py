"""Initial production kernels; optimize the GPU implementation of these APIs.

Inputs are contiguous FP16 tensors. Preserve shapes, dtypes, allocation and
residual semantics. The protected service integration calls exactly these APIs.
"""
import aiter


def rms_norm(x, weight, eps):
    return aiter.rmsnorm2d_fwd(x, weight, eps)


def fused_add_rms_norm(output, x, residual, residual_out, weight, eps):
    aiter.rmsnorm2d_fwd_with_add(output, x, residual, residual_out, weight, eps)
