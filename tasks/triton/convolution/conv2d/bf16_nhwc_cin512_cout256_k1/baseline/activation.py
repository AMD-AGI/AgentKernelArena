"""Activation helpers used by the conv kernels' epilogues.

Excerpt of aiter/ops/triton/_triton_kernels/activation.py (ROCm/aiter): only `_tanh`, `_gelu_tanh`, `_relu` and
`_relu6`, copied unchanged. The upstream file carries no license header, so the repository LICENSE (MIT) applies;
the rest of that file (SiLU, fused quantization kernels) is not used here.
"""

import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


@triton.jit
def _relu6(x):
    return tl.minimum(tl.maximum(0.0, x), 6.0)
