# Vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d, MIT, see LICENSE):
#   aiter/ops/triton/_triton_kernels/moe/activations.py
# The upstream file carries no copyright header, so the repository LICENSE applies.
# Changes: the _swiglu parameter ``input`` renamed ``acc_in`` (the candidate rules reject the builtin name).

import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(acc_in, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.
    """
    gelu, linear = tl.split(tl.reshape(acc_in, (acc_in.shape[0], acc_in.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear
