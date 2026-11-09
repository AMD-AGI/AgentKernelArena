import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _silu(x):
    return _silu_exp2(x)
