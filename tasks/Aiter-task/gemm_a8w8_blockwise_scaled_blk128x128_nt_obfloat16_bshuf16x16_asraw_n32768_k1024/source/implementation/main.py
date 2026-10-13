from __future__ import annotations
from functools import partial as partial
import torch as torch

def _blockwise_scaled_baseline(
    a, b, a_scale, b_scale, *, out_dtype="bfloat16", a_scale_storage="plain"
):
    """Allocation-inclusive AITER baseline; run with SIKL_DISABLE_PROXY=1.

    Logical replay materializes column-major scales INSIDE the timed callback.
    This is not native-stride/persistent-out timing. Disable schema dumping too.
    """
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError(
            "AITER blockwise-scaled baseline requires matching E4M3FN operands"
        )
    if a_scale_storage == "plain":
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale

        return gemm_a8w8_blockscale(
            a, b, a_scale, b_scale, dtype=getattr(torch, out_dtype)
        )
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

    if a_scale_storage == "logical":
        a_scale = a_scale.T.contiguous().T
    return gemm_a8w8_blockscale_bpreshuffle(
        a, b, a_scale, b_scale, dtype=getattr(torch, out_dtype)
    )


_callable = partial(_blockwise_scaled_baseline, **{'out_dtype': 'bfloat16', 'a_scale_storage': 'raw'})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
