"""EDITABLE candidate workspace for `materialize_bpreshuffle_fp8_scale`.

This file is the ONLY writable path in the task dir. It is rebound onto the live seam
`sglang.srt.layers.quantization.fp8_utils:materialize_bpreshuffle_fp8_scale` by
`meta.candidate_bind` (kind="rebind"); the baseline leg never imports it.

WHAT THE OP MUST DO (physical-layout contract — not just values!)
----------------------------------------------------------------
The per-1x128 fp8 activation scale comes out of the producing quant kernel
(`aiter_per1x128_quant(..., transpose_scale=False)` / the fused AR+RMSNorm+quant kernel /
`fused_clamp_act_mul`) as a ROW-MAJOR ``[M, G]`` fp32 tensor (``stride == (G, 1)``).
The gfx950 CK `gemm_a8w8_blockscale_bpreshuffle` GEMM reads those bytes COLUMN-MAJOR, so the
seam must return a tensor that

  * has the SAME logical shape ``[M, G]`` and the SAME values, and
  * is physically transposed-contiguous: ``out.stride() == (1, M)``.

The baseline body below (`scale.t().contiguous().t()`) is exactly the copy the profiler charges
~26k launches of `direct_copy_kernel_cuda` to. Any candidate MUST preserve both properties --
`cases.call` asserts the stride contract, so a value-correct but row-major return FAILS the
unittest instead of silently breaking the GEMM end-to-end.

Non-2-D input (a per-tensor scalar scale) passes through untouched.
"""

import torch


def materialize_bpreshuffle_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Materialize the physical scale layout consumed by the gfx95 bpreshuffle GEMM."""
    return scale.t().contiguous().t() if scale.dim() == 2 else scale
