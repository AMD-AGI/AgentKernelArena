# SPDX-License-Identifier: MIT
"""Starting implementation: the pinned AITER GEMM for the enrolled FP8 call.

Every enrolled invocation supplies an already quantized fp8 [m, k] activation
x and required fp32 per-1x128 x_scale [m, k/128] with stride (1, m). weight is
the load-time preshuffled fp8 [n, k] weight and weight_scale its fp32
[n/128, k/128] block scales. Returns bf16 [m, n]. The producer's RMSNorm and
activation quantization are outside this call and its timing.

The public x_scale=None default and BF16 fallback are retained for compatibility;
neither is used by the enrolled cases.

Replace this implementation and add helpers under source/ to reorganize the
complete operation. The protected runner owns inputs, oracle and timing.
"""


def fp8_blockscale_linear(x, weight, weight_scale, x_scale=None):
    import torch
    import aiter

    if x_scale is None:
        x, x_scale = aiter.get_hip_quant(aiter.QuantType.per_1x128)(
            x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True
        )
        # The transposed scale bytes are read through a column-major view.
        x_scale = torch.as_strided(x_scale, x_scale.shape, (1, x_scale.shape[0]))
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weight, x_scale, weight_scale, dtype=torch.bfloat16
    )
