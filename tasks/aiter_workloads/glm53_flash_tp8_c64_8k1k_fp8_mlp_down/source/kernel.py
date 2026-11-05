"""Initial implementation of the complete upstream AITER FP8 GEMM call."""


def mlp_down_gemm(x, weight, x_scale, weight_scale):
    import aiter
    import torch

    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weight, x_scale, weight_scale, dtype=torch.bfloat16
    )
