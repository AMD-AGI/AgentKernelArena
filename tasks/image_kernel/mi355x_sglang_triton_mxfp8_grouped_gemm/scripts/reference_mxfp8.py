"""Protected UE8M0 (bias 127), per-32-element MXFP8 dequantization."""
import torch


def dequant_mxfp8_to_bf16(values, scales):
    if scales.dtype != torch.uint8 or values.shape[-1] % 32:
        raise ValueError("Expected byte UE8M0 scales and complete 32-element blocks")
    expected=values.shape[:-1]+(values.shape[-1]//32,)
    if tuple(scales.shape)!=tuple(expected):
        raise ValueError("MXFP8 scale shape violates the per-1x32 contract")
    factors=torch.exp2(scales.float()-127).repeat_interleave(32,dim=-1)
    return (values.float()*factors).to(torch.bfloat16)
