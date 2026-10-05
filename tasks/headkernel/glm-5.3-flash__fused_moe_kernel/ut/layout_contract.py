"""Protected layout documentation and CPU codecs; not a numerical MoE oracle."""


def unpack_weight(packed):
    """Inverse of pinned AITER shuffle_weight(layout=(16,16)) for one-byte FP8."""
    if packed.element_size()!=1 or packed.ndim!=3:raise ValueError('Expected3D one-byte FP8 weights')
    e,n,k=packed.shape
    if n%16 or k%32:raise ValueError('Invalid AITER packed weight tile geometry')
    return packed.reshape(e,n//16,k//32,2,16,16).permute(0,1,4,2,3,5).contiguous().reshape(e,n,k)


def dequantize_weight(packed,scales):
    """Weight scales are plain [E,N/128,K/128] FP32, never FP4/E8M0 packing."""
    import torch
    e,n,k=packed.shape
    if list(scales.shape)!=[e,n//128,k//128] or scales.dtype!=torch.float32:raise ValueError('Wrong128x128 scale ABI')
    if not scales.is_contiguous():raise ValueError('Current block scales are row-major')
    return unpack_weight(packed).float()*scales.repeat_interleave(128,1).repeat_interleave(128,2)
