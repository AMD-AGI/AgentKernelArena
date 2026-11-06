"""Exact MXFP4 quantization with MoE-sorted E8M0 scales, used only for correctness.

The quantization is defined on float32 bit patterns (bfloat16 inputs are exact in float32), so it is computed with
integer arithmetic on the bits of the float32 values; there is no higher-precision variant of a code or a scale.
"""
import torch

GROUP = 32
ROW_CHUNK = 1 << 17


def _quantize(rows):
    """rows [R, K] bfloat16 -> (packed codes [R, K // 2] uint8, scale codes [R, K // 32] uint8)."""
    r, k = rows.shape
    x = rows.float().reshape(r, k // GROUP, GROUP)
    amax = x.abs().amax(dim=-1)
    bits = (amax.view(torch.int32) + 0x200000) & 0xFF800000          # round the exponent to nearest
    amax = bits.view(torch.float32)
    unbiased = (torch.log2(amax).floor() - 2).clamp(-127, 127)
    scale_code = (unbiased.to(torch.int32) + 127).to(torch.uint8)    # E8M0: biased exponent, 0..254
    qx = x * torch.exp2(-unbiased).unsqueeze(-1)
    qb = qx.view(torch.int32)
    sign = qb & 0x80000000
    exp = (qb >> 23) & 0xFF
    man = qb & 0x7FFFFF
    shift = (127 - (exp + 1)).clamp(min=0, max=31)
    man = torch.where(exp < 127, (0x400000 | (man >> 1)) >> shift, man)
    exp = torch.clamp(exp, min=126) - 126
    value = torch.clamp((((exp << 2) | (man >> 21)) + 1) >> 1, max=7)
    code = (((sign >> 28) & 0xF) | value).to(torch.uint8)             # sign bit + 3 magnitude bits
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.reshape(r, k // 2), scale_code


def reference(*, x, sorted_ids, num_valid_ids, token_num):
    rows, k = x.shape
    top_k = rows // token_num
    fp4 = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    scale_rows = torch.empty(rows, k // GROUP, dtype=torch.uint8, device=x.device)
    for start in range(0, rows, ROW_CHUNK):
        fp4[start:start + ROW_CHUNK], scale_rows[start:start + ROW_CHUNK] = _quantize(x[start:start + ROW_CHUNK])
    # Gather the scale rows in sorted order; sentinel rows (token >= token_num) hold the scale code of an all-zero group.
    ids = sorted_ids.to(torch.int64)
    token, slot = ids & 0xFFFFFF, ids >> 24
    valid = token < token_num
    source = torch.where(valid, token * top_k + slot, torch.zeros_like(token))
    sorted_rows = torch.where(valid[:, None], scale_rows[source], torch.zeros((), dtype=torch.uint8, device=x.device))
    # Tile layout: [R, Cb, n, m, k] <- sorted row 32 R + 16 a + m, group 8 Cb + 4 c + n, k = 2 c + a.
    length, groups = sorted_rows.shape
    tiles = sorted_rows.reshape(length // 32, 2, 16, groups // 8, 2, 4)      # R, a, m, Cb, c, n
    tiles = tiles.permute(0, 3, 5, 2, 4, 1)                                    # R, Cb, n, m, c, a
    return fp4, tiles.reshape(length, groups).contiguous()
