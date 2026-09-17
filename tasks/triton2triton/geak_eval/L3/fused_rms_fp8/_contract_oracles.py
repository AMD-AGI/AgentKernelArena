"""Protected PyTorch oracles for the declared FP8 wrapper variants."""
import torch
import torch.nn.functional as F


def quantize(x, dtype, group=128):
    x = x.float()
    rows, width = x.shape
    padded = F.pad(x, (0, (-width) % group))
    grouped = padded.reshape(rows, -1, group)
    maximum = grouped.abs().amax(dim=-1).clamp_min(1e-10)
    scales = maximum / torch.finfo(dtype).max
    q = (grouped / scales.unsqueeze(-1)).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max)
    return q.to(dtype).reshape(rows, -1)[:, :width], scales


def norm(x, weight, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def rms(saved, dtype, *, reduce=False, show=True):
    x = saved['x1'].float()
    if reduce and x.ndim == 3:
        x = x.sum(0)
    residual = saved.get('res1')
    if residual is not None:
        x = x + residual.float()
    y1 = norm(x, saved['w1'])
    y2 = None
    if 'x2' in saved:
        second = saved['x2'].float()
        if reduce and second.ndim == 3:
            second = second.sum(0)
        y2 = norm(second, saved['w2']).to(saved['x1'].dtype)
    result = (quantize(y1, dtype), y1.to(saved['x1'].dtype) if show else None,
              y2, x.to(saved['x1'].dtype) if residual is not None else None)
    if reduce:
        third = saved['x3'].float().sum(0).to(saved['x1'].dtype) if 'x3' in saved else None
        result += (third,)
    return result


def activation_mul(saved, dtype):
    x = saved['x'].float()
    if x.ndim == 3:
        x = x.sum(0)
    left, right = x.chunk(2, dim=-1)
    y = F.silu(left) * right
    aux = saved['x2'].float().sum(0).to(saved['x'].dtype) if 'x2' in saved else None
    return quantize(y, dtype), aux


def check_quant(actual, expected, *, atol=0.1, rtol=0.1):
    q, scale = actual
    ref_q, ref_scale = expected
    assert (scale > 0).all(), 'FP8 group scales must be positive'
    # A scale has its own meaning. Compensating wrong q and scale cannot pass
    # merely because their product happens to reconstruct the right values.
    torch.testing.assert_close(scale, ref_scale, atol=0, rtol=rtol)
    assert torch.isfinite(q.float()).all() and torch.isfinite(ref_q.float()).all()
    def ordered_codes(value):
        bits = value.contiguous().view(torch.uint8).to(torch.int16)
        return torch.where(bits < 128, 128 + bits, 128 - (bits & 127))
    # FP32 rsqrt/reduction/reciprocal rounding at an FP8 midpoint can choose
    # either adjacent code (one step can be 12.5%, greater than a 10% float
    # tolerance). More than one representable FP8 step is always rejected.
    assert ((ordered_codes(q) - ordered_codes(ref_q)).abs() <= 1).all(), 'FP8 output differs by more than one representable step'
    width = q.shape[1]
    expanded = scale.repeat_interleave(128, dim=-1)[:, :width]
    ref_expanded = ref_scale.repeat_interleave(128, dim=-1)[:, :width]
    torch.testing.assert_close(q.float() * expanded, ref_q.float() * ref_expanded,
                               atol=atol, rtol=rtol)


def check_fused(actual, expected, *, atol=0.1, rtol=0.1):
    check_quant(actual[0], expected[0], atol=atol, rtol=rtol)
    for output, reference in zip(actual[1:], expected[1:]):
        if reference is not None:
            torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
