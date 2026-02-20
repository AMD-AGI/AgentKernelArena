import torch
import triton
import triton.language as tl


_is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

if _is_rocm:
    @triton.jit
    def round_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)
else:
    @triton.jit
    def round_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = round_int8(x_q)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, original_shape[-1])
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    x = x.contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    x_q = x_q.view(*original_shape)
    scales = scales.view(*original_shape[:-1], 1)
    return x_q, scales
