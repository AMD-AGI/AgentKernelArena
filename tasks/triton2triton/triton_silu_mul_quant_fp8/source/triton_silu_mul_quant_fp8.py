import torch
import triton
import triton.language as tl


def _get_fp8_dtype():
    """Get the appropriate FP8 dtype for the current platform."""
    if hasattr(torch, 'float8_e4m3fnuz'):
        try:
            t = torch.zeros(1, device='cuda', dtype=torch.float8_e4m3fnuz)
            return torch.float8_e4m3fnuz
        except Exception:
            pass
    return torch.float8_e4m3fn


def _get_fp8_min_max():
    """Get FP8 min/max values for the current platform."""
    fp8_dtype = _get_fp8_dtype()
    if fp8_dtype == torch.float8_e4m3fnuz:
        return -240.0, 240.0
    else:
        finfo = torch.finfo(fp8_dtype)
        return finfo.min, finfo.max


@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M,
    N,
    y_s_col_stride: tl.int64,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each thread block computes [BLOCK_M, GROUP_SIZE] act-mul outputs. Then
    the thread block quantizes the [BLOCK_M, GROUP_SIZE] block of values and fills
    the outputs tensors at the right positions.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset

    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool = False,
    eps: float = 1e-10,
):
    """
    silu+mul + block-fp8 quant with group size 128.
    """
    GROUP_SIZE = 128
    assert input.ndim == 2
    if output is not None:
        assert output.ndim == 2
    assert input.size(0) % GROUP_SIZE == 0
    assert input.size(1) % (GROUP_SIZE * 2) == 0

    M, N = input.size()
    N_2 = N // 2

    fp8_dtype = _get_fp8_dtype()
    if output is None:
        output = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

    output_scales = torch.empty(
        ((N_2 // GROUP_SIZE), M), dtype=torch.float32, device=input.device
    ).transpose(0, 1)

    BLOCK_M = 8
    BLOCK_N = GROUP_SIZE
    assert M % BLOCK_M == 0
    assert N_2 % BLOCK_N == 0

    fp8_min, fp8_max = _get_fp8_min_max()

    grid = (M // BLOCK_M, N_2 // BLOCK_N)

    _silu_mul_per_token_group_quant_fp8_colmajor[grid](
        input,
        output,
        output_scales,
        M,
        N,
        output_scales.stride(-1),
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
    )

    return output, output_scales
