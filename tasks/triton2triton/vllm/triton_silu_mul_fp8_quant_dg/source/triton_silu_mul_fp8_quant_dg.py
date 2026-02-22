"""Fused SiLU + multiply + FP8 quantization for DeepGEMM MoE, adapted from vLLM batched_deep_gemm_moe.py.

Computes: y_q = fp8_quant(silu(gate) * up) where input is [E, T, 2*H].
For AMD GPUs, uses torch.float8_e4m3fnuz; for NVIDIA, torch.float8_e4m3fn.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers
    input_ptr,   # 16-bit activations (E, T, 2*H)
    y_q_ptr,     # fp8 quantized activations (E, T, H)
    y_s_ptr,     # float32 scales (E, T, G) with strides (T*G, 1, T)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes
    H: tl.constexpr,           # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per quantization group (usually 128)
    # Strides for input
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts
    stride_counts_e,
    # Numeric params
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Fused SiLU activation + elementwise multiply + FP8 quantization.
    Input layout: [..., 2*H] where first H elements are gate, second H are up.
    Output: fp8 quantized y = silu(gate) * up, with per-group scales.

    Grid: (E * G,) where G = H // GROUP_SIZE
    Each program handles one (expert, group) pair, iterating over tokens.
    """
    G = H // GROUP_SIZE

    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def silu_mul_fp8_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU + multiply + FP8 quantization.

    Args:
        y: [E, T, 2*H] input activations (gate || up)
        tokens_per_expert: [E] number of valid tokens per expert
        group_size: quantization group size (default 128)
    Returns:
        y_q: [E, T, H] FP8 quantized output
        y_s: [E, T, G] float32 per-group scales, G = H // group_size
    """
    assert y.ndim == 3
    E, T, H2 = y.shape
    assert H2 % 2 == 0
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # Use float8_e4m3fnuz for AMD, float8_e4m3fn for NVIDIA
    try:
        fp8_dtype = torch.float8_e4m3fnuz
        _ = torch.tensor([1.0]).to(fp8_dtype)
    except (RuntimeError, AttributeError):
        fp8_dtype = torch.float8_e4m3fn

    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    # Scales: shape (E, T, G) with strides (T*G, 1, T) -- column-major in last two dims
    y_s = torch.empty_strided(
        (E, T, G),
        (T * G, 1, T),
        dtype=torch.float32,
        device=y.device,
    )

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = f_info.min
    eps = 1e-10

    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    grid = (E * G,)
    _silu_mul_fp8_quant_deep_gemm[grid](
        y, y_q, y_s, tokens_per_expert,
        H, group_size,
        stride_i_e, stride_i_t, stride_i_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        T * G, 1, T,  # ys strides
        tokens_per_expert.stride()[0],
        eps, fp8_min, fp8_max,
        ceil_ue8m0=False,
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )
    return y_q, y_s
