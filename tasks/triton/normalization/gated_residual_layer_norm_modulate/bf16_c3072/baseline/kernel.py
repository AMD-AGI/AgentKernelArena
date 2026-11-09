"""Fused gated residual add + LayerNorm + adaLN modulation with per-token two-way selection, from SGLang.

Kernel: sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py::
_fused_residual_layernorm_scale_shift_gate_select01_kernel (copied unchanged below).
Launcher: fuse_residual_layernorm_scale_shift_gate_select01_kernel, specialized to a
non-affine LayerNorm (weight and bias absent) and inlined into run().

The upstream file carries no SPDX or copyright header; the Apache-2.0 LICENSE of the SGLang
repository at the recorded revision applies.
"""
import triton
import triton.language as tl


@triton.jit
def _fused_residual_layernorm_scale_shift_gate_select01_kernel(
    output_ptr,
    residual_out_ptr,
    gate_out_ptr,
    x_ptr,
    residual_ptr,
    residual_gate_ptr,
    weight_ptr,
    bias_ptr,
    scale0_ptr,
    shift0_ptr,
    gate0_ptr,
    scale1_ptr,
    shift1_ptr,
    gate1_ptr,
    index_ptr,
    inner_dim,
    seq_len,
    stride_x_row,
    stride_res_row,
    stride_rg_row,
    stride_out_row,
    stride_res_out_row,
    stride_go_row,
    stride_w,
    stride_b,
    stride_s0_b,
    stride_s0_c,
    stride_sh0_b,
    stride_sh0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sh1_b,
    stride_sh1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < inner_dim

    x_row_ptr = x_ptr + row * stride_x_row
    res_row_ptr = residual_ptr + row * stride_res_row
    rg_row_ptr = residual_gate_ptr + row * stride_rg_row
    out_row_ptr = output_ptr + row * stride_out_row
    res_out_row_ptr = residual_out_ptr + row * stride_res_out_row
    gate_row_ptr = gate_out_ptr + row * stride_go_row

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual_gate = tl.load(rg_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual_out = residual + residual_gate * x
    tl.store(res_out_row_ptr + cols, residual_out, mask=mask)

    mean = tl.sum(residual_out, axis=0) / inner_dim
    xbar = tl.where(mask, residual_out - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / inner_dim
    rstd = tl.rsqrt(var + eps)
    x_hat = (residual_out - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols * stride_w, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + cols * stride_b, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + b

    batch_idx = row // seq_len
    seq_idx = row % seq_len
    idx = tl.load(index_ptr + batch_idx * stride_i_b + seq_idx * stride_i_l).to(tl.int1)

    scale0_ptrs = scale0_ptr + batch_idx * stride_s0_b + cols * stride_s0_c
    shift0_ptrs = shift0_ptr + batch_idx * stride_sh0_b + cols * stride_sh0_c
    gate0_ptrs = gate0_ptr + batch_idx * stride_g0_b + cols * stride_g0_c

    scale1_ptrs = scale1_ptr + batch_idx * stride_s1_b + cols * stride_s1_c
    shift1_ptrs = shift1_ptr + batch_idx * stride_sh1_b + cols * stride_sh1_c
    gate1_ptrs = gate1_ptr + batch_idx * stride_g1_b + cols * stride_g1_c

    # Branch on scalar idx instead of using tl.where on pointers.
    # tl.where on pointers triggers an assertion in AMD Triton's
    # CanonicalizePointers pass (ConvertArithSelectOp) on gfx950.
    # This keeps it at 3 loads (not 6), avoids the pointer-level
    # tl.where entirely, and since idx is uniform across all threads
    # the branch has no divergence cost.
    if idx:
        scale = tl.load(scale1_ptrs, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift1_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate1_ptrs, mask=mask, other=0.0)
    else:
        scale = tl.load(scale0_ptrs, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift0_ptrs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate0_ptrs, mask=mask, other=0.0)
    y = x_hat * (1.0 + scale) + shift

    tl.store(out_row_ptr + cols, y, mask=mask)
    tl.store(gate_row_ptr + cols, gate, mask=mask)


def run(*, x, residual, residual_gate, scale0, shift0, gate0, scale1, shift1, gate1, index, epsilon,
        out, residual_out, gate_out):
    B, L, C = x.shape
    x_2d = x.view(B * L, C)
    residual_2d = residual.view(B * L, C)
    residual_gate_2d = residual_gate.view(B * L, C)
    output_2d = out.view(B * L, C)
    residual_out_2d = residual_out.view(B * L, C)
    gate_out_2d = gate_out.view(B * L, C)

    MAX_FUSED_SIZE = 65536 // x_2d.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    if C > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps, num_stages = 4, 4

    grid = (B * L,)
    # The adaLN norms of the served model are non-affine: x_2d stands in for the unused weight and bias.
    _fused_residual_layernorm_scale_shift_gate_select01_kernel[grid](
        output_2d,
        residual_out_2d,
        gate_out_2d,
        x_2d,
        residual_2d,
        residual_gate_2d,
        x_2d,
        x_2d,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        C,
        L,
        x_2d.stride(0),
        residual_2d.stride(0),
        residual_gate_2d.stride(0),
        output_2d.stride(0),
        residual_out_2d.stride(0),
        gate_out_2d.stride(0),
        0,
        0,
        scale0.stride(0),
        scale0.stride(1),
        shift0.stride(0),
        shift0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        epsilon,
        HAS_WEIGHT=False,
        HAS_BIAS=False,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
