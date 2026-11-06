# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""L2 normalization over the last dimension (forward), from SGLang's FLA ``l2norm_fwd_kernel``."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def run(*, x, epsilon, out):
    T, D = x.shape
    # The upstream launcher's D <= 512 path: 16 rows per program, one feature block per row.
    BD = triton.next_power_of_2(D)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    l2norm_fwd_kernel[grid](
        x,
        out,
        epsilon,
        T=T,
        D=D,
        BD=BD,
        BT=16,
        num_warps=8,
        num_stages=3,
    )
