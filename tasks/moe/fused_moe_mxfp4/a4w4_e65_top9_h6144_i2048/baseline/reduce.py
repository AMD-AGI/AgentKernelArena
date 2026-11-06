# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's grouped row reduction, used here to add the top_k weighted expert outputs of a token.

clip, _swiglu: aiter/ops/triton/_triton_kernels/moe/activations.py (the _swiglu parameter `input`
    renamed to `x`; the reduction below never enables it)
_reduce_grouped: aiter/ops/triton/_triton_kernels/moe/reduce.py (unchanged)
reduce_grouped: aiter/ops/triton/moe/reduce.py, the Triton (non-gfx1250) path only, called with
indx=None so it sums x[b, m, :] over b in float32 (its split-k mode).
"""
import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(x, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.
    """
    gelu, linear = tl.split(tl.reshape(x, (x.shape[0], x.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear


@triton.jit
def _reduce_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,
    Out,
    stride_om: tl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    M,
    N,
    num_blocks,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    USE_TDM: tl.constexpr,
    # Step 9: external residual fold-in. When HAS_EXT_RESIDUAL=True,
    # Residual[token, :] is added to `acc` before the writeback.
    Residual,
    stride_extres_m: tl.uint64,
    stride_extres_n,
    HAS_EXT_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_t = pid // num_blocks
    pid_n = pid % num_blocks

    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i),)
    XPtrs = X + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
    x_n_mask = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) < N
    if USE_TDM and EVEN_N:
        x_desc = tl.make_tensor_descriptor(
            base=X, shape=(B * M, N), strides=(N, 1), block_shape=(1, BLOCK_N)
        )
    # accumulate contributions for this tile
    for i in tl.static_range(0, K):
        curr = tl.zeros([BLOCK_N], dtype=tl.float32)
        # iterate over split_k partial values
        for b in tl.range(0, B):
            if USE_TDM and EVEN_N:
                row = b * M + indxs[i]
                vals = tl.reshape(x_desc.load([row, pid_n * BLOCK_N]), (BLOCK_N,))
            else:
                x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                if EVEN_N:
                    vals = tl.load(x_row_ptr)
                else:
                    vals = tl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(tl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(
                curr[None, :], alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL
            )
        curr = tl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # Step 9: optional external residual fold-in: load residual at this
    # tile and add to acc before writeback. Same per-token-row layout as Out.
    if HAS_EXT_RESIDUAL:
        res_offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        res_ptr = Residual + pid_t * stride_extres_m + res_offs_n * stride_extres_n
        if EVEN_N:
            res = tl.load(res_ptr).to(tl.float32)
            acc = acc + res
        else:
            res_mask = res_offs_n < Nrem
            res = tl.load(res_ptr, mask=res_mask, other=0.0).to(tl.float32)
            acc = acc + res
    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        tl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT) < Nrem
        tl.store(out_ptr, acc, mask=out_n_mask)


def reduce_grouped(
    x,
    indx,
    out,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype=None,
    swiglu_add_residual: bool = True,
    residual=None,
):
    """
    Grouped row reduction used during moe scatter and also compatible with split-k reduce.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]`. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Returns
    - The output tensor `out`.
    """

    if indx is None and x.shape[0] == 1:
        assert residual is None, (
            "reduce_grouped early-return path can't apply external residual; "
            "either rebuild routing with K>=1 or skip residual fold for this call"
        )
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0

    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)

    # Step 9: prep external residual buffer + strides for the kernel.
    if residual is not None:
        assert (
            residual.shape == out.shape
        ), f"residual.shape {tuple(residual.shape)} must match out.shape {tuple(out.shape)}"
        res_stride_m = residual.stride(0)
        res_stride_n = residual.stride(1)
        has_ext_residual = True
    else:
        res_stride_m = 0
        res_stride_n = 0
        has_ext_residual = False
    _reduce_grouped[(num_blocks * num_groups,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        num_blocks,
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,
        SWIGLU_ADD_RESIDUAL=swiglu_add_residual,
        USE_TDM=False,  # is_tdm_avail(): gfx1250 only
        Residual=residual,
        stride_extres_m=res_stride_m,
        stride_extres_n=res_stride_n,
        HAS_EXT_RESIDUAL=has_ext_residual,
        num_warps=2,
    )
    return out
