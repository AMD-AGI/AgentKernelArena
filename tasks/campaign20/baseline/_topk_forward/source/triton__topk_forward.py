import triton
import triton.language as tl

@triton.jit
def _topk_forward(X, stride_xm,  # inputs
                  PeerYvs, PeerYis, stride_ym,  # topk values/indices
                  USE_PROVIDED_INDX: tl.constexpr, PeerBits, stride_rm: tl.constexpr,
                  stride_rn: tl.constexpr,  # bitmatrix
                  n_rows, n_expts_tot,  # shape
                  dst_offs_m, APPLY_SOFTMAX: tl.constexpr,  # constant
                  BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):

    N_PEERS: tl.constexpr = len(PeerYvs)

    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid * BLOCK_M >= n_rows:
        # early exit:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    if USE_PROVIDED_INDX:
        tl.static_assert(len(PeerYis) == 1)
        Yi_ptrs = PeerYis[0] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        y_values, y_indices = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m,  #
                                             N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)

    # write back
    for rank in tl.static_range(N_PEERS):
        Yv_ptrs = PeerYvs[rank] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
        tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        for rank in tl.static_range(N_PEERS):
            Yi_ptrs = PeerYis[rank] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
            tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        for rank in tl.static_range(N_PEERS):
            BitsPtrs = PeerBits[rank] + (dst_offs_m + offs_m[:, None]) * stride_rm + offs_r_n[None, :] * stride_rn
            tl.store(BitsPtrs, r, mask=mask_m)

