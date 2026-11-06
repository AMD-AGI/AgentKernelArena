# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC post-mapping followed by the next layer's mHC pre block, TileLang kernels from vLLM.

Kernels and dispatch are those of vllm/model_executor/kernels/mhc/{tilelang_kernels,tilelang}.py
for a ROCm GPU without DeepGEMM (no split-K from the TF32 GEMM, no PDL), without fused output
RMSNorm, with the default n_splits/tile_n of mhc_fused_post_pre_tilelang.

ROCm adaptation: the three kernels that reduce a dot product across the thread block
(mhc_fused_tilelang, hc_prenorm_gemm_tilelang, hc_prenorm_gemm_block_m_tilelang) reduce through
shared memory instead of T.warp_reduce_sum plus a 32-lane warp split; see README.md.
"""
import math

import torch
import tilelang
import tilelang.language as T

ENABLE_PDL = False  # Programmatic dependent launch is a CUDA (sm90+) feature.
RED_GROUP = 32  # Partial sums combined per thread in the first stage of the block reductions.

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA"""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()

        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = s_comb[k, j]

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

        # Block reduction through shared memory. Upstream reduces with T.warp_reduce_sum and assumes
        # 32-lane warps (lane = tid % 32); an AMD wavefront has 64 lanes, where that scheme counts
        # each wave twice and scales the projection by sqrt(2). This reduction does not depend on
        # the hardware wave size.
        part = T.alloc_shared((n_thr, tile_n + 1), T.float32)
        grp = T.alloc_shared((n_thr // RED_GROUP, tile_n + 1), T.float32)
        gsum = T.alloc_local((tile_n + 1,), T.float32)
        fin = T.alloc_local((1,), T.float32)
        for n in T.unroll(tile_n):
            part[tid, n] = acc[n]
        part[tid, tile_n] = sqr[0]
        T.sync_threads()
        if tid < n_thr // RED_GROUP:
            for c in T.unroll(tile_n + 1):
                gsum[c] = 0.0
                for r in T.serial(RED_GROUP):
                    gsum[c] += part[tid * RED_GROUP + r, c]
                grp[tid, c] = gsum[c]
        T.sync_threads()
        if tid < tile_n + 1:
            fin[0] = 0.0
            for w in T.serial(n_thr // RED_GROUP):
                fin[0] += grp[w, tid]
            if tid < tile_n:
                yp_out[i_ks, i_n, i_nt * tile_n + tid] = fin[0]
            if tid == tile_n and i_nt == 0:
                rp_out[i_ks, i_n] = fin[0]

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_post_tilelang(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])
        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def hc_prenorm_gemm_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hidden_size: int,
    hc_mult: int = 4,
    n_out: int = 24,
    n_thr: int = 512,
    tile_n: int = 12,
    n_splits: int = 1,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    hc_hidden_size = hc_mult * hidden_size
    k_per_split = hc_hidden_size // n_splits
    k_iters = k_per_split // n_thr
    n_tiles = T.ceildiv(n_out, tile_n)

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)  # type: ignore[no-redef, valid-type]
    fn: T.Tensor((n_out, hc_hidden_size), T.float32)  # type: ignore[no-redef, valid-type]
    out: T.Tensor((n_splits, num_tokens, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((n_splits, num_tokens), T.float32)  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr) as (
        i_n,
        i_t,
        i_s,
    ):
        tid = T.get_thread_binding()
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        T.clear(acc)
        T.clear(sqr)

        if ENABLE_PDL:
            T.pdl_sync()

        for it in T.serial(k_iters):
            i_k = i_s * k_per_split + it * n_thr + tid
            x_val = x[i_n, i_k]
            for i_o in T.unroll(tile_n):
                out_idx = i_t * tile_n + i_o
                if out_idx < n_out:
                    acc[i_o] += x_val * fn[out_idx, i_k]
            if i_t == 0:
                sqr[0] += x_val * x_val

        # Block reduction through shared memory (independent of the hardware wave size, see
        # mhc_fused_tilelang).
        part = T.alloc_shared((n_thr, tile_n + 1), T.float32)
        grp = T.alloc_shared((n_thr // RED_GROUP, tile_n + 1), T.float32)
        gsum = T.alloc_local((tile_n + 1,), T.float32)
        fin = T.alloc_local((1,), T.float32)
        for i_o in T.unroll(tile_n):
            part[tid, i_o] = acc[i_o]
        part[tid, tile_n] = sqr[0]
        T.sync_threads()
        if tid < n_thr // RED_GROUP:
            for c in T.unroll(tile_n + 1):
                gsum[c] = 0.0
                for r in T.serial(RED_GROUP):
                    gsum[c] += part[tid * RED_GROUP + r, c]
                grp[tid, c] = gsum[c]
        T.sync_threads()
        if tid < tile_n + 1:
            fin[0] = 0.0
            for w in T.serial(n_thr // RED_GROUP):
                fin[0] += grp[w, tid]
            if tid < tile_n:
                if i_t * tile_n + tid < n_out:
                    out[i_s, i_n, i_t * tile_n + tid] = fin[0]
            if tid == tile_n and i_t == 0:
                sqrsum[i_s, i_n] = fin[0]

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def hc_prenorm_gemm_block_m_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hidden_size: int,
    hc_mult: int = 4,
    n_out: int = 24,
    n_thr: int = 512,
    tile_n: int = 12,
    block_m: int = 2,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    hc_hidden_size = hc_mult * hidden_size
    k_iters = hc_hidden_size // n_thr
    n_tiles = T.ceildiv(n_out, tile_n)
    m_tiles = T.ceildiv(num_tokens, block_m)

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)  # type: ignore[no-redef, valid-type]
    fn: T.Tensor((n_out, hc_hidden_size), T.float32)  # type: ignore[no-redef, valid-type]
    out: T.Tensor((1, num_tokens, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((1, num_tokens), T.float32)  # type: ignore[no-redef, valid-type]

    with T.Kernel(m_tiles, n_tiles, threads=n_thr) as (i_mt, i_t):
        tid = T.get_thread_binding()
        acc = T.alloc_local((block_m, tile_n), T.float32)
        sqr = T.alloc_local((block_m,), T.float32)
        T.clear(acc)
        T.clear(sqr)

        if ENABLE_PDL:
            T.pdl_sync()

        for it in T.serial(k_iters):
            i_k = it * n_thr + tid
            fn_val = T.alloc_local((tile_n,), T.float32)
            for i_o in T.unroll(tile_n):
                out_idx = i_t * tile_n + i_o
                if out_idx < n_out:
                    fn_val[i_o] = fn[out_idx, i_k]
                else:
                    fn_val[i_o] = 0.0
            for i_m in T.unroll(block_m):
                token_idx = i_mt * block_m + i_m
                if token_idx < num_tokens:
                    x_val = x[token_idx, i_k]
                    for i_o in T.unroll(tile_n):
                        acc[i_m, i_o] += x_val * fn_val[i_o]
                    if i_t == 0:
                        sqr[i_m] += x_val * x_val

        # Block reduction through shared memory (independent of the hardware wave size, see
        # mhc_fused_tilelang). Columns: block_m * tile_n outputs, then block_m square sums.
        n_acc = block_m * tile_n
        n_cols = n_acc + block_m
        part = T.alloc_shared((n_thr, n_cols), T.float32)
        grp = T.alloc_shared((n_thr // RED_GROUP, n_cols), T.float32)
        gsum = T.alloc_local((n_cols,), T.float32)
        fin = T.alloc_local((1,), T.float32)
        for i_m in T.unroll(block_m):
            for i_o in T.unroll(tile_n):
                part[tid, i_m * tile_n + i_o] = acc[i_m, i_o]
            part[tid, n_acc + i_m] = sqr[i_m]
        T.sync_threads()
        if tid < n_thr // RED_GROUP:
            for c in T.unroll(n_cols):
                gsum[c] = 0.0
                for r in T.serial(RED_GROUP):
                    gsum[c] += part[tid * RED_GROUP + r, c]
                grp[tid, c] = gsum[c]
        T.sync_threads()
        if tid < n_cols:
            fin[0] = 0.0
            for w in T.serial(n_thr // RED_GROUP):
                fin[0] += grp[w, tid]
            if tid < n_acc:
                token_idx = i_mt * block_m + tid // tile_n
                out_idx = i_t * tile_n + tid % tile_n
                if token_idx < num_tokens:
                    if out_idx < n_out:
                        out[0, token_idx, out_idx] = fin[0]
            if tid >= n_acc:
                token_idx2 = i_mt * block_m + (tid - n_acc)
                if token_idx2 < num_tokens:
                    if i_t == 0:
                        sqrsum[0, token_idx2] = fin[0]

        if ENABLE_PDL:
            T.pdl_trigger()


def _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult, tile_n=12, n_thr=512, n_splits=1):
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, 2)
        return
    if n_splits == 1 and use_default_config and x.shape[0] < 128 and x.shape[1] % 1024 == 0:
        hc_prenorm_gemm_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], 1024, 4, n_splits)
        return
    hc_prenorm_gemm_tilelang(x, fn, out, sqrsum, hidden_size, hc_mult, fn.shape[0], n_thr, tile_n, n_splits)


def run(*, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, residual_cur, post_mix_cur, comb_mix_cur,
        layer_input_cur) -> None:
    """One mHC post block followed by the next mHC pre block (mhc_fused_post_pre_tilelang)."""
    num_tokens, hc_mult, hidden_size = residual.shape
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    rms_eps, hc_pre_eps = float(rms_eps), float(hc_pre_eps)
    hc_sinkhorn_eps, hc_post_mult_value = float(hc_sinkhorn_eps), float(hc_post_mult_value)
    sinkhorn_repeat = int(sinkhorn_repeat)

    post_flat = post_layer_mix.view(num_tokens, hc_mult)
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        n_splits = 1

    gemm_out_mul = torch.empty(n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=residual.device)

    if use_small_fma:
        mhc_fused_tilelang(comb_res_mix, residual, post_flat, x, fn.view(hc_mult3, hc_mult, hidden_size),
                           gemm_out_mul, gemm_out_sqrsum, residual_cur, hc_mult, hidden_size, hc_mult3,
                           tile_n=tile_n, split_k=n_splits)
    else:
        mhc_post_tilelang(comb_res_mix, residual, post_flat, x, residual_cur, hc_mult, hidden_size)
        _tilelang_hc_prenorm_gemm(residual_cur.view(num_tokens, hc_hidden_size), fn, gemm_out_mul,
                                  gemm_out_sqrsum, hidden_size, hc_mult)

    mhc_pre_big_fuse_tilelang(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual_cur,
                              post_mix_cur.view(num_tokens, hc_mult), comb_mix_cur.view(num_tokens, hc_mult2),
                              layer_input_cur, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                              hc_post_mult_value, sinkhorn_repeat, n_splits, hc_mult)
