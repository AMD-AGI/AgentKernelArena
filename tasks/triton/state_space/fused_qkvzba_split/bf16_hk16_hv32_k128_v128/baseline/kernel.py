# Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d
# (aiter/ops/triton/_triton_kernels/gated_delta_rule/fused_qkvzba_split.py).
# The upstream file carries no SPDX header (the ROCm/aiter LICENSE, MIT, is copied to LICENSE); its
# kernels match SGLang's fused_qkvzba_split_reshape_cat_kernel, see README.md for the unresolved provenance.
#
# The five prefill kernels are kept unchanged; the decode kernel and its wrapper are removed.
# run() is the dispatch of fused_qkvzba_split_reshape_cat_prefill, writing preallocated outputs.
"""Fused kernels for QKVZBA split, reshape and concatenation."""

import triton
import triton.language as tl

# Query/key heads and their size of this scenario (Qwen3-Next, tensor-parallel 1); the value heads and
# their size come from the shape of z. The upstream function takes all four as arguments.
NUM_HEADS_QK = 16
HEAD_QK = 128


@triton.jit
def _fused_qkvzba_prefill_kernel_nqk1(
    mixed_qkv,
    z,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    """Prefill kernel specialized for NUM_HEADS_QK=1."""
    i_seq = tl.program_id(0)
    V_PER_QK: tl.constexpr = NUM_HEADS_V
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V * HEAD_V

    base_in = mixed_qkvz + i_seq * QKVZ_DIM_T
    base_out = mixed_qkv + i_seq * QKV_DIM_T

    # Load and store q
    blk_q = tl.load(base_in + tl.arange(0, HEAD_QK))
    tl.store(base_out + tl.arange(0, HEAD_QK), blk_q)

    # Load and store k
    blk_k = tl.load(base_in + HEAD_QK + tl.arange(0, HEAD_QK))
    tl.store(base_out + HEAD_QK + tl.arange(0, HEAD_QK), blk_k)

    # Load and store v
    blk_v = tl.load(base_in + HEAD_QK * 2 + tl.arange(0, V_PER_QK * HEAD_V))
    tl.store(base_out + HEAD_QK * 2 + tl.arange(0, V_PER_QK * HEAD_V), blk_v)

    # Load and store z
    blk_z = tl.load(
        base_in + HEAD_QK * 2 + V_PER_QK * HEAD_V + tl.arange(0, V_PER_QK * HEAD_V)
    )
    tl.store(z + i_seq * NUM_HEADS_V * HEAD_V + tl.arange(0, V_PER_QK * HEAD_V), blk_z)

    # Load and store b, a
    base_ba = mixed_ba + i_seq * BA_DIM_T
    for i in tl.static_range(V_PER_QK):
        tl.store(b_out + i_seq * NUM_HEADS_V + i, tl.load(base_ba + i))
        tl.store(a_out + i_seq * NUM_HEADS_V + i, tl.load(base_ba + V_PER_QK + i))


@triton.jit
def _fused_qkvzba_prefill_kernel_nqk2(
    mixed_qkv,
    z,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    """Prefill kernel specialized for NUM_HEADS_QK=2, single block processes entire row."""
    i_seq = tl.program_id(0)
    NUM_HEADS_QK: tl.constexpr = 2
    V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = NUM_HEADS_QK * HEAD_QK * 2 + NUM_HEADS_V * HEAD_V

    base_out = mixed_qkv + i_seq * QKV_DIM_T

    # Process all heads sequentially to ensure contiguous writes
    for i_qk in tl.static_range(NUM_HEADS_QK):
        base_in = mixed_qkvz + i_seq * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T

        # Load q, k, v, z
        blk_q = tl.load(base_in + tl.arange(0, HEAD_QK))
        blk_k = tl.load(base_in + HEAD_QK + tl.arange(0, HEAD_QK))
        blk_v = tl.load(base_in + HEAD_QK * 2 + tl.arange(0, V_PER_QK * HEAD_V))
        blk_z = tl.load(
            base_in + HEAD_QK * 2 + V_PER_QK * HEAD_V + tl.arange(0, V_PER_QK * HEAD_V)
        )

        # Store q (all q contiguous)
        tl.store(base_out + i_qk * HEAD_QK + tl.arange(0, HEAD_QK), blk_q)
        # Store k (all k contiguous, after q)
        tl.store(
            base_out + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK + tl.arange(0, HEAD_QK),
            blk_k,
        )
        # Store v (all v contiguous, after k)
        tl.store(
            base_out
            + NUM_HEADS_QK * HEAD_QK * 2
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_v,
        )
        # Store z
        tl.store(
            z
            + i_seq * NUM_HEADS_V * HEAD_V
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_z,
        )

        # Store b, a
        base_ba = mixed_ba + i_seq * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
        for i in tl.static_range(V_PER_QK):
            tl.store(
                b_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i, tl.load(base_ba + i)
            )
            tl.store(
                a_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i,
                tl.load(base_ba + V_PER_QK + i),
            )


@triton.jit
def _fused_qkvzba_prefill_kernel_nqk4(
    mixed_qkv,
    z,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    """Prefill kernel specialized for NUM_HEADS_QK=4, single block processes entire row."""
    i_seq = tl.program_id(0)
    NUM_HEADS_QK: tl.constexpr = 4
    V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = NUM_HEADS_QK * HEAD_QK * 2 + NUM_HEADS_V * HEAD_V

    base_out = mixed_qkv + i_seq * QKV_DIM_T

    for i_qk in tl.static_range(NUM_HEADS_QK):
        base_in = mixed_qkvz + i_seq * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T

        blk_q = tl.load(base_in + tl.arange(0, HEAD_QK))
        blk_k = tl.load(base_in + HEAD_QK + tl.arange(0, HEAD_QK))
        blk_v = tl.load(base_in + HEAD_QK * 2 + tl.arange(0, V_PER_QK * HEAD_V))
        blk_z = tl.load(
            base_in + HEAD_QK * 2 + V_PER_QK * HEAD_V + tl.arange(0, V_PER_QK * HEAD_V)
        )

        tl.store(base_out + i_qk * HEAD_QK + tl.arange(0, HEAD_QK), blk_q)
        tl.store(
            base_out + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK + tl.arange(0, HEAD_QK),
            blk_k,
        )
        tl.store(
            base_out
            + NUM_HEADS_QK * HEAD_QK * 2
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_v,
        )
        tl.store(
            z
            + i_seq * NUM_HEADS_V * HEAD_V
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_z,
        )

        base_ba = mixed_ba + i_seq * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
        for i in tl.static_range(V_PER_QK):
            tl.store(
                b_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i, tl.load(base_ba + i)
            )
            tl.store(
                a_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i,
                tl.load(base_ba + V_PER_QK + i),
            )


@triton.jit
def _fused_qkvzba_prefill_kernel_nqk8(
    mixed_qkv,
    z,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    """Prefill kernel specialized for NUM_HEADS_QK=8, single block processes entire row."""
    i_seq = tl.program_id(0)
    NUM_HEADS_QK: tl.constexpr = 8
    V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = NUM_HEADS_QK * HEAD_QK * 2 + NUM_HEADS_V * HEAD_V

    base_out = mixed_qkv + i_seq * QKV_DIM_T

    for i_qk in tl.static_range(NUM_HEADS_QK):
        base_in = mixed_qkvz + i_seq * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T

        blk_q = tl.load(base_in + tl.arange(0, HEAD_QK))
        blk_k = tl.load(base_in + HEAD_QK + tl.arange(0, HEAD_QK))
        blk_v = tl.load(base_in + HEAD_QK * 2 + tl.arange(0, V_PER_QK * HEAD_V))
        blk_z = tl.load(
            base_in + HEAD_QK * 2 + V_PER_QK * HEAD_V + tl.arange(0, V_PER_QK * HEAD_V)
        )

        tl.store(base_out + i_qk * HEAD_QK + tl.arange(0, HEAD_QK), blk_q)
        tl.store(
            base_out + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK + tl.arange(0, HEAD_QK),
            blk_k,
        )
        tl.store(
            base_out
            + NUM_HEADS_QK * HEAD_QK * 2
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_v,
        )
        tl.store(
            z
            + i_seq * NUM_HEADS_V * HEAD_V
            + i_qk * V_PER_QK * HEAD_V
            + tl.arange(0, V_PER_QK * HEAD_V),
            blk_z,
        )

        base_ba = mixed_ba + i_seq * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T
        for i in tl.static_range(V_PER_QK):
            tl.store(
                b_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i, tl.load(base_ba + i)
            )
            tl.store(
                a_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i,
                tl.load(base_ba + V_PER_QK + i),
            )


@triton.jit
def _fused_qkvzba_split_reshape_cat_prefill_kernel(
    mixed_qkv,
    z,
    b_out,
    a_out,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    """
    Generic prefill kernel (fallback for unsupported NUM_HEADS_QK values).

    Uses 2D grid where each program processes one (seq_pos, qk_head).
    Note: This may cause cache line conflicts; prefer specialized kernels.
    """
    i_seq, i_qk = tl.program_id(0), tl.program_id(1)

    V_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = V_PER_QK * 2
    QKV_DIM_T: tl.constexpr = NUM_HEADS_QK * HEAD_QK * 2 + NUM_HEADS_V * HEAD_V

    # Load from mixed_qkvz
    base_qkvz = mixed_qkvz + i_seq * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T

    q_end: tl.constexpr = HEAD_QK
    blk_q = tl.load(base_qkvz + tl.arange(0, q_end))

    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k = tl.load(base_qkvz + tl.arange(q_end, k_end))

    v_end: tl.constexpr = k_end + V_PER_QK * HEAD_V
    blk_v = tl.load(base_qkvz + tl.arange(k_end, v_end))

    z_end: tl.constexpr = v_end + V_PER_QK * HEAD_V
    blk_z = tl.load(base_qkvz + tl.arange(v_end, z_end))

    # Load from mixed_ba
    base_ba = mixed_ba + i_seq * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T

    # Store to mixed_qkv (concatenated q, k, v)
    q_out_ptr = mixed_qkv + i_seq * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    tl.store(q_out_ptr, blk_q)

    k_out_ptr = (
        mixed_qkv
        + i_seq * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    tl.store(k_out_ptr, blk_k)

    v_out_ptr = (
        mixed_qkv
        + i_seq * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * V_PER_QK * HEAD_V
        + tl.arange(0, V_PER_QK * HEAD_V)
    )
    tl.store(v_out_ptr, blk_v)

    # Store z
    z_out_ptr = (
        z
        + i_seq * NUM_HEADS_V * HEAD_V
        + i_qk * V_PER_QK * HEAD_V
        + tl.arange(0, V_PER_QK * HEAD_V)
    )
    tl.store(z_out_ptr, blk_z)

    # Store b, a
    for i in tl.static_range(V_PER_QK):
        blk_b_ptr = base_ba + i
        blk_b_st_ptr = b_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

        blk_a_ptr = base_ba + V_PER_QK + i
        blk_a_st_ptr = a_out + i_seq * NUM_HEADS_V + i_qk * V_PER_QK + i
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def run(*, mixed_qkvz, mixed_ba, mixed_qkv, z, b, a) -> None:
    """Prefill stage fused split: mixed_qkvz [T, nk * (2 hk + 2 r hv)] and mixed_ba [T, nk * 2 r]
    into mixed_qkv [T, 2 nk hk + nv hv] (q | k | v), z [T, nv, hv], b and a [T, nv]."""
    seq_len = mixed_qkvz.shape[0]
    num_heads_v = z.shape[1]
    head_v = z.shape[2]
    num_heads_qk = NUM_HEADS_QK
    head_qk = HEAD_QK

    # Select specialized kernel based on num_heads_qk
    if num_heads_qk == 1:
        grid = (seq_len,)
        _fused_qkvzba_prefill_kernel_nqk1[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
    elif num_heads_qk == 2:
        grid = (seq_len,)
        _fused_qkvzba_prefill_kernel_nqk2[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
    elif num_heads_qk == 4:
        grid = (seq_len,)
        _fused_qkvzba_prefill_kernel_nqk4[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
    elif num_heads_qk == 8:
        grid = (seq_len,)
        _fused_qkvzba_prefill_kernel_nqk8[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
    else:
        # Fallback to generic 2D-grid kernel
        grid = (seq_len, num_heads_qk)
        _fused_qkvzba_split_reshape_cat_prefill_kernel[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
