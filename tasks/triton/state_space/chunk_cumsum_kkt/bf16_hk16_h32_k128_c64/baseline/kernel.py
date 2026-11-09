# Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d
# (aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_cumsum_kkt.py).
# The upstream file carries no SPDX header (the ROCm/aiter LICENSE, MIT, is copied to LICENSE); the
# kernel is an AMD fusion of flash-linear-attention's chunk-local cumsum and scaled K K^T kernels,
# see README.md for the unresolved provenance.
#
# Only fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel is kept; its body is unchanged. Its
# @triton.autotune decorator (a single entry, BK 32 / 4 warps / 2 stages, unless the
# GATED_DELTA_RULE_TRITON_AUTOTUNE environment variable is set) is replaced by that entry passed at
# launch, and `exp` (tl.exp, from utils/op.py) is defined here. run() replaces
# fused_chunk_local_cumsum_scaled_dot_kkt_fwd for one sequence without cu_seqlens.

import math

import triton
import triton.language as tl

# log2(e): converts natural-log gate values to log2 space (gated_delta_rule_utils.RCP_LN2).
RCP_LN2 = math.log2(math.e)

exp = tl.exp


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel(
    g,
    k,
    beta,
    g_cumsum_out,
    A_out,
    cu_seqlens,
    sequence_ids,
    chunk_ids,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    USE_EXP2: tl.constexpr = False,
    G_SCALE: tl.constexpr = 1.0,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_flat = T
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(sequence_ids + i_t * INDEX_STRIDE).to(tl.int32),
            tl.load(chunk_ids + i_t * INDEX_STRIDE).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_cumsum = tl.cumsum(b_g, axis=0)
    # Store g_cumsum in log2 space when downstream kernels consume it with exp2:
    # exp2(x * RCP_LN2) == exp(x), keeping results identical. The scale arrives
    # as the constexpr G_SCALE (RCP_LN2 when use_exp2 else 1.0) so the kernel
    # never reads a module-level global; cumsum's linearity makes scaling before
    # or after the cumsum equivalent.
    if G_SCALE != 1.0:
        b_g_cumsum = b_g_cumsum * G_SCALE

    # g_cumsum is stored head-major [B, H, T] (stride 1 along T) so the
    # downstream solve/recompute, hidden-state and output kernels can read it
    # contiguously per (batch, head).
    if IS_VARLEN:
        g_out_base = g_cumsum_out + i_h * T_flat + bos
    else:
        g_out_base = g_cumsum_out + (i_b * H + i_h) * T_flat
    p_go = tl.make_block_ptr(g_out_base, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_go, b_g_cumsum.to(p_go.dtype.element_ty), boundary_check=(0,))

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A = tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k), acc=b_A)

    b_g_diff = b_g_cumsum[:, None] - b_g_cumsum[None, :]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_gate = tl.math.exp2(b_g_diff) if USE_EXP2 else exp(b_g_diff)
    b_A = tl.where(m_A, b_A * b_gate, 0.0)

    p_A = tl.make_block_ptr(
        A_out + (bos * H + i_h) * BT,
        (T, BT),
        (BT * H, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_A.to(A_out.dtype.element_ty), boundary_check=(0, 1))


def run(*, k, beta, g, g_cumsum, A) -> None:
    """Fused chunk-local cumsum of the gate and scaled K K^T of one sequence (no cu_seqlens).

    k: [T, Hg, K] bfloat16; beta, g: [T, H] float32 (g the raw per-token log-decay); g_cumsum:
    [H, T] float32, head-major, the chunk-local cumulative sum in log2 space (use_exp2=True);
    A: [T, H, 64] float32, strictly lower triangular.
    """
    T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = 64
    NT = triton.cdiv(T, BT)

    fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel[(NT, H)](
        g,
        k,
        beta,
        g_cumsum,
        A,
        None,
        None,
        None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=32,
        INDEX_STRIDE=1,
        USE_EXP2=True,
        G_SCALE=RCP_LN2,
        num_warps=4,
        num_stages=2,
    )
