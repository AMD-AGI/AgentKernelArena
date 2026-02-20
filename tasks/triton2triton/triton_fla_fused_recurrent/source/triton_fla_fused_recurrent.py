"""Triton kernel: fused_recurrent_gated_delta_rule_fwd_kernel — fused recurrent gated delta rule."""
import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, g, beta, o, h0, ht,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = i_nh // H
    i_h = i_nh % H

    bos = i_n * T
    eos = bos + T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_beta = beta + bos * H + i_h
    p_g = g + bos * H + i_h
    p_o = o + ((i_k * B * T + bos) * H + i_h) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_q = b_q * scale
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta_val = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Store final state for every timestep
        p_ht = ht + (bos + i_t) * H * V * K + i_h * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += H * V
        p_v += H * V
        p_g += H
        p_beta += H


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
) -> tuple:
    """Fused recurrent gated delta rule forward.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] — scalar gate (log decay)
        beta: [B, T, H] — scalar beta
        scale: float
        initial_state: [B, H, V, K] or None
    Returns:
        o: [B, T, H, V], final_state: [B*T, H, V, K]
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = B
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1

    o = q.new_empty(NK, B, T, H, V)
    final_state = q.new_empty(B * T, H, V, K, dtype=torch.float32)

    grid = (NK, NV, N * H)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state,
        scale=scale, N=N, T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV,
        num_warps=1, num_stages=3,
    )
    o = o.squeeze(0)
    return o, final_state
