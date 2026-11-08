from .onekernel_bwd_wrapper import flash_attn_onekernel_backward

CAUSAL = False


def run(*, q, k, v, o, do, lse, sm_scale, dq, dk, dv):
    flash_attn_onekernel_backward(
        do,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        None,
        sm_scale,
        None,
        CAUSAL,
        None,
        None,
        q.shape[1],
        k.shape[1],
        0.0,
    )
