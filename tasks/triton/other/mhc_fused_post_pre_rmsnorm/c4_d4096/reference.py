"""Float64 reference of the fused mHC post + next pre block with output RMSNorm, for correctness only."""
import torch


def reference(*, x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias, norm_weight,
              rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters, norm_eps):
    """Post, then the next pre on the bfloat16 residual the operator stores, then RMSNorm.

    Post: next_residual[t, j, :] = post_mix[t, j] * x[t, :] + sum_i comb_mix[t, i, j] * residual[t, i, :]
    (comb_mix is [source, destination]). The returned next_residual is the unrounded value; the
    operator stores it in bfloat16 and the pre block reads that stored residual.

    Pre: an RMS-scaled projection of the flattened bfloat16 residual by proj_weight, then per group
    (pre | post | comb) a mix_scale factor and the mix_bias. The pre gates are sigmoid + pre_eps;
    next_post_mix = sigmoid(post logits) * post_multiplier; next_comb_mix = softmax over the last
    axis plus sinkhorn_eps, a column normalization, then sinkhorn_iters - 1 row/column
    normalizations (every denominator plus sinkhorn_eps), returned row-major as [tokens, 16].

    Output norm: layer_input = RMSNorm(sum_i pre_gate_i * residual_i) * norm_weight with norm_eps,
    the gated sum taken over the stored bfloat16 residual and kept in high precision; only the
    normalized result is rounded (by the comparison) to bfloat16.
    """
    tokens, hc_mult, hidden = residual.shape
    mixed = torch.bmm(comb_mix.double().transpose(1, 2), residual.double())
    mixed.addcmul_(post_mix.double().unsqueeze(-1), x.double().unsqueeze(1))

    stream = mixed.to(torch.bfloat16).double()
    flat = stream.reshape(tokens, hc_mult * hidden)
    mean_square = torch.bmm(flat.unsqueeze(1), flat.unsqueeze(2)).reshape(tokens, 1) / flat.shape[1]
    mixes = (flat @ proj_weight.double().t()) * torch.rsqrt(mean_square + rms_eps)
    scale, base = mix_scale.double(), mix_bias.double()

    pre_gate = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + pre_eps
    next_post = torch.sigmoid(mixes[:, hc_mult:2 * hc_mult] * scale[1] + base[hc_mult:2 * hc_mult]) \
        * post_multiplier

    logits = (mixes[:, 2 * hc_mult:] * scale[2] + base[2 * hc_mult:]).reshape(tokens, hc_mult, hc_mult)
    comb = torch.softmax(logits, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(int(sinkhorn_iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)

    folded = torch.bmm(pre_gate.unsqueeze(1), stream).reshape(tokens, hidden)
    del stream, flat
    layer_input = folded * torch.rsqrt(folded.square().mean(dim=-1, keepdim=True) + norm_eps) \
        * norm_weight.double()
    return {"next_post_mix": next_post, "next_comb_mix": comb.reshape(tokens, hc_mult * hc_mult),
            "layer_input": layer_input, "next_residual": mixed}
