"""Float64 mHC post-mapping + next mHC pre block reference used only for correctness."""
import torch


def reference(*, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """Post: residual_cur[t, j, :] = sum_i comb_res_mix[t, i, j] * residual[t, i, :]
    + post_layer_mix[t, j] * x[t, :]. Pre on that residual: RMS-scaled projection fn @ residual_cur,
    sigmoid pre/post mixes, a softmax + Sinkhorn-normalised comb_mix, and
    layer_input = sum_j pre_mix[j] * residual_cur[j]. The pre block consumes the bfloat16
    residual_cur that the operator stores (as the unfused mhc_post then mhc_pre composition does);
    the returned residual_cur is the unrounded value."""
    tokens, hc_mult, hidden = residual.shape
    x64, res64 = x.double(), residual.double()
    post = post_layer_mix.double().reshape(tokens, hc_mult, 1)
    mixed = torch.einsum("tij,tih->tjh", comb_res_mix.double(), res64) + post * x64.unsqueeze(1)

    stream = mixed.to(torch.bfloat16).double()
    flat = stream.reshape(tokens, hc_mult * hidden)
    mixes = flat @ fn.double().t()
    mixes = mixes * torch.rsqrt(flat.square().sum(dim=-1, keepdim=True) / (hc_mult * hidden) + rms_eps)
    scale, base = hc_scale.double(), hc_base.double()

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + hc_pre_eps
    post_mix = torch.sigmoid(mixes[:, hc_mult:2 * hc_mult] * scale[1] + base[hc_mult:2 * hc_mult]) \
        * hc_post_mult_value

    logits = mixes[:, 2 * hc_mult:].reshape(tokens, hc_mult, hc_mult) * scale[2] \
        + base[2 * hc_mult:].reshape(1, hc_mult, hc_mult)
    comb = torch.softmax(logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(int(sinkhorn_repeat) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre_mix.unsqueeze(-1) * stream).sum(dim=1)
    return {"residual_cur": mixed, "post_mix_cur": post_mix.reshape(tokens, hc_mult, 1),
            "comb_mix_cur": comb, "layer_input_cur": layer_input}
