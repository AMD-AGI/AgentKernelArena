"""Float64 reference: fp8 group-quantized batched projection of q_nope, interleaved RoPE and the MLA cache write.

For every token and head the 128 values of q_nope are quantized as the operator does: scale = max(max |x|, 1e-10) *
(1 / 448) in float32, codes = fp8(clamp(x * (1 / scale), -448, 448)) with round to nearest even. The product of the
codes with the fp8 weight codes is accumulated exactly (float64), multiplied by the token scale and the per-tensor
weight scale. RoPE is applied to q_pe and k_pe in float64 with the cos/sin rows of the token's position.
"""
import torch

_FP8_MAX = 448.0


def _rope_interleaved(x, cos, sin):
    """x: [T, H, D]; cos, sin: [T, D // 2] (float64). Pairs (x[2i], x[2i+1]) rotate together."""
    cos = cos.repeat_interleave(2, dim=-1)[:, None, :]
    sin = sin.repeat_interleave(2, dim=-1)[:, None, :]
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def reference(*, q_nope, w_k, w_k_scale, q_pe, k_nope, k_pe, positions, slot_mapping, cos, sin, kv_cache):
    heads, n_tokens, _ = q_nope.shape
    lora = w_k.shape[1]
    one_over_max = torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32, device=q_nope.device)
    scale_w = w_k_scale.double().reshape(())
    q_out = torch.empty((n_tokens, heads, lora + q_pe.shape[-1]), dtype=torch.float64, device=q_nope.device)
    for h in range(heads):
        x = q_nope[h].float()
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        scale = amax * one_over_max
        codes = (x * (1.0 / scale)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        q_out[:, h, :lora] = (codes.double() @ w_k[h].double().T) * scale.double() * scale_w
    cos_t = cos.double()[positions]
    sin_t = sin.double()[positions]
    q_out[..., lora:] = _rope_interleaved(q_pe.double(), cos_t, sin_t)
    k_pe_out = _rope_interleaved(k_pe.double(), cos_t, sin_t)
    cache = kv_cache.double()
    cache[slot_mapping] = torch.cat((k_nope.double(), k_pe_out), dim=-1)
    return {"q_out": q_out, "k_pe_out": k_pe_out, "kv_cache": cache}
