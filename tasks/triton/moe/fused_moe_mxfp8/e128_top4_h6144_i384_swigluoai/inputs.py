"""MXFP8 expert weights and router decisions of the MiniMax-M3 MoE layer."""
import torch

_E4M3_MAX = 448.0
_EXPERT_CHUNK = 8


def quantize_mxfp8(x):
    """Per-1x32-block E8M0 scale (rounded up) and float8_e4m3fn values along the last axis, float32 math."""
    shape = x.shape
    blocked = x.to(torch.float32).view(*shape[:-1], shape[-1] // 32, 32)
    amax = blocked.abs().amax(dim=-1).clamp(min=1e-30)
    biased = (torch.ceil(torch.log2(amax / _E4M3_MAX)) + 127.0).clamp(0.0, 254.0)
    scaled = (blocked / torch.exp2(biased - 127.0).unsqueeze(-1)).clamp(-_E4M3_MAX, _E4M3_MAX)
    return scaled.view(shape).to(torch.float8_e4m3fn), biased.to(torch.uint8)


def _quantized_weights(shape, generator, device, std=0.1):
    values = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty((*shape[:-1], shape[-1] // 32), dtype=torch.uint8, device=device)
    for start in range(0, shape[0], _EXPERT_CHUNK):
        stop = min(start + _EXPERT_CHUNK, shape[0])
        drawn = torch.randn((stop - start, *shape[1:]), generator=generator, device=device,
                            dtype=torch.float32) * std
        values[start:stop], scales[start:stop] = quantize_mxfp8(drawn.to(torch.bfloat16))
    return values, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_tokens, top_k, num_experts = axes["num_tokens"], axes["top_k"], axes["num_experts"]
    hidden, inter = axes["hidden_size"], axes["intermediate_size"]
    w13, w13_scale = _quantized_weights((num_experts, 2 * inter, hidden), generator, device)
    w2, w2_scale = _quantized_weights((num_experts, hidden, inter), generator, device)
    # Each token selects top_k distinct experts from random normal gate logits; its routing weights are the
    # softmax over the selected logits, so they sum to one.
    logits = torch.randn(num_tokens, num_experts, generator=generator, device=device, dtype=torch.float32)
    values, ids = logits.topk(top_k, dim=-1)
    return {"w13": w13, "w13_scale": w13_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_ids": ids.to(torch.int32), "topk_weights": torch.softmax(values, dim=-1)}
