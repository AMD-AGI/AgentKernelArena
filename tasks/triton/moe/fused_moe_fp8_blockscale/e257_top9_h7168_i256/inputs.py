"""Block-quantized expert weights and DeepSeek-V3 router decisions for the fp8 block-scale MoE layer.

Weights. A DeepSeek-V3 FP8 checkpoint stores each expert matrix as float8_e4m3fn codes with one float32
scale per 128 x 128 block: scale = max|w| / 448 over the block and code = fp8(w / scale). The generator
draws normal weights (std 0.02 for the 7168-wide gate/up projection, 1/16 for the 256-wide down
projection) and quantizes them that way, a few experts at a time to bound temporary memory.

Routing. DeepSeek-V3 routes each token to 8 of 256 experts with group-limited sigmoid gating: the experts
form 8 groups of 32, the 4 groups with the highest sum of their two best bias-corrected scores are kept,
and the 8 best bias-corrected scores inside them are selected; the routing weights are the selected
sigmoid scores (without the bias) renormalized to sum to one. With the shared expert fused into the
routed experts (SGLang's fused shared experts), the shared expert is expert 256 in the last slot of every
token, weighted 1 / routed_scaling_factor (2.5) relative to the routed experts.
"""
import torch

FP8_MAX = 448.0
BLOCK = 128
W1_STD = 0.02
W2_STD = 0.0625
EXPERT_CHUNK = 8

NUM_ROUTED_EXPERTS = 256
NUM_EXPERT_GROUPS = 8
TOPK_GROUPS = 4
ROUTED_TOP_K = 8
ROUTED_SCALING_FACTOR = 2.5
CORRECTION_BIAS_STD = 0.05


def _quantized_weights(num_experts, rows, cols, std, generator, device):
    """float8_e4m3fn codes [E, rows, cols] and float32 block scales [E, rows/128, cols/128]."""
    codes = torch.empty(num_experts, rows, cols, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(num_experts, rows // BLOCK, cols // BLOCK, dtype=torch.float32, device=device)
    for start in range(0, num_experts, EXPERT_CHUNK):
        count = min(EXPERT_CHUNK, num_experts - start)
        w = torch.randn(count, rows, cols, generator=generator, device=device, dtype=torch.float32) * std
        blocks = w.view(count, rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
        scale = blocks.abs().amax(dim=(2, 4)) / FP8_MAX
        q = (blocks / scale[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
        codes[start:start + count] = q.view(count, rows, cols).to(torch.float8_e4m3fn)
        scales[start:start + count] = scale
    return codes, scales


def _deepseek_v3_routing(num_tokens, generator, device):
    logits = torch.randn(num_tokens, NUM_ROUTED_EXPERTS, generator=generator, device=device)
    bias = CORRECTION_BIAS_STD * torch.randn(NUM_ROUTED_EXPERTS, generator=generator, device=device)
    scores = logits.sigmoid()
    choice = scores + bias
    group_scores = choice.view(num_tokens, NUM_EXPERT_GROUPS, -1).topk(2, dim=-1).values.sum(dim=-1)
    group_idx = group_scores.topk(TOPK_GROUPS, dim=-1).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0).bool()
    expert_mask = group_mask.repeat_interleave(NUM_ROUTED_EXPERTS // NUM_EXPERT_GROUPS, dim=1)
    ids = choice.masked_fill(~expert_mask, float("-inf")).topk(ROUTED_TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    shared_id = torch.full((num_tokens, 1), NUM_ROUTED_EXPERTS, dtype=ids.dtype, device=device)
    shared_weight = torch.full((num_tokens, 1), 1.0 / ROUTED_SCALING_FACTOR, dtype=weights.dtype, device=device)
    ids = torch.cat([ids, shared_id], dim=-1)
    weights = torch.cat([weights, shared_weight], dim=-1)
    return ids.to(torch.int32).contiguous(), weights.to(torch.float32).contiguous()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_experts, top_k = axes["num_experts"], axes["top_k"]
    hidden, inter = axes["hidden_size"], axes["intermediate_size"]
    assert num_experts == NUM_ROUTED_EXPERTS + 1 and top_k == ROUTED_TOP_K + 1
    w1, w1_scale = _quantized_weights(num_experts, 2 * inter, hidden, W1_STD, generator, device)
    w2, w2_scale = _quantized_weights(num_experts, hidden, inter, W2_STD, generator, device)
    topk_ids, topk_weights = _deepseek_v3_routing(axes["num_tokens"], generator, device)
    return {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights, "topk_ids": topk_ids}
