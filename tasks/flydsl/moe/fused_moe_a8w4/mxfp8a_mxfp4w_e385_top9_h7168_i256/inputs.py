# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 expert weights, Kimi-K2 router decisions and the kernel-facing (AITER preshuffled) weight layout.

Logical weights (what the reference reads), one e8m0 scale byte (2^(byte - 127)) per 32 contracted elements:
  w1 [E, 2*I, H/2] uint8   e2m1 codes, two per byte along H (low nibble = even element); rows 0..I-1 are the
                           gate projection, rows I..2I-1 the up projection.
  w1_scale [E, 2*I, H/32]  e8m0 bytes.
  w2 [E, H, I/2] uint8     e2m1 codes packed along I.
  w2_scale [E, H, I/32]    e8m0 bytes.
The generator quantizes normal weights (standard deviation 0.02 for the 7168-wide gate/up projection, 1/16 for
the 256-wide down projection) per 1 x 32 block with the source task's MXFP4 rule: e8m0 = biased exponent of
max|w| / 4, plus one when bit 22 of that float32 is set (rounding the scale up at 1.5x), and e2m1 codes of
w / 2^(e8m0 - 127) rounded to nearest even and saturated at 6.

Hidden states are unit normal bfloat16; the range-control case (t2_range) scales them to standard
deviation 512, so that the bfloat16 stage-1 activation and output exceed the float16 range.

Routing (Kimi-K2): 384 routed experts in one group, sigmoid scores, the 8 best bias-corrected scores per token,
weights = the selected sigmoid scores (without the bias) renormalized to sum to one; the shared expert is fused
as expert 384 in the last slot of every token with weight 1 / routed_scaling_factor (2.827), as SGLang's fused
shared experts do.

Kernel-facing layout (to_kernel_layout; weight preparation done once at load time in production), as the
task's AITER-derived kernels read it (ROCm/aiter, MIT, commit 652f2d4b768239aefd515c929e84817681ff598d):
  w1         aiter.ops.shuffle.shuffle_weight(w1, (16, 16)): 16-row x 32-byte tiles split in 16-byte halves.
  w1_scale   aiter.utility.fp4_utils.e8m0_shuffle on the [E*2I, H/32] matrix.
  w2         aiter.ops.shuffle.shuffle_weight_a16w4(w2, 16, False): 16-row x 64-byte tiles split in 4 x 16 bytes.
  w2_scale   aiter.ops.shuffle.shuffle_scale_a16w4(w2_scale, E, False) on the [E*H, I/32] matrix, the same
             permutation as e8m0_shuffle for these shapes.
All shapes are unchanged by the layout.
"""
import torch

BLOCK = 32
W1_STD = 0.02
W2_STD = 0.0625
EXPERT_CHUNK = 8
# Hidden-state standard deviation of the range-control case (id ending in "_range"): the stage-1
# activation and about 80 % of the output elements exceed the float16 range (65504).
RANGE_CONTROL_STD = 512.0

NUM_ROUTED_EXPERTS = 384
ROUTED_TOP_K = 8
ROUTED_SCALING_FACTOR = 2.827
CORRECTION_BIAS_STD = 0.05


def _f32_to_e2m1_codes(x):
    """Round float32 values to 4-bit e2m1 codes (sign in bit 3), nearest even, saturating at 6.

    The torchao custom_fp_utils conversion that AITER's f32_to_mxfp4 uses.
    """
    ebits, mbits = 2, 1
    ebits_f32, mbits_f32 = 8, 23
    f32_exp_bias = (1 << (ebits_f32 - 1)) - 1
    exp_bias = (1 << (ebits - 1)) - 1
    max_int = (1 << (ebits + mbits)) - 1
    sign_mask = 1 << (ebits + mbits)
    magic_adder = (1 << (mbits_f32 - mbits - 1)) - 1
    max_normal = 2 ** ((1 << ebits) - 1 - exp_bias) * (((1 << (mbits + 1)) - 1) / (2 ** mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (f32_exp_bias - exp_bias) + (mbits_f32 - mbits) + 1
    denorm_mask_int = denorm_exp << mbits_f32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32).item()

    bits = x.float().contiguous().view(torch.int32)
    sign = bits & 0x80000000
    magnitude = (bits ^ sign).view(torch.float32)
    saturate = magnitude >= max_normal
    denormal = torch.logical_and(torch.logical_not(saturate), magnitude < min_normal)
    normal = torch.logical_not(torch.logical_or(saturate, denormal))

    denormal_x = (magnitude + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)
    normal_x = magnitude.view(torch.int32)
    mant_odd = (normal_x >> (mbits_f32 - mbits)) & 1
    normal_x = normal_x + (((exp_bias - f32_exp_bias) << mbits_f32) + magic_adder) + mant_odd
    normal_x = (normal_x >> (mbits_f32 - mbits)).to(torch.uint8)

    codes = torch.full_like(magnitude, max_int, dtype=torch.uint8)
    codes = torch.where(denormal, denormal_x, codes)
    codes = torch.where(normal, normal_x, codes)
    sign_lp = (sign >> (mbits_f32 + ebits_f32 - mbits - ebits)).to(torch.uint8) & sign_mask
    return codes | sign_lp


def _mxfp4_quantize(w):
    """float32 [..., K] -> packed e2m1 codes [..., K/2] and e8m0 scale bytes [..., K/32] (uint8)."""
    shape = w.shape
    blocks = w.reshape(-1, BLOCK)
    bits = (blocks.abs().amax(dim=1) / 4.0).contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    round_up = ((bits & 0x400000) != 0) & (((bits & 0x200000) != 0) | ((bits & 0x1FFFFF) != 0) | (exponent > 0))
    scale = torch.where(exponent == 0xFF, 0xFF, exponent + round_up.to(exponent.dtype))
    inv = ((254 - scale).clamp_min(1) << 23).view(torch.float32)  # 2^(127 - e8m0), exact
    codes = _f32_to_e2m1_codes(blocks * inv.unsqueeze(1)).view(-1, BLOCK // 2, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return (packed.reshape(*shape[:-1], shape[-1] // 2),
            scale.to(torch.uint8).reshape(*shape[:-1], shape[-1] // BLOCK))


def _quantized_weights(num_experts, rows, cols, std, generator, device):
    codes = torch.empty(num_experts, rows, cols // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(num_experts, rows, cols // BLOCK, dtype=torch.uint8, device=device)
    for start in range(0, num_experts, EXPERT_CHUNK):
        count = min(EXPERT_CHUNK, num_experts - start)
        w = torch.randn(count, rows, cols, generator=generator, device=device, dtype=torch.float32) * std
        codes[start:start + count], scales[start:start + count] = _mxfp4_quantize(w)
    return codes, scales


def _kimi_k2_routing(num_tokens, generator, device):
    logits = torch.randn(num_tokens, NUM_ROUTED_EXPERTS, generator=generator, device=device)
    bias = CORRECTION_BIAS_STD * torch.randn(NUM_ROUTED_EXPERTS, generator=generator, device=device)
    scores = logits.sigmoid()
    ids = (scores + bias).topk(ROUTED_TOP_K, dim=-1).indices
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
    std = RANGE_CONTROL_STD if case["id"].endswith("_range") else 1.0
    hidden_states = (torch.randn(axes["num_tokens"], hidden, generator=generator, device=device,
                                 dtype=torch.float32) * std).to(torch.bfloat16)
    w1, w1_scale = _quantized_weights(num_experts, 2 * inter, hidden, W1_STD, generator, device)
    w2, w2_scale = _quantized_weights(num_experts, hidden, inter, W2_STD, generator, device)
    topk_ids, topk_weights = _kimi_k2_routing(axes["num_tokens"], generator, device)
    return {"hidden_states": hidden_states, "w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights, "topk_ids": topk_ids}


def _shuffle_weight_16x16(x):
    """aiter shuffle_weight(x, layout=(16, 16)) of a uint8 [E, N, K] tensor."""
    experts, n, k = x.shape
    x = x.view(experts, n // 16, 16, k // 32, 2, 16)
    return x.permute(0, 1, 3, 4, 2, 5).contiguous().view(experts, n, k)


def _shuffle_weight_a16w4(x):
    """aiter shuffle_weight(x, layout=(16, 16), is_guinterleave=True, gate_up=False) of a uint8 [E, N, K]."""
    experts, n, k = x.shape
    x = x.view(experts, n // 16, 16, k // 64, 4, 16)
    return x.permute(0, 1, 3, 4, 2, 5).contiguous().view(experts, n, k)


def _shuffle_scale(scale):
    """aiter shuffle_scale(src) (= e8m0_shuffle) of a [rows, groups] e8m0 matrix with aligned rows and groups."""
    rows, groups = scale.shape
    assert rows % 256 == 0 and groups % 8 == 0
    x = scale.view(rows // 32, 2, 16, groups // 8, 2, 4)
    return x.permute(0, 3, 5, 2, 4, 1).contiguous().view(rows, groups)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    for name in ("w1_scale", "w2_scale"):
        e, n, g = inputs[name].shape
        inputs[name] = _shuffle_scale(inputs[name].reshape(e * n, g)).view(e, n, g)
    inputs["w1"] = _shuffle_weight_16x16(inputs["w1"])
    inputs["w2"] = _shuffle_weight_a16w4(inputs["w2"])
    return inputs
