"""MXFP4 expert weights, GLM-5-style routing and the kernel-facing weight layout.

Logical weights (what the reference reads), per expert e:
  w1 [E, 2*I, H/2] uint8    e2m1 codes, two per byte along H (low nibble = even element); rows 0..I-1 are
                            the gate projection, rows I..2I-1 the up projection.
  w1_scale [E, 2*I, H/32]   e8m0 bytes, one per 32 consecutive H values (value 2^(byte - 127)).
  w2 [E, H, I/2] uint8      e2m1 codes packed along I.
  w2_scale [E, H, I/32]     e8m0 bytes.
The weights are drawn N(0, 1/fan_in) in bfloat16 and quantized offline as in the source task: the block scale
is the upward-rounded power of two 2^ceil(log2(amax / 6)) and the scaled values round to the nearest e2m1 value,
ties to even.

Routing: 256 routed experts plus one fused shared expert (id 256). Each token takes the top-8 routed experts of
sigmoid router scores, weighted by those scores renormalized to sum to one, and the shared expert in its ninth
slot with weight 1 / routed_scaling_factor (2.5), the fused-shared-expert convention of SGLang's
biased_grouped_topk.

Kernel-facing layout (to_kernel_layout; weight preparation done once at load time in production):
  w1 rows are interleaved gate/up (row 2i = gate i, row 2i+1 = up i), which the fused SiLU-and-mul epilogue of
  the AITER/triton_kernels MoE GEMM expects, and both scale tensors are permuted into the gfx950 (CDNA4)
  MX-scale tile order of AITER's shuffle_scale_moe (aiter/ops/triton/utils/shuffle.py, ROCm/aiter, MIT), stored
  as the [E, N/32, (K/32)*32] tile tensor whose transposed view the kernel reads: w1_scale becomes
  [E, 2*I/32, H] and w2_scale [E, H/32, I].
"""
import math

import torch

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROUTED_SCALING_FACTOR = 2.5
_EXPERT_CHUNK = 4


def _quantize_mxfp4(values):
    """float32 [..., K] -> (packed e2m1 codes [..., K/2], e8m0 bytes [..., K/32]), upward-rounded scales."""
    blocks = values.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax)
    # 2^exponent is the smallest power of two with amax / 2^exponent <= 6 (6 = 0.75 * 2^3).
    exponent = exponent - 3 + (mantissa > 0.75).to(torch.int32)
    exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
    scaled = blocks * torch.exp2(-exponent.to(torch.float32))
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=values.device)
    midpoints = (grid[1:] + grid[:-1]) / 2
    magnitude = scaled.abs()
    code = torch.bucketize(magnitude, midpoints)
    # Ties go to the even code (round to nearest even).
    tie = (code < 7) & (magnitude == midpoints[code.clamp(max=6)]) & (code % 2 == 1)
    code = code + tie.to(code.dtype)
    code = (code | ((scaled < 0) & (code > 0)).to(code.dtype) << 3).to(torch.uint8).flatten(-2)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, (exponent + 127).to(torch.uint8).squeeze(-1)


def _quantized_weights(experts, rows, cols, generator, device):
    codes = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((experts, rows, cols // 32), dtype=torch.uint8, device=device)
    std = 1.0 / math.sqrt(cols)
    for start in range(0, experts, _EXPERT_CHUNK):
        stop = min(start + _EXPERT_CHUNK, experts)
        plain = torch.randn((stop - start, rows, cols), generator=generator, device=device, dtype=torch.float32)
        plain = (plain * std).to(torch.bfloat16).to(torch.float32)
        codes[start:stop], scales[start:stop] = _quantize_mxfp4(plain)
    return codes, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    t, e, k = axes["n_tokens"], axes["num_experts"], axes["topk"]
    h, i = axes["hidden_size"], axes["inter_size"]
    routed = e - 1
    scores = torch.sigmoid(torch.randn((t, routed), generator=generator, device=device, dtype=torch.float32))
    top = scores.topk(k - 1, dim=1)
    topk_ids = torch.full((t, k), routed, dtype=torch.int32, device=device)
    topk_ids[:, : k - 1] = top.indices.to(torch.int32)
    topk_weights = torch.full((t, k), 1.0 / _ROUTED_SCALING_FACTOR, dtype=torch.float32, device=device)
    topk_weights[:, : k - 1] = top.values / top.values.sum(dim=1, keepdim=True)
    w1, w1_scale = _quantized_weights(e, 2 * i, h, generator, device)
    w2, w2_scale = _quantized_weights(e, h, i, generator, device)
    return {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights, "topk_ids": topk_ids}


def _interleave_gate_up(x):
    """[E, 2I, ...] with gate rows then up rows -> rows gate0, up0, gate1, up1, ..."""
    e, n = x.shape[:2]
    return x.reshape(e, 2, n // 2, *x.shape[2:]).transpose(1, 2).reshape(x.shape).contiguous()


def _shuffle_scale_cdna4(scale, preshuffle_factor=32, scale_kwidth=8):
    """AITER _shuffle_scale_tile_gfx950 of an [E, N, K/32] e8m0 tensor -> [E, N/32, (K/32)*32] tiles."""
    e, rows, cols = scale.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scale.reshape(e, num_stripes, 2, preshuffle_factor // 2, num_kchunks, 2, scale_kwidth // 2, 1)
    x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    return x.view(e, num_stripes, cols * preshuffle_factor)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    inputs["w1"] = _interleave_gate_up(inputs["w1"])
    inputs["w1_scale"] = _shuffle_scale_cdna4(_interleave_gate_up(inputs["w1_scale"]))
    inputs["w2_scale"] = _shuffle_scale_cdna4(inputs["w2_scale"])
    return inputs
