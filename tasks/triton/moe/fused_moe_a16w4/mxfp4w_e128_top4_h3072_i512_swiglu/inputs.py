"""MXFP4 expert weights, top-k routing and the kernel-facing weight layout.

Logical weights (what the reference reads):
  w1 [E, 2*I, H/2] uint8   e2m1 codes, two per byte along H (low nibble first); rows 0..I-1 are the gate
                           projection, rows I..2I-1 the up projection.
  w1_scale [E, 2*I, H/32]  e8m0 bytes, one per 32 consecutive H values (value 2^(byte - 127)).
  w2 [E, H, I/2] uint8     e2m1 codes packed along I.
  w2_scale [E, H, I/32]    e8m0 bytes.
Codes are uniform random bytes (every e2m1 value, both signs); scale bytes are drawn from a few binades so that
the gate/up projections of unit-normal tokens reach the SwiGLU clamp (|x| > 7) for a small fraction of elements.

Routing: GPT-OSS style, the top-4 of unit-normal router logits per token (distinct experts) with softmax-renormalized
float32 weights.

Kernel-facing layout (to_kernel_layout, weight preparation done once at load time in production):
  w1 rows are interleaved gate/up (row 2i = gate i, row 2i+1 = up i), which the fused SwiGLU epilogue of the
  triton_kernels MoE GEMM expects, and both scale tensors are permuted into the gfx950 (CDNA4) MX-scale tile order
  of AITER's shuffle_scale_moe (aiter/ops/triton/utils/shuffle.py, ROCm/aiter, MIT): w1_scale becomes
  [E, H, 2*I/32] and w2_scale [E, I, H/32].
"""
import torch

_W1_SCALE_BYTES = (119, 123)  # 2^-8 .. 2^-5
_W2_SCALE_BYTES = (117, 121)  # 2^-10 .. 2^-7


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    t, e, k = axes["n_tokens"], axes["num_experts"], axes["topk"]
    h, i = axes["hidden_size"], axes["inter_size"]
    logits = torch.randn(t, e, generator=generator, device=device, dtype=torch.float32)
    top = logits.topk(k, dim=1)
    topk_ids = top.indices.to(torch.int32)
    topk_weights = torch.softmax(top.values, dim=1)
    w1 = torch.randint(0, 256, (e, 2 * i, h // 2), generator=generator, device=device, dtype=torch.uint8)
    w1_scale = torch.randint(*_W1_SCALE_BYTES, (e, 2 * i, h // 32), generator=generator, device=device,
                             dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), generator=generator, device=device, dtype=torch.uint8)
    w2_scale = torch.randint(*_W2_SCALE_BYTES, (e, h, i // 32), generator=generator, device=device,
                             dtype=torch.uint8)
    return {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights, "topk_ids": topk_ids}


def _interleave_gate_up(x):
    """[E, 2I, ...] with gate rows then up rows -> rows gate0, up0, gate1, up1, ..."""
    e, n = x.shape[:2]
    return x.reshape(e, 2, n // 2, *x.shape[2:]).transpose(1, 2).reshape(x.shape).contiguous()


def _shuffle_scale_cdna4(scale, preshuffle_factor=32, scale_kwidth=8):
    """AITER shuffle_scale_moe (gfx950): [E, N, K/32] e8m0 -> [E, (K/32)*32, N/32]."""
    e, rows, cols = scale.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scale.reshape(e, num_stripes, 2, preshuffle_factor // 2, num_kchunks, 2, scale_kwidth // 2, 1)
    x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    tiled = x.view(e, num_stripes, cols * preshuffle_factor)
    return tiled.transpose(-1, -2).contiguous()


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    inputs["w1"] = _interleave_gate_up(inputs["w1"])
    inputs["w1_scale"] = _shuffle_scale_cdna4(_interleave_gate_up(inputs["w1_scale"]))
    inputs["w2_scale"] = _shuffle_scale_cdna4(inputs["w2_scale"])
    return inputs
