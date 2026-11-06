"""MXFP4 expert weights, top-k routing and the kernel-facing (AITER preshuffled) weight layout.

Logical weights (what the reference reads):
  w1 [E, 2*I, H/2] uint8   e2m1 codes, two per byte along H (low nibble first); rows 0..I-1 are the gate
                           projection, rows I..2I-1 the up projection (AITER GateMode.SEPARATED).
  w1_scale [E, 2*I, H/32]  e8m0 bytes, one per 32 consecutive H values (value 2^(byte - 127)).
  w2 [E, H, I/2] uint8     e2m1 codes packed along I.
  w2_scale [E, H, I/32]    e8m0 bytes.
Codes are uniform random bytes (every e2m1 value, both signs); scale bytes span four binades per weight. With
unit-normal tokens the gate/up projections have a standard deviation of about 0.8 (largest magnitudes about 5),
below the +-7 clamp that AITER's a16w4 stage-1 kernel applies before SiTUv2 (the model's SiTUv2 has no clamp; see
README.md), and the weighted output has a root-mean-square of about 0.4.

Routing: the top-16 of unit-normal router logits per token (distinct experts) with softmax-renormalized float32
weights, as AITER fused_topk(..., renormalize=True) produces them.

Kernel-facing layout (to_kernel_layout; weight preparation done once at load time in production), as vLLM's
Kimi-K3 SiTU loader does it (Mxfp4MoEMethod._convert_k3_situ_weight_to_kernel_format in
vllm/model_executor/layers/quantization/mxfp4.py, vllm-project/vllm) with the shuffles of ROCm/aiter (MIT) at commit
652f2d4b768239aefd515c929e84817681ff598d:
  w1, w2     aiter.ops.shuffle.shuffle_weight_a16w4(w, 16, gate_up=False): 16-row x 64-byte MFMA tiles
             (shape unchanged).
  w1_scale   aiter.ops.shuffle.shuffle_scale_a16w4(scale, E, gate_up=False) on the [E*2I, H/32] matrix.
  w2_scale   aiter.utility.fp4_utils.e8m0_shuffle on the [E*H, I/32] matrix, whose K groups are padded from 12
             to 16 (AITER leaves the pad uninitialized; here it is 0x7F): shape [E, H, 16].
"""
import torch

_W1_SCALE_BYTES = (117, 121)  # 2^-10 .. 2^-7
_W2_SCALE_BYTES = (121, 125)  # 2^-6 .. 2^-3
_UNIT_SCALE = 0x7F


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


def _shuffle_weight_a16w4(w, n_lane=16, k_pack=16):
    """aiter shuffle_weight(w, layout=(16, 16), is_guinterleave=True, gate_up=False) of one expert's [N, K/2]."""
    n, k_packed = w.shape
    k_lane = 64 // n_lane
    w = w.view(n // n_lane, n_lane, k_packed // (k_lane * k_pack), k_lane, k_pack)
    return w.permute(0, 2, 3, 1, 4).reshape(n, k_packed)


def _shuffle_scale(scale):
    """aiter shuffle_scale(src, is_guinterleave=False) (= e8m0_shuffle) of rows of a [rows, groups] e8m0 matrix.

    The permutation stays inside blocks of 32 rows, so one expert's rows (768 or 3584, multiples of 256 as
    e8m0_shuffle requires of the whole matrix) are shuffled on their own. For GateMode.SEPARATED this is
    byte-identical to shuffle_scale_a16w4.
    """
    rows, groups = scale.shape
    assert rows % 32 == 0 and groups % 8 == 0
    x = scale.view(rows // 32, 2, 16, groups // 8, 2, 4)
    return x.permute(0, 3, 5, 2, 4, 1).reshape(rows, groups)


def to_kernel_layout(case, inputs):
    """Preshuffle the (already copied) weights in place, one expert at a time to bound temporary memory."""
    inputs = dict(inputs)
    w1, w2, w1_scale, w2_scale = inputs["w1"], inputs["w2"], inputs["w1_scale"], inputs["w2_scale"]
    experts, n2, g2 = w2_scale.shape
    assert (experts * w1_scale.shape[1]) % 256 == 0 and (experts * n2) % 256 == 0
    g2_padded = (g2 + 7) // 8 * 8
    w2_scale_padded = torch.full((experts, n2, g2_padded), _UNIT_SCALE, dtype=torch.uint8, device=w2_scale.device)
    for e in range(experts):
        w1[e].copy_(_shuffle_weight_a16w4(w1[e]))
        w2[e].copy_(_shuffle_weight_a16w4(w2[e]))
        w1_scale[e].copy_(_shuffle_scale(w1_scale[e]))
        padded = torch.full((n2, g2_padded), _UNIT_SCALE, dtype=torch.uint8, device=w2_scale.device)
        padded[:, :g2] = w2_scale[e]
        w2_scale_padded[e].copy_(_shuffle_scale(padded))
    inputs["w2_scale"] = w2_scale_padded
    return inputs
