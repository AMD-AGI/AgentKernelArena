"""Inputs of the fused FP8 BMM + RoPE + MLA cache write: fp8 weights, RoPE tables and unique cache slots."""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude


def make_inputs(case, inputs, generator, device):
    """fp8 e4m3fn absorbed weights with one per-tensor scale, interleaved-RoPE tables (base 10000) and slots.

    No slot is negative: the baseline leaves k_pe_out unwritten for a negative-slot token, so padding would make
    that output undefined.
    """
    axes = case["axes"]
    heads, lora, nope = axes["num_q_heads"], axes["kv_lora_dim"], axes["qk_nope_head_dim"]
    n_tokens, max_position = axes["n_tokens"], axes["max_position"]
    rope_dim, num_slots = axes["rope_dim"], axes["num_slots"]
    w = 0.05 * torch.randn(heads, lora, nope, generator=generator, device=device, dtype=torch.float32)
    # One scale for the whole batched tensor (amax over every head), as the dynamic per-batched-tensor
    # quantization of the serving frameworks computes it; the operator multiplies by this scale (amax / 448).
    amax = w.abs().amax().clamp_min(1e-10)
    w_k = (w * (_FP8_MAX / amax)).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    inv_freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    slots = torch.randperm(num_slots, generator=generator, device=device)[:n_tokens]
    return {"w_k": w_k, "w_k_scale": (amax / _FP8_MAX).reshape(1),
            "cos": torch.cos(angles).to(torch.bfloat16), "sin": torch.sin(angles).to(torch.bfloat16),
            "slot_mapping": slots.to(torch.int64)}
