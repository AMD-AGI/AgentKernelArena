"""Per-token modulation index and the previous block's per-token gate for the gated residual task."""
import torch

# Image tokens of the edited image come first (index 0); each reference image appended after it
# (index 1) has the same token count, and a request carries one to three of them.
MIN_IMAGES, MAX_IMAGES = 2, 4
GATE_STD = 0.5


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, seq_len, hidden = axes["batch"], axes["seq_len"], axes["hidden_size"]
    n_images = torch.randint(MIN_IMAGES, MAX_IMAGES + 1, (batch,), generator=generator, device=device)
    first_len = torch.clamp(-(-seq_len // n_images), min=1, max=max(seq_len - 1, 1))
    positions = torch.arange(seq_len, device=device)
    index = (positions[None, :] >= first_len[:, None]).to(torch.int32)
    # The gate that scales the attention output is the previous norm's selected gate: one row of
    # two per sample, repeated over the tokens that share the index.
    gates = (torch.randn((2, batch, hidden), generator=generator, device=device) * GATE_STD).to(torch.bfloat16)
    second = index.bool().unsqueeze(-1)
    residual_gate = torch.where(second, gates[1].unsqueeze(1), gates[0].unsqueeze(1)).contiguous()
    return {"index": index, "residual_gate": residual_gate}
