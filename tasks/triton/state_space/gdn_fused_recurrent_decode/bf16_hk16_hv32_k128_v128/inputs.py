"""State-pool slots of the GDN packed decode task."""
import torch


def make_inputs(case, inputs, generator, device):
    batch, slots = case["axes"]["batch"], case["axes"]["pool_slots"]
    # Every sequence owns a distinct slot; the pool also holds slots no sequence touches.
    indices = torch.randperm(slots, generator=generator, device=device)[:batch].to(torch.int32)
    # Padded sequences (as in a CUDA-graph-padded decode batch) carry slot -1.
    indices[15::16] = -1
    return {"ssm_state_indices": indices}
