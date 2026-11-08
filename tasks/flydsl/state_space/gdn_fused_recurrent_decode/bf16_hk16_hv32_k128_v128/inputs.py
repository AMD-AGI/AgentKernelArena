"""State-pool slots of the GDN decode task."""
import torch


def make_inputs(case, inputs, generator, device):
    batch, slots = case["axes"]["batch"], case["axes"]["pool_slots"]
    # Every sequence owns a distinct slot; the pool also holds slots no sequence touches.
    indices = torch.randperm(slots, generator=generator, device=device)[:batch].to(torch.int32)
    return {"state_indices": indices}
