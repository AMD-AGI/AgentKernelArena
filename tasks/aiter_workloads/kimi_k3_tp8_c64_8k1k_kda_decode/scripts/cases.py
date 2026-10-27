"""Upstream-style generated operands for the fixed Kimi decode scenario."""
from __future__ import annotations

READ_ONLY = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "ssm_state_indices")


def refresh_operands(payload, seed):
    """Change values in the existing public tensors, preserving their addresses."""
    import torch

    kw = payload["kwargs"]
    gen = torch.Generator(device=kw["q"].device).manual_seed(seed)
    for name in ("q", "k", "v", "beta"):
        tensor = kw[name]
        tensor.copy_(torch.rand(tensor.shape, dtype=tensor.dtype,
                                device=tensor.device, generator=gen))
    for name in ("g", "dt_bias"):
        tensor = kw[name]
        tensor.copy_(torch.randn(tensor.shape, dtype=tensor.dtype,
                                 device=tensor.device, generator=gen))
    tensor = kw["A_log"]
    tensor.copy_((1 + 15 * torch.rand(tensor.shape, dtype=tensor.dtype,
                                    device=tensor.device, generator=gen)).log())


def make_case(case, seed=42, *, device="cuda"):
    import torch

    b, t, h, d = case["shape"]
    if t != 1:
        raise ValueError("This task has exactly one token per sequence")
    pad = case["params"]["state_padding_floats"]
    slots, plane = 3 * b, h * d * d
    stride = plane + pad
    gen = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randn(slots * stride, dtype=torch.float32, device=device,
                      generator=gen)
    state = raw.as_strided((slots, h, d, d), (stride, d * d, d, 1))
    indices = torch.randperm(slots, device=device, generator=gen)[:b].int()
    unused = torch.ones(slots, dtype=torch.bool, device=device)
    unused[indices.long()] = False
    # This is the upstream paged-state test's sentinel, not a model state.
    raw.view(slots, stride)[unused] = 1e30
    kw = {
        name: torch.empty((b, t, h, d), dtype=torch.bfloat16, device=device)
        for name in ("q", "k", "v")
    }
    kw.update(
        g=torch.empty((b, t, h, d), dtype=torch.float32, device=device),
        beta=torch.empty((b, t, h), dtype=torch.float32, device=device),
        A_log=torch.empty(h, dtype=torch.float32, device=device),
        dt_bias=torch.empty(h * d, dtype=torch.float32, device=device),
        initial_state=state, ssm_state_indices=indices,
        out=torch.empty_like(kw["v"]),
        scale=d ** -0.5, output_final_state=True, inplace_final_state=True,
        state_v_first=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
    )
    payload = {"kwargs": kw, "raw": raw, "unused": unused,
               "plane": plane, "stride": stride, "slots": slots}
    refresh_operands(payload, seed)
    payload["pristine"] = raw.clone()
    return payload


def prepare(payload):
    """Restore the same incoming state and poison output before one invocation."""
    payload["raw"].copy_(payload["pristine"])
    payload["kwargs"]["out"].fill_(float("nan"))


def tensor_identity(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
            tensor.device, tensor.storage_offset(),
            tensor.untyped_storage()._cdata, tensor.data_ptr(),
            tensor.untyped_storage().nbytes())


def snapshot(payload):
    kw = payload["kwargs"]
    names = (*READ_ONLY, "initial_state", "out")
    return {
        "identities": {name: tensor_identity(kw[name]) for name in names},
        "read_only": {name: kw[name].clone() for name in READ_ONLY},
        "raw": payload["raw"].clone(),
    }
