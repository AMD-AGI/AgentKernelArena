# SPDX-License-Identifier: MIT
"""Explicit generated scenario, using pinned upstream quantization/preparation.

No checkpoint tensors or historical task fixtures are consumed. Prepared weights
are generated one expert at a time to bound preparation memory. Preparation is
outside timing for both roles; candidate-specific conversions remain inside.
"""
from __future__ import annotations


def partition_mask(routed_experts=256, ep_size=4, rank=0):
    if (type(routed_experts) is not int or type(ep_size) is not int
            or routed_experts <= 0 or ep_size <= 0 or routed_experts % ep_size
            or type(rank) is not int or not 0 <= rank < ep_size):
        raise ValueError("Invalid contiguous EP partition")
    width = routed_experts // ep_size
    return [int(rank * width <= expert < (rank + 1) * width)
            for expert in range(routed_experts)] + [1]


def local_expert_map(mask):
    if not mask or any(type(value) is not int or value not in (0, 1) for value in mask):
        raise ValueError("EP mask must contain explicit binary integers")
    result, local_id = [], 0
    for present in mask:
        result.append(local_id if present else -1)
        local_id += present
    if not local_id:
        raise ValueError("EP partition has no resident experts")
    return result


def dynamic_inputs(case, seed, *, device="cuda"):
    import torch

    m, hidden, _, _, topk = case["shape"]
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn((m, hidden), dtype=torch.bfloat16,
                                generator=generator, device=device)
    scores = torch.randn((m, 256), dtype=torch.float32, generator=generator, device=device)
    probabilities = scores.softmax(dim=-1)
    weights, ids = probabilities.topk(topk - 1, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    # The shared coefficient is an explicit synthetic input-policy choice.
    # It is not an inferred SGLang shared-expert or all-reduce coefficient.
    weights = torch.cat((weights, torch.ones((m, 1), device=device)), dim=1)
    ids = torch.cat((ids, torch.full((m, 1), 256, device=device)), dim=1).to(torch.int32)
    return {"hidden_states": hidden_states, "topk_weight": weights, "topk_ids": ids}


def load_case(case):
    import torch
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    _, hidden, intermediate, experts, _ = case["shape"]
    generator = torch.Generator(device="cuda").manual_seed(case["params"]["weight_seed"])
    canonical = {}
    public = {}
    for name, rows, columns in (("w1", 2 * intermediate, hidden), ("w2", hidden, intermediate)):
        packed, scales = [], []
        for _ in range(experts):
            weight = torch.randn((rows, columns), dtype=torch.bfloat16,
                                 generator=generator, device="cuda")
            # The recorded checkpoint's public config selects Quark EVEN.
            # This recreates the format policy on generated values, not its bytes.
            quantized, scale = per_1x32_f4_quant(weight, quant_dtype=dtypes.fp4x2, round_mode=2)
            packed.append(quantized)
            scales.append(scale)
        canonical[name] = torch.stack([part.view(torch.uint8) for part in packed]).view(dtypes.fp4x2)
        canonical[name + "_scale"] = torch.cat([part.view(torch.uint8) for part in scales]).view(dtypes.fp8_e8m0)
        public[name] = shuffle_weight(canonical[name], layout=(16, 16))
        public[name + "_scale"] = e8m0_shuffle(canonical[name + "_scale"])
    mask = partition_mask(rank=case["params"]["ep_rank"])
    if sum(mask) != experts:
        raise ValueError("Local expert count disagrees with the EP mask")
    dynamic = dynamic_inputs(case, case["params"]["input_seed"])
    kwargs = {
        "expert_mask": torch.tensor(mask, dtype=torch.int32, device="cuda"),
        "activation": ActivationType.Silu, "quant_type": QuantType.per_1x32,
        "gate_mode": "separated", "doweight_stage1": False,
        "w1_scale": public["w1_scale"], "w2_scale": public["w2_scale"],
    }
    payload = {
        "args": (dynamic["hidden_states"], public["w1"], public["w2"],
                 dynamic["topk_weight"], dynamic["topk_ids"]),
        "kwargs": kwargs, "reference": canonical,
    }
    return payload


def replace_dynamic(payload, new):
    """Retain public objects/addresses for the actual timed-graph replay check."""
    for index, name in ((0, "hidden_states"), (3, "topk_weight"), (4, "topk_ids")):
        original = payload["args"][index]
        replacement = new[name]
        if (original.shape != replacement.shape or original.dtype != replacement.dtype
                or original.stride() != replacement.stride()):
            raise ValueError("Fresh dynamic state changes the public geometry")
        original.copy_(replacement)


def snapshot_inputs(payload):
    """Record all public tensors, full backing bytes and layout attributes."""
    import torch

    tensors, storages = [], {}
    for value in (*payload["args"], *payload["kwargs"].values()):
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        identity = storage._cdata
        if identity not in storages:
            nbytes = storage.nbytes()
            whole = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (nbytes,), (1,))
            storages[identity] = (storage, storage.data_ptr(), nbytes, whole.clone())
        attributes = dict(value.__dict__)
        tensors.append((value, value.shape, value.stride(), value.dtype, value.device,
                        identity, value.storage_offset(), attributes))
    return tensors, storages


def check_inputs_unchanged(snapshot):
    import torch

    tensors, storages = snapshot
    for value, shape, stride, dtype, device, identity, offset, attributes in tensors:
        if ((value.shape, value.stride(), value.dtype, value.device, value.untyped_storage()._cdata,
             value.storage_offset()) != (shape, stride, dtype, device, identity, offset)):
            raise AssertionError("Public tensor metadata changed")
        actual = dict(value.__dict__)
        if actual != attributes:
            raise AssertionError("Prepared weight attributes changed")
    for storage, pointer, nbytes, before in storages.values():
        if storage.data_ptr() != pointer or storage.nbytes() != nbytes:
            raise AssertionError("Public storage address or extent changed")
        whole = torch.empty(0, dtype=torch.uint8, device=before.device).set_(storage, 0, (nbytes,), (1,))
        if not torch.equal(whole, before):
            raise AssertionError("Public inputs are read-only, including storage slack")
