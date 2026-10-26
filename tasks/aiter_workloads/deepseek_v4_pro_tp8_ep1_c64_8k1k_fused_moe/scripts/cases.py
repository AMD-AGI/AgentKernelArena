"""Load protected fresh-case tensors; no synthetic workload is inferred here.

Each case['inputs'] holds path/sha256 for a tensor-only torch.save mapping,
plus explicit call_kwargs and tensor_attributes for the public w1/w2.
Required mapping keys:
hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale, and
canonical_w1, canonical_w2, canonical_w1_scale, canonical_w2_scale.
Only optional bias1/bias2 tensors are currently supported. The canonical four
are never exposed to the candidate. A fresh binding must qualify this carrier
against actual caller inputs; historical kernel fixtures are not admitted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from reference import validate_contract

_PUBLIC = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids", "w1_scale", "w2_scale")
_CANONICAL = ("w1", "w2", "w1_scale", "w2_scale")


def load_case(root, case, contract, *, device="cuda"):
    validate_contract(contract)
    descriptor = case.get("inputs")
    if not isinstance(descriptor, dict):
        raise ValueError("Case has no fresh, protected input binding")
    # This first adapter handles one explicitly described public-call family.
    # A caller with masks, shared experts, scatter, prequantized inputs, output
    # destinations or extra kwargs requires an extended reference/input binding.
    scalar_kwargs = {
        "activation": "silu", "quant_type": "per_1x32", "gate_mode": "interleave",
        "doweight_stage1": contract["doweight_stage1"], "swiglu_limit": contract["swiglu_limit"],
        "hidden_pad": 0, "intermediate_pad": 0,
    }
    if descriptor.get("call_kwargs") != scalar_kwargs:
        raise ValueError("Case caller kwargs are unresolved or outside this reference binding")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Case input path must be relative to the task")
    root = Path(root).resolve()
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Case input path escapes the task or is not a file")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Case input binding requires a lowercase SHA256")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Case input digest differs from protected binding")
        stream.seek(0)
        import torch
        tensors = torch.load(stream, map_location=device, weights_only=True)
    required = set(_PUBLIC) | {"canonical_" + name for name in _CANONICAL}
    if not isinstance(tensors, dict) or not required <= tensors.keys() or tensors.keys() - required - {"bias1", "bias2"}:
        raise ValueError("Input payload has missing or unsupported tensors")
    if any(not isinstance(value, torch.Tensor) or value.layout != torch.strided for value in tensors.values()):
        raise ValueError("Input payload must contain only dense tensors")
    public_names = set(_PUBLIC) | ({"bias1", "bias2"} & tensors.keys())
    declared = case.get("params", {}).get("tensors")
    if not isinstance(declared, dict) or set(declared) != public_names:
        raise ValueError("Manifest must declare every public tensor shape/dtype/stride")
    for name in public_names:
        tensor = tensors[name]
        actual = {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}
        if declared[name] != actual:
            raise ValueError(f"Actual tensor differs from protected manifest: {name}")
    attributes = descriptor.get("tensor_attributes")
    if not isinstance(attributes, dict) or set(attributes) != {"w1", "w2"}:
        raise ValueError("Exact public weight layout attributes must be bound")
    for name, gate_up in (("w1", True), ("w2", False)):
        expected_attrs = {"is_shuffled": True, "gate_up_interleave": gate_up}
        if attributes[name] != expected_attrs or any(type(v) is not bool for v in attributes[name].values()):
            raise ValueError(f"Unsupported or unresolved layout attributes for {name}")
        for key, value in attributes[name].items():
            setattr(tensors[name], key, value)

    # Verify that oracle and public inputs encode the very same weights.
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    canonical = {name: tensors["canonical_" + name] for name in _CANONICAL}
    experts = canonical["w1"].shape[0]
    for name in ("w1", "w2"):
        if canonical[name].ndim != 3 or canonical[name].dtype != dtypes.fp4x2:
            raise ValueError("Canonical weights must be rank3 packed FP4 tensors")
        if tensors[name].dtype != dtypes.fp4x2 or tensors[name + "_scale"].dtype != dtypes.fp8_e8m0:
            raise ValueError("Public weights/scales differ from the bound FP4/E8M0 formats")
        if canonical[name + "_scale"].dtype != dtypes.fp8_e8m0:
            raise ValueError("Canonical scales must use E8M0")
    for name, gate_up in (("w1", True), ("w2", False)):
        packed = shuffle_weight_a16w4(canonical[name], 16, gate_up)
        scales = shuffle_scale_a16w4(canonical[name + "_scale"], experts, gate_up)
        for field, rebuilt in ((name, packed), (name + "_scale", scales)):
            public = tensors[field]
            if public.shape != rebuilt.shape or public.dtype != rebuilt.dtype or not torch.equal(
                public.contiguous().view(torch.uint8), rebuilt.contiguous().view(torch.uint8)
            ):
                raise ValueError(f"Public/canonical weight binding disagrees: {field}")
    hidden, ids, weights = (tensors[name] for name in ("hidden_states", "topk_ids", "topk_weight"))
    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or ids.ndim != 2 or ids.dtype != torch.int32:
        raise ValueError("Unsupported public hidden-state or routing layout")
    if weights.shape != ids.shape or weights.dtype != torch.float32 or ids.shape[0] != hidden.shape[0]:
        raise ValueError("Routing tensor geometry/dtype disagrees")
    if not torch.isfinite(hidden).all() or not torch.isfinite(weights).all() or (ids < 0).any() or (ids >= experts).any():
        raise ValueError("Nonfinite inputs or invalid local expert IDs")
    inter_dim = canonical["w2"].shape[-1] * 2
    if tuple(canonical["w1"].shape) != (experts, 2 * inter_dim, hidden.shape[1] // 2) or tuple(canonical["w2"].shape[:2]) != (experts, hidden.shape[1]):
        raise ValueError("Canonical expert weight geometry disagrees with hidden states")
    actual_shape = [hidden.shape[0], hidden.shape[1], inter_dim, experts, ids.shape[1]]
    if case.get("shape") != actual_shape or case.get("dtype") != "bfloat16":
        raise ValueError("Actual operation geometry/dtype differs from the manifest")
    kwargs = {
        "activation": ActivationType.Silu, "quant_type": QuantType.per_1x32,
        "gate_mode": GateMode.INTERLEAVE.value, "doweight_stage1": contract["doweight_stage1"],
        "swiglu_limit": contract["swiglu_limit"], "hidden_pad": 0, "intermediate_pad": 0,
        "w1_scale": tensors["w1_scale"], "w2_scale": tensors["w2_scale"],
    }
    kwargs.update({name: tensors[name] for name in ("bias1", "bias2") if name in tensors})
    return {"args": tuple(tensors[name] for name in _PUBLIC[:5]), "kwargs": kwargs, "reference": canonical}


def snapshot_inputs(payload):
    """One snapshot per case/action, outside every measured interval."""
    import torch
    values = list(payload["args"]) + list(payload["kwargs"].values())
    seen, snapshots = set(), []
    for value in values:
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            snapshots.append((value, value.shape, value.stride(), value.dtype,
                              value.detach().clone(memory_format=torch.preserve_format)))
    return snapshots


def check_inputs_unchanged(snapshots):
    import torch
    for tensor, shape, stride, dtype, before in snapshots:
        if tensor.shape != shape or tensor.stride() != stride or tensor.dtype != dtype:
            raise AssertionError("Public input metadata changed")
        if not torch.equal(tensor.contiguous().view(torch.uint8), before.contiguous().view(torch.uint8)):
            raise AssertionError("The complete-call input contract is read-only; an input changed")
