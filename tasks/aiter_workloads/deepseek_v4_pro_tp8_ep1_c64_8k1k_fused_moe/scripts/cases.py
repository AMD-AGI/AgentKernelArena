"""Load protected fresh-case tensors; no synthetic workload is inferred here.

Each case['inputs'] holds path/sha256 for hidden_states/topk_weight/topk_ids,
a separate weights path/sha256 descriptor, and explicit call_kwargs and
tensor_attributes for the public w1/w2. The weights file holds w1/w2, their
scales, canonical unshuffled copies and optional bias1/bias2 tensors. Cases can
share one immutable weight file per rank/layer without duplicating its bytes.
inputs.current_value_replay binds a second activation/routing file whose full
backing buffers can be copied into the original public storage after timing.
The canonical four are never exposed to the candidate. A fresh binding must
qualify these carriers against actual caller inputs; historical kernel fixtures
are not admitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from reference import validate_contract

_PUBLIC = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids", "w1_scale", "w2_scale")
_MUTABLE = ("hidden_states", "topk_weight", "topk_ids")
_CANONICAL = ("w1", "w2", "w1_scale", "w2_scale")
_LAYOUT_ATTRIBUTES = ("is_shuffled", "gate_up_interleave")


def _restore_public_weight_attributes(tensors, attributes):
    """Restore exactly the captured markers, including an absent optional one.

    The canonical interleaved preparation convention is separately bound by the
    task and checked by forward relayout. It is not inferred from this marker.
    """
    if not isinstance(attributes, dict) or set(attributes) != {"w1", "w2"}:
        raise ValueError("Exact public weight layout attributes must be bound")
    for name, gate_up in (("w1", True), ("w2", False)):
        row = attributes[name]
        if (not isinstance(row, dict) or set(row) not in ({"is_shuffled"}, set(_LAYOUT_ATTRIBUTES))
                or row.get("is_shuffled") is not True
                or ("gate_up_interleave" in row
                    and (type(row["gate_up_interleave"]) is not bool or row["gate_up_interleave"] is not gate_up))):
            raise ValueError(f"Unsupported or unresolved layout attributes for {name}")
    for name in ("w1", "w2"):
        for key in _LAYOUT_ATTRIBUTES:
            if key in attributes[name]:
                setattr(tensors[name], key, attributes[name][key])
            elif hasattr(tensors[name], key):
                delattr(tensors[name], key)


def _validate_canonical_scale_shape(weight, scale):
    # Forward shuffle pads scale columns to eight. A padded canonical scale can
    # therefore reshuffle to the same public bytes but cannot feed the oracle's
    # (experts, rows, logical_K // 32) reshape.
    if len(weight.shape) != 3 or any(size <= 0 for size in weight.shape) or weight.shape[2] % 16:
        raise ValueError("Canonical packed FP4 weights require positive per-1x32 geometry")
    experts, rows, packed_k = weight.shape
    if tuple(scale.shape) != (experts * rows, packed_k // 16):
        raise ValueError("Canonical scales must have the unpadded per-1x32 weight geometry")


def _storage_identity(tensor):
    # Keep storage identity distinct from its address: resize_ may move a
    # storage, and different empty storages can have the same null address.
    return tensor.device, tensor.untyped_storage()._cdata


def _validate_public_tensors(tensors, declared):
    if not isinstance(declared, dict) or set(declared) != set(tensors):
        raise ValueError("Manifest must declare every public tensor's layout and storage")
    group_to_storage, storage_to_group = {}, {}
    for name, tensor in tensors.items():
        metadata = declared[name]
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing public tensor metadata: {name}")
        group = metadata.get("storage_group")
        if type(group) is not int or group < 0:
            raise ValueError(f"storage_group must be a capture-local nonnegative integer: {name}")
        for field in ("storage_offset", "storage_nbytes"):
            if type(metadata.get(field)) is not int or metadata[field] < 0:
                raise ValueError(f"Invalid {field} for public tensor: {name}")
        actual = {
            "shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "storage_offset": tensor.storage_offset(),
            "storage_nbytes": tensor.untyped_storage().nbytes(), "storage_group": group,
        }
        if metadata != actual:
            raise ValueError(f"Actual tensor differs from protected manifest: {name}")
        identity = _storage_identity(tensor)
        if group_to_storage.setdefault(group, identity) != identity or storage_to_group.setdefault(identity, group) != group:
            raise ValueError(f"Public tensor storage aliases differ from protected manifest: {name}")


def resolve_input_path(root, descriptor):
    logical = descriptor.get("path") if isinstance(descriptor, dict) else None
    if (not isinstance(logical, str) or not logical or Path(logical).is_absolute()
            or Path(logical).as_posix() != logical or ".." in Path(logical).parts
            or logical == "."):
        raise ValueError("Case input path must be a canonical task-relative path")
    root = Path(root).resolve(strict=True)
    workload = root / "workloads.json"
    data = json.loads(workload.read_text()) if workload.is_file() else {}
    binding = data.get("runtime_binding") or {}
    assets = binding.get("input_assets", {})
    if not isinstance(assets, dict):
        raise ValueError("Runtime input_assets must be a mapping")
    local = root / logical
    if logical not in assets:
        path = local.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Case input path escapes the task or is not a file")
        return path

    asset = assets[logical]
    if not isinstance(asset, dict) or set(asset) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Runtime input asset requires path, sha256 and size_bytes")
    digest = asset["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest != descriptor.get("sha256")):
        raise ValueError("Runtime input asset differs from the case digest")
    if type(asset["size_bytes"]) is not int or asset["size_bytes"] < 0:
        raise ValueError("Runtime input asset requires an exact nonnegative size")
    if not isinstance(asset["path"], str):
        raise ValueError("Runtime input asset requires a canonical absolute path")
    path = Path(asset["path"])
    if (not path.is_absolute() or str(path) != asset["path"]
            or path.resolve(strict=True) != path or path.is_relative_to(root)):
        raise ValueError("Runtime input asset must be canonical and outside the task")
    if os.path.lexists(local):
        raise ValueError("A declared runtime input asset cannot have a local shadow")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != asset["size_bytes"]:
        raise ValueError("Runtime input asset file type or size differs")
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        raise ValueError("Runtime input asset must be on a read-only mount")
    return path


def _load_tensor_file(root, descriptor, device):

    if not isinstance(descriptor, dict):
        raise ValueError("Every input/weight carrier requires a protected path and SHA256")
    path = resolve_input_path(root, descriptor)
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Case input binding requires a lowercase SHA256")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Case input digest differs from protected binding")
        stream.seek(0)
        import torch
        tensors = torch.load(stream, map_location=device, weights_only=True)
    if not isinstance(tensors, dict):
        raise ValueError("Input/weight carrier must contain a tensor mapping")
    if any(not isinstance(value, torch.Tensor) or value.layout != torch.strided for value in tensors.values()):
        raise ValueError("Input payload must contain only dense tensors")
    return tensors


def _validate_mutable_tensors(tensors, experts):
    import torch
    hidden, ids, weights = (tensors[name] for name in ("hidden_states", "topk_ids", "topk_weight"))
    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or ids.ndim != 2 or ids.dtype != torch.int32:
        raise ValueError("Unsupported public hidden-state or routing layout")
    if weights.shape != ids.shape or weights.dtype != torch.float32 or ids.shape[0] != hidden.shape[0]:
        raise ValueError("Routing tensor geometry/dtype disagrees")
    if not torch.isfinite(hidden).all() or not torch.isfinite(weights).all() or (ids < 0).any() or (ids >= experts).any():
        raise ValueError("Nonfinite inputs or invalid local expert IDs")


def _public_tensors(payload):
    return dict(zip(_PUBLIC[:5], payload["args"])) | {
        name: payload["kwargs"][name] for name in (*_PUBLIC[5:], "bias1", "bias2")
        if name in payload["kwargs"]
    }


def load_current_value_replay(root, case, payload):
    """Load a second authentic point with the same public geometry and weights.

    The protected capture receipt owns the same-caller/prepared-weight join.
    This carrier contains only the three mutable tensors, in the original
    carrier format; it cannot supply replacement weights or caller arguments.
    """
    descriptor = case.get("inputs", {}).get("current_value_replay")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("Every case requires a protected current_value_replay carrier")
    public = _public_tensors(payload)
    tensors = _load_tensor_file(root, descriptor, public["hidden_states"].device)
    if set(tensors) != set(_MUTABLE):
        raise ValueError("Replay carrier must contain only hidden states and routing tensors")
    _validate_public_tensors(public | tensors, case.get("params", {}).get("tensors"))
    _validate_mutable_tensors(tensors, payload["reference"]["w1"].shape[0])
    return tensors


def copy_current_value_replay(payload, replacements):
    """Copy complete mutable backing buffers without rebinding public tensors."""
    import torch
    if not isinstance(replacements, dict) or set(replacements) != set(_MUTABLE):
        raise ValueError("Replay carrier must contain only hidden states and routing tensors")
    public = _public_tensors(payload)
    protected = {_storage_identity(value) for name, value in public.items() if name not in _MUTABLE}
    pairs, source_to_destination = {}, {}
    # Preflight every buffer before the first deliberate write. Preserve shared
    # storage and slack bytes; copying each visible tensor separately cannot do
    # that for overlapping views or mixed-dtype aliases.
    for name in _MUTABLE:
        destination, source = public[name], replacements[name]
        layout = lambda value: (value.shape, value.stride(), value.dtype, value.device,
                                value.storage_offset(), value.untyped_storage().nbytes(),
                                value.is_conj(), value.is_neg())
        if layout(source) != layout(destination):
            raise ValueError(f"Replay tensor metadata differs from original input: {name}")
        destination_id, source_id = _storage_identity(destination), _storage_identity(source)
        if destination_id in protected:
            raise ValueError("Replay mutable inputs must not alias public weight storage")
        if destination_id in pairs and pairs[destination_id][0] != source_id:
            raise ValueError("Replay mutable storage aliases differ from original inputs")
        if source_to_destination.setdefault(source_id, destination_id) != destination_id:
            raise ValueError("Replay mutable storage aliases differ from original inputs")
        pairs[destination_id] = (source_id, destination, source)
    for _, destination, source in pairs.values():
        nbytes = destination.untyped_storage().nbytes()
        whole = lambda value: torch.empty(0, dtype=torch.uint8, device=value.device).set_(
            value.untyped_storage(), 0, (nbytes,), (1,))
        whole(destination).copy_(whole(source))


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
    if descriptor.get("input_weight_storages_disjoint") is not True:
        raise ValueError("Capture must attest disjoint input/weight storages before splitting carriers")
    tensors = _load_tensor_file(root, descriptor, device)
    if set(tensors) != {"hidden_states", "topk_weight", "topk_ids"}:
        raise ValueError("Case carrier must contain only hidden states and routing tensors")
    weights = _load_tensor_file(root, descriptor.get("weights"), device)
    required_weights = set(_CANONICAL) | {"canonical_" + name for name in _CANONICAL}
    if not required_weights <= weights.keys() or weights.keys() - required_weights - {"bias1", "bias2"}:
        raise ValueError("Weight carrier has missing or unsupported tensors")
    tensors.update(weights)
    import torch

    public_names = set(_PUBLIC) | ({"bias1", "bias2"} & tensors.keys())
    _validate_public_tensors(
        {name: tensors[name] for name in sorted(public_names)}, case.get("params", {}).get("tensors")
    )
    _restore_public_weight_attributes(tensors, descriptor.get("tensor_attributes"))

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
        _validate_canonical_scale_shape(canonical[name], canonical[name + "_scale"])
    for name, gate_up in (("w1", True), ("w2", False)):
        packed = shuffle_weight_a16w4(canonical[name], 16, gate_up)
        scales = shuffle_scale_a16w4(canonical[name + "_scale"], experts, gate_up)
        for field, rebuilt in ((name, packed), (name + "_scale", scales)):
            public = tensors[field]
            if public.shape != rebuilt.shape or public.dtype != rebuilt.dtype or not torch.equal(
                public.contiguous().view(torch.uint8), rebuilt.contiguous().view(torch.uint8)
            ):
                raise ValueError(f"Public/canonical weight binding disagrees: {field}")
    _validate_mutable_tensors(tensors, experts)
    hidden, ids = (tensors[name] for name in ("hidden_states", "topk_ids"))
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
    """Snapshot each full public backing storage once, outside timing."""
    import torch
    values = list(payload["args"]) + list(payload["kwargs"].values())
    seen, tensors, storages = set(), [], {}
    for value in values:
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            identity = _storage_identity(value)
            storage = value.untyped_storage()
            if identity not in storages:
                nbytes = storage.nbytes()
                whole = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (nbytes,), (1,))
                storages[identity] = (storage, storage.data_ptr(), nbytes, whole.clone())
            attributes = {name: getattr(value, name) for name in _LAYOUT_ATTRIBUTES if hasattr(value, name)}
            tensors.append((value, value.shape, value.stride(), value.dtype, value.device,
                            identity, value.storage_offset(), attributes))
    return {"tensors": tensors, "storages": storages}


def check_inputs_unchanged(snapshots):
    import torch
    for tensor, shape, stride, dtype, device, identity, offset, attributes in snapshots["tensors"]:
        if tensor.shape != shape or tensor.stride() != stride or tensor.dtype != dtype or tensor.device != device:
            raise AssertionError("Public input metadata changed")
        if _storage_identity(tensor) != identity or tensor.storage_offset() != offset:
            raise AssertionError("Public input storage association or offset changed")
        current_attributes = {name: getattr(tensor, name) for name in _LAYOUT_ATTRIBUTES if hasattr(tensor, name)}
        if current_attributes != attributes or any(type(current_attributes[name]) is not type(value) for name, value in attributes.items()):
            raise AssertionError("Public input layout attributes changed")
    for storage, pointer, nbytes, before in snapshots["storages"].values():
        if storage.data_ptr() != pointer or storage.nbytes() != nbytes:
            raise AssertionError("Public input backing storage address or size changed")
        whole = torch.empty(0, dtype=torch.uint8, device=before.device).set_(storage, 0, (nbytes,), (1,))
        if not torch.equal(whole, before):
            raise AssertionError("The complete-call input contract is read-only; backing storage changed")
