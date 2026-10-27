"""GPU inputs for one mHC post+pre boundary call.

A case with case['inputs'] = {"path", "sha256"} loads a protected, captured
tensor-only torch.save mapping of the eight tensors below (actual activations
and that call's own layer weights), keeping the captured shapes, strides and
dtypes. A case without it is SYNTHETIC: values are generated on the device
from params.seed and the evaluator never scores them. Synthetic distributions:
  layer_input    bf16 [m, hidden]        N(0, 1)
  residual_in    bf16 [m, hc, hidden]    N(0, 1)
  post_layer_mix fp32 [m, hc]            2 * sigmoid(N(0, 1)), the squeezed form SGLang passes
  comb_res_mix   fp32 [m, hc, hc]        row softmax of N(0, 1)
  fn             fp32 [(2+hc)*hc, hc*hidden]  N(0, 1) / sqrt(hc*hidden)       (weight)
  hc_scale       fp32 [3]                U(0.5, 1.0)                           (weight)
  hc_base        fp32 [(2+hc)*hc]        0.1 * N(0, 1)                         (weight)
  norm_weight    bf16 [hidden]           1 + 0.1 * N(0, 1)                     (weight)
Shapes, dtypes and the call constants follow the pinned SGLang call site.
Captured qualification also requires inputs.current_value_replay for a second
eight-tensor carrier with the same prepared weights and new dynamic state.
Public functions take explicit arguments and keep no module state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

TENSORS = ("layer_input", "residual_in", "post_layer_mix", "comb_res_mix",
           "fn", "hc_scale", "hc_base", "norm_weight")
DYNAMIC = TENSORS[:4]
PREPARED = TENSORS[4:]


def is_captured(case):
    return isinstance(case.get("inputs"), dict)


def _storage_identity(tensor):
    # Keep storage identity distinct from its address: resize_ may move a
    # storage, and different empty storages can have the same null address.
    return tensor.device, tensor.untyped_storage()._cdata


def _validate_public_tensors(tensors, declared):
    """Exact layout plus full backing storage (offset, size incl. holes, aliases) per the manifest."""
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


def _captured(root, descriptor, device):
    import torch

    path = resolve_input_path(root, descriptor)
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Case input binding requires a lowercase SHA256")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Case input digest differs from protected binding")
        stream.seek(0)
        tensors = torch.load(stream, map_location=device, weights_only=True)
    if not isinstance(tensors, dict) or set(tensors) != set(TENSORS):
        raise ValueError("Captured payload must hold exactly the eight call tensors")
    if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided or not torch.isfinite(t).all()
           for t in tensors.values()):
        raise ValueError("Captured payload must contain only dense finite tensors")
    return tensors


def _synthetic(m, seed, hc, hidden, device):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def normal(*shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    mix = (2 + hc) * hc
    return {
        "layer_input": normal(m, hidden).to(torch.bfloat16),
        "residual_in": normal(m, hc, hidden).to(torch.bfloat16),
        "post_layer_mix": 2.0 * torch.sigmoid(normal(m, hc)),
        "comb_res_mix": normal(m, hc, hc).softmax(-1),
        "fn": normal(mix, hc * hidden) * (hc * hidden) ** -0.5,
        "hc_scale": 0.5 + 0.5 * torch.rand((3,), generator=generator, device=device, dtype=torch.float32),
        "hc_base": 0.1 * normal(mix),
        "norm_weight": (1.0 + 0.1 * normal(hidden)).to(torch.bfloat16),
    }


def load_case(root, case, contract, *, device="cuda"):
    """Return {"args": 7 positional tensors, "kwargs": constants + norm_weight, "captured": bool}."""
    params = case["params"]
    m = params["m"]
    hc, hidden = contract["hc_mult"], contract["hidden_size"]
    if type(m) is not int or m <= 0:
        raise ValueError("Case requires a positive integer m")
    if case.get("shape") != [m, hc, hidden] or case.get("dtype") != "bfloat16":
        raise ValueError("Case geometry/dtype differs from the bound model geometry")
    if is_captured(case):
        tensors = _captured(root, case["inputs"], device)
    elif type(params.get("seed")) is int:
        tensors = _synthetic(m, params["seed"], hc, hidden, device)
    else:
        raise ValueError("Case needs a protected captured input or an integer synthetic seed")
    _validate_public_tensors(tensors, params.get("tensors"))
    kwargs = dict(contract["call_kwargs"], norm_weight=tensors["norm_weight"])
    return {"args": tuple(tensors[name] for name in TENSORS[:7]), "kwargs": kwargs,
            "captured": is_captured(case)}


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
            tensors.append((value, value.shape, value.stride(), value.dtype, value.device,
                            identity, value.storage_offset()))
    return {"tensors": tensors, "storages": storages}


def check_inputs_unchanged(snapshots):
    import torch
    for tensor, shape, stride, dtype, device, identity, offset in snapshots["tensors"]:
        if tensor.shape != shape or tensor.stride() != stride or tensor.dtype != dtype or tensor.device != device:
            raise AssertionError("Public input metadata changed")
        if _storage_identity(tensor) != identity or tensor.storage_offset() != offset:
            raise AssertionError("Public input storage association or offset changed")
    for storage, pointer, nbytes, before in snapshots["storages"].values():
        if storage.data_ptr() != pointer or storage.nbytes() != nbytes:
            raise AssertionError("Public input backing storage address or size changed")
        whole = torch.empty(0, dtype=torch.uint8, device=before.device).set_(storage, 0, (nbytes,), (1,))
        if not torch.equal(whole, before):
            raise AssertionError("The complete-call input contract is read-only; backing storage changed")


def _whole_storage(tensor):
    import torch
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


def _replay_storage_pairs(payload, second):
    """Preflight all eight operands before any dynamic storage is overwritten."""
    import torch
    if not isinstance(second, dict) or set(second) != set(TENSORS):
        raise ValueError("Replay carrier must hold exactly the eight public tensors")
    public = dict(zip(TENSORS, payload["args"] + (payload["kwargs"]["norm_weight"],)))
    prepared = {_storage_identity(public[name]) for name in PREPARED}
    pairs, source_to_destination = {}, {}
    for name in TENSORS:
        destination, source = public[name], second[name]
        layout = lambda value: (value.shape, value.stride(), value.dtype, value.device,
                                value.storage_offset(), value.untyped_storage().nbytes(),
                                value.is_conj(), value.is_neg())
        if layout(source) != layout(destination):
            raise ValueError(f"Replay tensor metadata differs from original input: {name}")
        destination_id, source_id = _storage_identity(destination), _storage_identity(source)
        if name in DYNAMIC and destination_id in prepared:
            raise ValueError("Dynamic replay inputs must not alias prepared weight storage")
        if destination_id in pairs and pairs[destination_id][0] != source_id:
            raise ValueError("Replay public storage aliases differ from original inputs")
        if source_to_destination.setdefault(source_id, destination_id) != destination_id:
            raise ValueError("Replay public storage aliases differ from original inputs")
        pairs[destination_id] = (source_id, destination, source)
    for identity in prepared:
        _, destination, source = pairs[identity]
        if not torch.equal(_whole_storage(destination), _whole_storage(source)):
            raise ValueError("Replay prepared weight backing bytes differ from original inputs")
    return [(destination, source) for identity, (_, destination, source) in pairs.items()
            if identity not in prepared]


def load_current_value_replay(root, case, payload):
    """Load a second authentic point; its prepared weights must equal the first's.

    The protected capture receipt owns the same-caller/rank/layer/prepared-seal
    join. The carrier retains all eight public tensors and their full storage
    layout; only the dynamic quartet will update the original public buffers.
    """
    descriptor = case.get("inputs", {}).get("current_value_replay")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("Every captured case requires a protected current_value_replay carrier")
    second = _captured(root, descriptor, payload["args"][0].device)
    _validate_public_tensors(second, case.get("params", {}).get("tensors"))
    _replay_storage_pairs(payload, second)
    return second


def copy_current_value_replay(payload, second):
    """Copy each dynamic backing buffer once, retaining every public object."""
    for destination, source in _replay_storage_pairs(payload, second):
        _whole_storage(destination).copy_(_whole_storage(source))
