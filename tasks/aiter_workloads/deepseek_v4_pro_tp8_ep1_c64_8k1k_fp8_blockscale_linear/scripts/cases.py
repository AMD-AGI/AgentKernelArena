"""Fixed captured inputs for the enrolled DeepSeek V4 FP8 linear call.

The prefill and decode cases each bind an original carrier and a
current_value_replay carrier: four authentic tensor-only torch.save mappings
in total. Each contains fp8 x, required fp32 x_scale, and the call's own
preshuffled weight and weight_scale, with captured shapes, strides, dtypes and
backing storage retained. Enrolled inputs have no random distribution and are
not regenerated from a seed. The producer's activation quantization is outside
the call.

The canonical reference weight is recovered by inverting the (16, 16)
preshuffle and checked by re-shuffling it with the pinned shuffle_weight.
Each replay carrier has the same prepared weight and weight_scale and a new
activation state. Shapes, dtypes and layouts are bound in workloads.json.

Synthetic/BF16 helpers remain available for unenrolled diagnostics; synthetic
inputs are never scored.
Public functions take explicit arguments and keep no module state.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat

from reference import quantize_1x128, validate_contract

TENSORS = {"bf16": ("x", "weight", "weight_scale"), "fp8": ("x", "x_scale", "weight", "weight_scale")}
PREPARED = ("weight", "weight_scale")


def preshuffle_layout(canonical):
    """Independent statement of the (16, 16) fp8 preshuffle (inverse of SGLang unshuffle_aiter_fp8_weight)."""
    n, k = canonical.shape
    return canonical.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).reshape(n, k)


def unshuffle_layout(weight):
    """Inverse of preshuffle_layout."""
    n, k = weight.shape
    return weight.view(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(n, k)


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


def _captured(root, descriptor, names, device):
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
    if not isinstance(tensors, dict) or set(tensors) != set(names):
        raise ValueError("Captured payload must hold exactly the call tensors for its input_form")
    if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided for t in tensors.values()):
        raise ValueError("Captured payload must contain only dense tensors")
    if any(not torch.isfinite(tensors[name].float()).all() for name in names):
        raise ValueError("Captured payload must be finite")
    return tensors


def _synthetic(m, n, k, seed, form, contract, device):
    import torch

    block = contract["block_size"]
    fp8 = getattr(torch, contract["fp8_dtype"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    activation = torch.randn((m, k), generator=generator, device=device).to(torch.bfloat16)
    canonical = (torch.randn((n, k), generator=generator, device=device) * 100.0).clamp(-448.0, 448.0).to(fp8)
    exponents = torch.randint(-1, 2, (n // block, k // block), generator=generator, device=device)
    weight_scale = torch.exp2(exponents.float() - round(math.log2(100.0 * math.sqrt(k))))
    tensors = {"x": activation, "weight": preshuffle_layout(canonical), "weight_scale": weight_scale}
    if form == "fp8":
        q, scale = quantize_1x128(activation, contract)
        x_scale = torch.empty((k // block, m), dtype=torch.float32, device=device).t()
        x_scale.copy_(scale)
        tensors.update(x=q, x_scale=x_scale)
    return tensors, canonical


def load_case(root, case, contract, *, device="cuda"):
    """Return {"args": (x, weight, weight_scale), "kwargs": {} or {"x_scale"},
    "reference": {"canonical_weight"}, "captured": bool}.

    Enrolled cases are captured FP8 inputs and always supply kwargs["x_scale"].
    """
    validate_contract(contract)
    params = case["params"]
    m, n, k, seed, form = (params.get(key) for key in ("m", "n", "k", "seed", "input_form"))
    block = contract["block_size"]
    if any(type(v) is not int or v <= 0 for v in (m, n, k)) or form not in TENSORS:
        raise ValueError("Case requires positive integer m/n/k and a known input_form")
    if n % block or k % block or case.get("shape") != [m, n, k] or case.get("dtype") != "bfloat16":
        raise ValueError("Case geometry/dtype is outside this binding")
    import torch
    from aiter.ops.shuffle import shuffle_weight

    if is_captured(case):
        tensors = _captured(root, case["inputs"], TENSORS[form], device)
        canonical = unshuffle_layout(tensors["weight"])
    elif type(seed) is int:
        tensors, canonical = _synthetic(m, n, k, seed, form, contract, device)
    else:
        raise ValueError("Case needs a protected captured input or an integer synthetic seed")
    if not torch.equal(shuffle_weight(canonical, (16, 16)).view(torch.uint8), tensors["weight"].view(torch.uint8)):
        raise ValueError("Pinned shuffle_weight disagrees with the declared (16, 16) preshuffle layout")
    _validate_public_tensors(tensors, params.get("tensors"))
    kwargs = {"x_scale": tensors["x_scale"]} if form == "fp8" else {}
    return {"args": (tensors["x"], tensors["weight"], tensors["weight_scale"]), "kwargs": kwargs,
            "reference": {"canonical_weight": canonical}, "captured": is_captured(case)}


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


def _public(payload):
    names = ("x", "weight", "weight_scale")
    return dict(zip(names, payload["args"]), **payload["kwargs"])


def _whole_storage(tensor):
    import torch
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


def _replay_storage_pairs(payload, second):
    """Preflight every operand before any dynamic storage is overwritten."""
    import torch
    public = _public(payload)
    if not isinstance(second, dict) or set(second) != set(public):
        raise ValueError("Replay carrier must hold exactly the public call tensors")
    prepared = {_storage_identity(public[name]) for name in PREPARED}
    pairs, source_to_destination = {}, {}
    for name in public:
        destination, source = public[name], second[name]
        layout = lambda value: (value.shape, value.stride(), value.dtype, value.device,
                                value.storage_offset(), value.untyped_storage().nbytes(),
                                value.is_conj(), value.is_neg())
        if layout(source) != layout(destination):
            raise ValueError(f"Replay tensor metadata differs from original input: {name}")
        destination_id, source_id = _storage_identity(destination), _storage_identity(source)
        if name not in PREPARED and destination_id in prepared:
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
    join. The carrier retains every public tensor and its full storage layout;
    only the activation (and x_scale for fp8in) will update the original buffers.
    The canonical reference weight therefore stays valid for both points.
    """
    descriptor = case.get("inputs", {}).get("current_value_replay")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("Every captured case requires a protected current_value_replay carrier")
    second = _captured(root, descriptor, TENSORS[case["params"]["input_form"]], payload["args"][0].device)
    _validate_public_tensors(second, case.get("params", {}).get("tensors"))
    _replay_storage_pairs(payload, second)
    return second


def copy_current_value_replay(payload, second):
    """Copy each dynamic backing buffer once, retaining every public object."""
    for destination, source in _replay_storage_pairs(payload, second):
        _whole_storage(destination).copy_(_whole_storage(source))
