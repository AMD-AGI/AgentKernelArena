"""Immutable helpers for reconstructing captured tensor contracts.

No tensors are synthesized here: shape, dtype, physical stride and Python tensor
attributes are restored from the recorded payload, or reconstruction fails.
"""
from pathlib import Path


def task_path(task_root, relative):
    """Resolve a declared task input while rejecting absolute paths and escapes."""
    root = Path(task_root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"task input must use a relative path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"task input escapes its task directory: {relative}")
    return result


def _dtype(torch, value):
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported captured dtype: {value!r}")
    return dtype


def tensor_attrs(tensor, attrs):
    """Attributes such as is_shuffled select real kernel layouts; never drop them."""
    for name, value in (attrs or {}).items():
        if not isinstance(name, str) or name.startswith("__"):
            raise ValueError(f"invalid captured tensor attribute: {name!r}")
        setattr(tensor, name, value)
        if getattr(tensor, name) != value:
            raise RuntimeError(f"could not restore captured tensor attribute {name!r}")
    return tensor


def restore_tensor(torch, node, device):
    """Restore a serialized tensor without silently casting or flattening layouts."""
    data = node["data"]
    value = data.to(device)
    packed = node.get("packed_view_dtype")
    if not packed and node.get("bitcast"):
        packed = node.get("dtype")
        if not packed:
            raise ValueError("bitcast tensor is missing its recorded dtype")
    if packed:
        value = value.view(_dtype(torch, packed))
    expected_dtype = node.get("dtype")
    if expected_dtype and value.dtype != _dtype(torch, expected_dtype):
        raise ValueError(f"capture payload dtype {value.dtype} != recorded {expected_dtype}")
    shape = tuple(int(v) for v in node.get("shape", value.shape))
    if any(v < 0 for v in shape):
        raise ValueError(f"invalid captured shape: {shape}")
    if tuple(value.shape) != shape:
        value = value.reshape(shape)
    stride = node.get("stride", node.get("strides"))
    if stride is not None:
        stride = tuple(int(v) for v in stride)
        if len(stride) != len(shape) or any(v < 0 for v in stride):
            raise ValueError(f"invalid captured stride {stride} for shape {shape}")
        if tuple(value.stride()) != stride:
            exact = torch.empty_strided(shape, stride, dtype=value.dtype, device=device)
            exact.copy_(value)
            value = exact
        if tuple(value.stride()) != stride:
            raise RuntimeError("failed to restore captured physical stride")
    elif node.get("contiguous") is True:
        value = value.contiguous()
    return tensor_attrs(value, node.get("tensor_attrs") or node.get("attrs"))


def verified_torch_load(torch, path, **kwargs):
    """Check a frozen oracle/geometry digest immediately before deserialization.

    Generated baseline parity outputs are loaded by a separate path in the
    harness; they must never be mistaken for a persistent captured oracle.
    """
    import hashlib
    import json
    import re
    task_ut = Path(__file__).resolve().parent
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    relative = path.relative_to(task_ut)
    path = task_path(task_ut, relative)
    with (task_ut / "meta.json").open() as stream:
        meta = json.load(stream)
    names = {
        meta.get("reference_io", "reference_io.pt"): "reference_io_sha256",
        meta.get("geometry_file", "timing_geometry.pt"): "timing_geometry_sha256",
    }
    field = names.get(relative.as_posix())
    expected = meta.get(field, "") if field else ""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"missing frozen SHA-256 for captured input {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"captured input SHA-256 mismatch for {relative}: "
                           f"expected {expected}, got {actual}")
    return torch.load(str(path), **kwargs)
