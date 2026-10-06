"""Full-storage raw fixture validation/restoration; no pickle or missing bytes."""
import hashlib
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fixture path escapes package")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("fixture symlinks are forbidden")
    if not path.is_file():
        raise ValueError("fixture file is missing: " + name)
    return path


def validate_fixture(root, fixture, *, max_storage_bytes=2 << 30, allow_diagnostic=False):
    if fixture.get("provenance", {}).get("synthetic") and not allow_diagnostic:
        raise ValueError("diagnostic synthetic fixtures cannot be admitted as workload cases")
    if fixture.get("schema") != "served-tensor-fixture-v1" or fixture.get("startup_values") is not False:
        raise ValueError("actual served fixture required")
    if fixture.get("family") != "minimax_fp4_gemm" or fixture.get("origin") not in ("served_eager", "served_graph"):
        raise ValueError("wrong captured operator or execution origin")
    if set(fixture["inputs"]) != {"x", "w", "x_scales", "w_scales", "y"} or set(fixture["outputs"]) != {"result"}:
        raise ValueError("captured tensor bindings differ")
    for phase in ("inputs", "outputs"):
        groups = fixture["payload"][phase]
        if sum(g["storage_nbytes"] for g in groups.values()) > max_storage_bytes:
            raise ValueError("fixture storage exceeds the explicit restoration budget")
        for group in groups.values():
            size = group["storage_nbytes"]
            if type(size) is not int or size < 0:
                raise ValueError("invalid storage size")
            end = 0
            for segment in sorted(group["segments"], key=lambda item: item["offset_bytes"]):
                path = safe_file(root, segment["blob"])
                if segment["offset_bytes"] != end or path.stat().st_size != segment["bytes"] or file_sha(path) != segment["sha256"]:
                    raise ValueError("full-storage segment has a gap, overlap, or changed bytes")
                end += segment["bytes"]
            if end != size:
                raise ValueError("fixture does not contain the complete physical storage")
        for meta in fixture[phase].values():
            if meta is None:
                continue
            if meta.get("attributes") or meta["role"]["footprint"] != "full_storage":
                raise ValueError("unexpected tensor attributes or partial storage capture")
            geometry = [meta["storage_offset"], *meta["shape"], *meta["stride"]]
            if len(meta["shape"]) != len(meta["stride"]) or any(type(v) is not int or v < 0 for v in geometry):
                raise ValueError("invalid captured geometry")
            extent = meta["storage_offset"] + 1 + sum((n - 1) * stride for n, stride in zip(meta["shape"], meta["stride"]))
            if extent * meta["element_size"] > groups[meta["alias"]]["storage_nbytes"]:
                raise ValueError("captured view exceeds physical storage")


def views(raw, metadata):
    import torch
    result = {}
    for name, meta in metadata.items():
        if meta is None:
            result[name] = None
            continue
        dtype = getattr(torch, meta["dtype"].removeprefix("torch."))
        buffer = raw[meta["alias"]]
        result[name] = torch.empty(0, dtype=dtype, device=buffer.device).set_(
            buffer.untyped_storage(), meta["storage_offset"], meta["shape"], meta["stride"])
    return result


def restore(root, fixture, phase, *, allow_diagnostic=False):
    import torch
    validate_fixture(root, fixture, allow_diagnostic=allow_diagnostic)
    raw = {}
    for alias, group in fixture["payload"][phase].items():
        buffer = torch.empty(group["storage_nbytes"], dtype=torch.uint8, device="cpu")
        for segment in group["segments"]:
            offset = segment["offset_bytes"]
            with safe_file(root, segment["blob"]).open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    buffer[offset:offset + len(block)].copy_(torch.frombuffer(bytearray(block), dtype=torch.uint8))
                    offset += len(block)
        raw[alias] = buffer
    return raw, views(raw, fixture[phase])
