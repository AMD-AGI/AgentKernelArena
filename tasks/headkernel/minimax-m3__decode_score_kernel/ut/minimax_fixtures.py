"""Portable raw fixture bundles; no capture-library or external repository imports."""
import hashlib
from pathlib import Path, PurePosixPath

from evaluation_contract import canonical, require, strict_json
from served_contract import FLOAT_INPUTS, span_bytes


def path_inside(root, relative):
    require(isinstance(relative, str) and relative and not PurePosixPath(relative).is_absolute()
            and not any(part in ("", ".", "..") for part in relative.split("/")), "unsafe fixture path")
    root = Path(root).resolve()
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "fixture is missing or escapes the materialized task")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def specification(meta, role):
    return {"role": role, "shape": meta["shape"], "strides": meta["stride"],
            "storage_offset": meta["storage_offset"],
            "dtype": meta["dtype"].removeprefix("torch."), "device_type": "cuda"}


def load_bundle(root, case, definition):
    reference = case["fixture"]
    path = path_inside(root, reference["path"])
    require(reference["path"].startswith("fixtures/") and path.stat().st_size <= 64 << 20
            and file_hash(path) == reference["sha256"], "representative bundle identity differs")
    bundle = strict_json(path.read_text())
    require(bundle.get("schema") == "served-tensor-fixture-v1"
            and bundle.get("bundle_schema") == "minimax-representatives-v1"
            and bundle.get("case_id") == case["case_id"], "wrong representative bundle")
    entries = bundle.get("representatives")
    require(isinstance(entries, list) and len(entries) == len(case["states"]) <= 3,
            "complete first/min/max state bundle is required")
    result = []
    for state, entry in zip(case["states"], entries):
        source = entry.get("source_json")
        require(isinstance(source, str) and hashlib.sha256(source.encode()).hexdigest() == state["fixture_sha256"]
                == entry.get("source_sha256"), "original served fixture bytes changed")
        fixture = strict_json(source)
        require(fixture.get("source_sha256") == definition["source_sha256"]
                and fixture.get("family") == definition["kind"] and fixture.get("startup_values") is False
                and fixture.get("origin") == "served_" + case["production_mode"], "wrong fixture source/origin")
        source_key = case.get("fixture_transfer", {}).get("representative_case_key", case.get("source_capture_key"))
        require(source_key is None or fixture.get("case_key") == source_key, "fixture belongs to another source case")
        require(fixture["served"] == state["served_context"]
                and fixture["tensor_controls"] == state["tensor_controls"], "fixture work state differs")
        expected_inputs = {name for name, spec in case["tensors"].items() if spec["role"] == "input"}
        inputs = {name: meta for name, meta in fixture["inputs"].items() if name in definition["arguments"]}
        require(set(inputs) == expected_inputs, "fixture argument set differs")
        for name, meta in inputs.items():
            require(specification(meta, "input") == case["tensors"][name]
                    and meta["alias"] == case["input_aliases"][name]
                    and meta["storage_nbytes"] == case["original_storage_nbytes"][name], "fixture physical input ABI differs")
        scalar_args = {k: v for k, v in case["scalars"].items() if not k.startswith(("work.", "result"))}
        require(canonical(fixture["controls"]["operator_scalars"]) == canonical(scalar_args), "fixture scalar ABI differs")
        for name, meta in fixture["outputs"].items():
            if meta is None:
                require(name in case["scalars"] and case["scalars"][name] is None, "optional output differs")
            else:
                require(specification(meta, "output") == case["tensors"][name], "fixture output ABI differs")
        storage_directory = entry["storage_directory"]
        require(isinstance(storage_directory, str) and "/" not in storage_directory
                and storage_directory not in ("", ".", ".."), "invalid fixture storage directory")
        result.append({"fixture": fixture, "root": Path(root)/"fixtures"/storage_directory})
    return result


def _segments(entry, phase, alias, verified):
    group = entry["fixture"]["payload"][phase][alias]
    end = 0
    for segment in sorted(group["segments"], key=lambda item: item["offset_bytes"]):
        start, size = segment["offset_bytes"], segment["bytes"]
        require(type(start) is int and type(size) is int and 0 <= end <= start <= start+size <= group["storage_nbytes"],
                "invalid or overlapping captured storage segment")
        path = path_inside(entry["root"], segment["blob"])
        identity = (str(path), size, segment["sha256"])
        if identity not in verified:
            require(path.stat().st_size == size and file_hash(path) == segment["sha256"], "captured tensor blob changed")
            verified.add(identity)
        yield start, size, path
        end = start+size


def restore_cpu(entry, phase, names, verified=None, *, limit=2 << 30):
    import torch
    verified = set() if verified is None else verified
    metadata = entry["fixture"][phase]
    specs = {name: specification(metadata[name], "input") for name in names if metadata[name] is not None}
    sizes = {}
    for name, spec in specs.items():
        alias = metadata[name]["alias"]
        sizes[alias] = max(sizes.get(alias, 0), span_bytes(spec))
    require(sum(sizes.values()) <= limit, "bounded CPU fixture reconstruction exceeded")
    storage = {alias: torch.zeros(size, dtype=torch.uint8) for alias, size in sizes.items()}
    coverage = {}
    for alias, target in storage.items():
        coverage[alias] = []
        for start, size, path in _segments(entry, phase, alias, verified):
            end = min(start+size, target.numel())
            if start >= end:
                continue
            with path.open("rb") as source:
                cursor = start
                while cursor < end:
                    data = source.read(min(8 << 20, end-cursor))
                    require(data, "truncated captured storage segment")
                    target[cursor:cursor+len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
                    cursor += len(data)
            coverage[alias].append((start, end))
    values = {name: None if metadata[name] is None else torch.empty(0, dtype=getattr(torch, specs[name]["dtype"])).set_(
        storage[metadata[name]["alias"]].untyped_storage(), specs[name]["storage_offset"],
        tuple(specs[name]["shape"]), tuple(specs[name]["strides"])) for name in names}
    def covered(alias, first, last):
        cursor = first
        for a, b in coverage[alias]:
            if a > cursor:
                return False
            cursor = max(cursor, b)
            if cursor >= last:
                return True
        return cursor >= last
    for name, spec in specs.items():
        if name == "req_to_token":
            continue
        meta = metadata[name]
        require(covered(meta["alias"], meta["storage_offset"]*meta["element_size"], span_bytes(spec)),
                "captured control/output span is incomplete: " + name)
    if "req_to_token" in names:
        table = values["req_to_token"]; meta = metadata["req_to_token"]
        total = entry["fixture"]["inputs"]["k_cache"]["shape"][0]
        for length, slot in zip(values["seq_lens"].tolist(), values["slot_ids"].tolist()):
            require(0 <= length <= table.shape[1], "captured sequence length escapes paging capacity")
            if length:
                row = (slot+total) % total
                require(0 <= row < table.shape[0], "captured request row escapes paging capacity")
                start = (meta["storage_offset"]+row*meta["stride"][0])*meta["element_size"]
                require(covered(meta["alias"], start, start+table.shape[1]*meta["element_size"]),
                        "active captured paging row is incomplete")
    return values


def geometry(entry, names, verified):
    return restore_cpu(entry, "inputs", set(names)-FLOAT_INPUTS, verified)


def restore_recorded_inputs(entry, storage, verified):
    import torch
    for alias, target in storage.items():
        target.zero_()
        for start, size, path in _segments(entry, "inputs", alias, verified):
            require(start+size <= target.numel(), "captured bytes exceed original input storage")
            with path.open("rb") as source:
                cursor = start
                while cursor < start+size:
                    data = source.read(min(8 << 20, start+size-cursor))
                    require(data, "truncated captured input")
                    target[cursor:cursor+len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).to(target.device))
                    cursor += len(data)
