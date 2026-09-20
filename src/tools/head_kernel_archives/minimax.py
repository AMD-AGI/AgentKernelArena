#!/usr/bin/env python3
"""Extract only MiniMax layout/scalar/routing contracts from a verified archive.

Run once where reference_io.pt already exists. No floating-point input values or
output values enter the resulting JSON; the original archive is opened read-only.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

RANDOM_INPUTS = {"q", "k_cache", "v_cache", "sink"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def positional_names(task):
    meta = json.loads((task / "ut/meta.json").read_text())
    target = meta["target_callable"].split(":")[-1]
    for path in (task / "source").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name == target:
                return [arg.arg for arg in node.args.posonlyargs + node.args.args]
    raise ValueError(f"cannot identify positional ABI for {target}")


def recorded_specs(meta):
    result = [entry for entry in meta["cases"] if entry.get("source", "recorded") == "recorded"]
    if len(result) != int(meta["num_cases"]):
        raise ValueError("metadata does not enumerate every recorded correctness case")
    return result


def scalar(node):
    if isinstance(node, dict) and set(node) == {"__repr__"}:
        return ast.literal_eval(node["__repr__"])
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    raise ValueError(f"unsupported captured scalar: {type(node).__name__}")


def extract_blob(blob, meta, names, torch):
    records = blob.get("records") or []
    expected = recorded_specs(meta)
    if [record["sig"] for record in records] != [entry["sig"] for entry in expected]:
        raise ValueError("archive case identity/order differs from recorded metadata")
    shared = blob.get("shared") or {}

    def resolve(value):
        while isinstance(value, dict) and set(value) == {"__shared__"}:
            value = shared[value["__shared__"]]
        return value

    result = []
    for index, record in enumerate(records):
        groups = {}
        positions = [resolve(x) for x in record.get("args", ())]
        keywords = {key: resolve(x) for key, x in (record.get("kwargs") or {}).items()}
        operands = dict(zip(names, positions))
        operands.update(keywords)
        slots = operands.get("slot_ids")
        if isinstance(slots, dict) and slots.get("__tensor__"):
            slot_values = slots["data"].reshape(-1).tolist()
        elif torch.is_tensor(slots):
            slot_values = slots.reshape(-1).tolist()
        else:
            raise ValueError("captured slot_ids tensor is required")

        def encode(value, name, output=False):
            value = resolve(value)
            if torch.is_tensor(value):
                value = {"__tensor__": True, "data": value}
            if isinstance(value, dict) and value.get("__tensor__"):
                data = value["data"]
                storage = data.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                group = groups.setdefault(key, f"storage_{len(groups)}")
                shape = list(value.get("shape", data.shape))
                stride = list(value.get("stride", data.stride()))
                offset = int(value.get("storage_offset", data.storage_offset()))
                required = offset + sum(max(int(n) - 1, 0) * int(s) for n, s in zip(shape, stride)) + 1
                desc = {"tensor": True, "shape": shape, "stride": stride,
                        "dtype": str(value.get("dtype", data.dtype)), "storage_offset": offset,
                        "storage_group": group,
                        "storage_numel": max(storage.nbytes() // data.element_size(), required),
                        "tensor_attrs": value.get("tensor_attrs") or value.get("attrs") or {}}
                if output:
                    desc["recipe"] = "runtime_reference"
                elif name in RANDOM_INPUTS:
                    desc["recipe"] = "normal_unit"
                elif name == "req_to_token":
                    desc.update(recipe="captured_rows", row_indices=slot_values,
                                rows=data[slot_values].tolist(), max_slots=expected[index]["max_slots"])
                else:
                    if data.numel() > 2_000_000:
                        raise ValueError(f"unexpected large structural tensor: {name}")
                    desc.update(recipe="captured_values", values=data.tolist())
                return desc
            if isinstance(value, (list, tuple)):
                return {"sequence": type(value).__name__, "items": [encode(x, name, output) for x in value]}
            if isinstance(value, dict) and "__repr__" not in value:
                return {key: encode(x, key, output) for key, x in value.items()}
            return scalar(value)

        encoded_args = [encode(value, names[i]) for i, value in enumerate(positions)]
        encoded_kwargs = {key: encode(value, key) for key, value in keywords.items()}
        result.append({"sig": record["sig"], "regime": record.get("regime") or expected[index]["regime"],
                       "args": encoded_args, "kwargs": encoded_kwargs,
                       "output_contract": encode(record.get("output"), "output", True)})
    return {"schema_version": 1, "input_policy": "generated_values_at_captured_contract",
            "source_reference_sha256": (meta.get("reference_io_sha256") or meta["archival_capture"]["reference_io_sha256"]),
            "correctness_case_count": len(result), "records": result}



@contextmanager
def verified_archive(path, expected):
    """Pin the read-only inode while hashing and CPU mmap deserialization."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source archive is not a regular file")
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        h = hashlib.sha256()
        while block := os.read(fd, 8 << 20):
            h.update(block)
        if h.hexdigest() != expected:
            raise ValueError("source archive SHA-256 mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        yield f"/proc/self/fd/{fd}"
        if stamp(os.fstat(fd)) != stamp(before):
            raise ValueError("source archive changed during extraction")
    finally:
        os.close(fd)
