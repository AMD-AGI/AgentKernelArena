"""Protected MiniMax generated-input contract; never infers missing captures."""
from __future__ import annotations
import base64
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_contract(ut):
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())
    settings = meta["generated_inputs"]
    path = ut / settings["contract_file"]
    if not path.is_file() or not settings.get("contract_sha256"):
        raise RuntimeError("authoritative compact capture extraction is required: ut/generated_cases.json")
    if digest(path) != settings["contract_sha256"]:
        raise RuntimeError("generated input contract SHA-256 mismatch")
    data = json.loads(path.read_text())
    expected = [row["sig"] for row in meta["cases"] if row.get("source", "recorded") == "recorded"]
    observed = [row["sig"] for row in data["records"]]
    if (data.get("schema_version") != 1 or data.get("source_reference_sha256") != (meta.get("reference_io_sha256") or meta["archival_capture"]["reference_io_sha256"])
            or observed != expected or len(observed) != int(meta["num_cases"])
            or len(set(observed)) != len(observed)):
        raise RuntimeError("compact contract dropped, reordered or changed a captured correctness case")
    return data


def dtype(torch, name):
    value = getattr(torch, name.removeprefix("torch."), None)
    if value is None or not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {name!r}")
    return value


def build_record(record, seed, torch, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    storages = {}

    def build(node):
        if isinstance(node, dict) and node.get("tensor"):
            dt = dtype(torch, node["dtype"])
            key = node["storage_group"]
            spec = (node["dtype"], int(node["storage_numel"]))
            if key not in storages:
                raw = torch.empty(spec[1], dtype=dt, device=device)
                if raw.is_floating_point():
                    raw.normal_(generator=generator)
                else:
                    raw.random_(0, max(1, int(node.get("max_slots", 127))), generator=generator)
                storages[key] = (spec, raw)
            prior, raw = storages[key]
            if prior != spec:
                raise ValueError("mixed dtype/extent aliases need an explicit storage recipe")
            value = raw.as_strided(tuple(node["shape"]), tuple(node["stride"]), node["storage_offset"])
            recipe = node["recipe"]
            if recipe == "captured_values":
                value.copy_(torch.tensor(node["values"], dtype=dt, device=device))
            elif recipe == "captured_rows":
                rows = torch.tensor(node["rows"], dtype=dt, device=device)
                for index, row in zip(node["row_indices"], rows):
                    value[int(index)].copy_(row)
            elif recipe != "normal_unit":
                raise ValueError(f"unsupported generated tensor recipe {recipe!r}")
            for name, val in node.get("tensor_attrs", {}).items():
                setattr(value, name, val)
            return value
        if isinstance(node, dict) and "sequence" in node:
            items = [build(item) for item in node["items"]]
            return tuple(items) if node["sequence"] == "tuple" else items
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    return tuple(build(item) for item in record["args"]), {key: build(value) for key, value in record["kwargs"].items()}


def encode_output(value, torch):
    """Use typed byte JSON, not pickle, across the untrusted output boundary."""
    if type(value) is torch.Tensor:
        cpu = value.detach().cpu().contiguous()
        data = cpu.view(torch.uint8).numpy().tobytes()
        return {"tensor": True, "shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "device": str(value.device), "storage_offset": value.storage_offset(),
                "data": base64.b64encode(data).decode("ascii")}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [encode_output(item, torch) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("candidate returned a non-contract output type")


def decode_output(value, torch):
    if isinstance(value, dict) and value.get("tensor"):
        dt = dtype(torch, value["dtype"])
        raw = base64.b64decode(value["data"], validate=True)
        expected = math.prod(value["shape"]) * torch.empty((), dtype=dt).element_size()
        if len(raw) != expected or expected > 256 << 20:
            raise ValueError("invalid candidate output byte count")
        return torch.frombuffer(bytearray(raw), dtype=dt).reshape(value["shape"])
    if isinstance(value, dict) and "sequence" in value:
        values = [decode_output(item, torch) for item in value["items"]]
        return tuple(values) if value["sequence"] == "tuple" else values
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("invalid worker output encoding")


def compare_output(observed, expected, tol, torch):
    """Use mixed tolerance for floating outputs and exact equality for integer/bool outputs."""
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(observed, dict) or not observed.get("tensor"):
            return False
        if any(observed[k] != expected[k] for k in ("shape", "stride", "dtype", "storage_offset", "device")):
            return False
        left, right = decode_output(observed, torch), decode_output(expected, torch)
        if not right.is_floating_point():
            return bool(torch.equal(left, right))
        left, right = left.float(), right.float()
        atol = float(tol) * right.pow(2).mean().sqrt().clamp_min(1e-6)
        return bool(((left - right).abs() <= atol + float(tol) * right.abs()).all())
    if isinstance(expected, dict) and "sequence" in expected:
        return (isinstance(observed, dict) and observed.get("sequence") == expected["sequence"]
                and len(observed.get("items", [])) == len(expected["items"])
                and all(compare_output(a, b, tol, torch) for a, b in zip(observed["items"], expected["items"])))
    return type(observed) is type(expected) and observed == expected


def load_geometry(ut, torch):
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())["generated_inputs"]
    path = ut / meta["geometry_file"]
    if digest(path) != meta["geometry_sha256"]:
        raise RuntimeError("compact geometry SHA-256 mismatch")

    def restore(node):
        if isinstance(node, dict) and node.get("tensor"):
            value = torch.tensor(node["values"], dtype=dtype(torch, node["dtype"]))
            value = value.reshape(node["shape"])
            if list(value.stride()) != node["stride"]:
                exact = torch.empty_strided(node["shape"], node["stride"], dtype=value.dtype)
                exact.copy_(value)
                value = exact
            return value
        if isinstance(node, dict):
            return {key: restore(value) for key, value in node.items()}
        if isinstance(node, list):
            return [restore(value) for value in node]
        return node

    return restore(json.loads(path.read_text())["records"])


def tensor_leaves(value, torch):
    if type(value) is torch.Tensor:
        return [value]
    if isinstance(value, dict):
        return [leaf for val in value.values() for leaf in tensor_leaves(val, torch)]
    if isinstance(value, (tuple, list)):
        return [leaf for val in value for leaf in tensor_leaves(val, torch)]
    return []


def snapshot_inputs(args, kwargs, torch):
    return [(value, (list(value.shape), list(value.stride()), str(value.dtype),
                    value.storage_offset(), value.data_ptr(), dict(value.__dict__)), value.clone())
            for value in tensor_leaves((args, kwargs), torch)]


def require_inputs_unchanged(saved, torch):
    for value, signature, copy in saved:
        current = (list(value.shape), list(value.stride()), str(value.dtype),
                   value.storage_offset(), value.data_ptr(), dict(value.__dict__))
        if current != signature or not torch.equal(value, copy):
            raise RuntimeError("candidate changed a read-only MiniMax input or its tensor contract")


def require_output_contract(encoded, expected):
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(encoded, dict) or not encoded.get("tensor"):
            raise RuntimeError("candidate changed the captured output tree")
        if any(encoded.get(key) != expected[key] for key in ("shape", "stride", "dtype", "storage_offset")):
            raise RuntimeError("candidate changed the captured output shape, stride or dtype")
    elif isinstance(expected, dict) and "sequence" in expected:
        if (not isinstance(encoded, dict) or encoded.get("sequence") != expected["sequence"]
                or len(encoded.get("items", [])) != len(expected["items"])):
            raise RuntimeError("candidate changed the captured tuple/None output ABI")
        for left, right in zip(encoded["items"], expected["items"]):
            require_output_contract(left, right)
    elif type(encoded) is not type(expected) or encoded != expected:
        raise RuntimeError("candidate changed a captured non-tensor output")


def clone_tree(value, torch):
    """Clone backing storages once, retaining exact tensor views and aliases."""
    storages = {}

    def clone(node):
        if type(node) is torch.Tensor:
            storage = node.untyped_storage()
            key = (str(node.device), storage.data_ptr(), storage.nbytes())
            if key not in storages:
                view = torch.empty(0, device=node.device, dtype=torch.uint8)
                view.set_(storage, 0, (storage.nbytes(),), (1,))
                storages[key] = view.clone().untyped_storage()
            output = torch.empty(0, device=node.device, dtype=node.dtype)
            output.set_(storages[key], node.storage_offset(), node.shape, node.stride())
            output.__dict__.update(node.__dict__)
            return output
        if isinstance(node, dict):
            return {key: clone(item) for key, item in node.items()}
        if isinstance(node, (tuple, list)):
            values = [clone(item) for item in node]
            return tuple(values) if isinstance(node, tuple) else values
        return node

    return clone(value)
