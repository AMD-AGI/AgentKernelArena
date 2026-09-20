"""Generate numeric Qwen operands at frozen layouts and real structural values."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def load_contract(ut):
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())
    declaration = meta["generated_inputs"]
    path = ut / declaration["contract_file"]
    if not path.resolve().is_relative_to(ut.resolve()) or path.is_symlink():
        raise ValueError("generated contract must be a task-local regular file")
    if digest(path) != declaration["contract_sha256"]:
        raise ValueError("generated contract SHA-256 mismatch")
    data = json.loads(path.read_text())
    if (data.get("schema_version") != 1 or data.get("task_id") != meta["task_id"]
            or data.get("correctness_case_count") != meta["num_cases"]
            or len(data.get("records", [])) != meta["num_cases"]
            or data.get("source_reference_sha256") != meta["archival_capture"]["reference_io_sha256"]):
        raise ValueError("generated contract differs from the complete captured task identity")
    return data


def dtype(torch, name):
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported captured dtype: {name}")
    return value


def decode_integers(encoded):
    if encoded.get("encoding") != "affine_runs" or type(encoded.get("count")) is not int:
        raise ValueError("unknown structural integer encoding")
    count = encoded["count"]
    if not 0 <= count <= 2_000_000:
        raise ValueError("invalid structural integer count")
    values = []
    for row in encoded.get("runs", []):
        if len(row) != 3 or any(type(x) is not int for x in row) or row[2] <= 0:
            raise ValueError("invalid structural integer run")
        start, step, length = row
        if len(values) + length > count:
            raise ValueError("structural integer run exceeds declared count")
        values.extend(start + step * i for i in range(length))
    if len(values) != count:
        raise ValueError("structural integer count mismatch")
    return values


def tensor_signature(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "storage_offset": value.storage_offset(),
            "device": str(value.device)}


def build_blob(data, seed, torch, device="cuda"):
    """Build each backing storage once; numeric generation never reads an archive."""
    storages, shared = {}, {}

    def build(node):
        if isinstance(node, dict) and set(node) == {"__shared__"}:
            name = node["__shared__"]
            if name not in shared:
                shared[name] = build(data["shared"][name])
            return shared[name]
        if isinstance(node, dict) and node.get("tensor"):
            dt = dtype(torch, node["dtype"])
            shape, stride, offset = node["shape"], node["stride"], node["storage_offset"]
            nbytes = node["storage_nbytes"]
            if (len(shape) != len(stride) or any(type(x) is not int or x < 0 for x in [*shape, *stride, offset])
                    or type(nbytes) is not int or nbytes < 0):
                raise ValueError("invalid captured tensor layout")
            element_size = torch.empty((), dtype=dt).element_size()
            extent = offset + sum(max(n - 1, 0) * s for n, s in zip(shape, stride)) + (1 if math.prod(shape) else 0)
            if extent * element_size > nbytes:
                raise ValueError("captured backing storage is smaller than its view")
            group = node["storage_group"]
            if group not in storages:
                storages[group] = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            backing = storages[group]
            if backing.numel() != nbytes:
                raise ValueError("aliased tensor descriptors disagree on backing extent")
            value = torch.empty(0, dtype=dt, device=device)
            value.set_(backing.untyped_storage(), offset, tuple(shape), tuple(stride))
            if str(device) == "meta":
                for name, attribute in node.get("tensor_attrs", {}).items():
                    setattr(value, name, attribute)
                return value
            key = f"{int(seed)}:{group}:{node['name']}".encode()
            generator = torch.Generator(device=device).manual_seed(
                int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1))
            recipe = node["recipe"]
            if recipe == "captured_structure":
                values = decode_integers(node["values"])
                if len(values) != value.numel():
                    raise ValueError("structural values differ from the tensor shape")
                value.copy_(torch.tensor(values, dtype=dt, device=device).reshape(shape))
            elif recipe == "packed_fp4":
                value.view(torch.uint8).random_(0, 256, generator=generator)
            elif recipe == "positive_e8m0_scale":
                # E8M0 bytes 120..124 are finite positive powers of two.
                value.view(torch.uint8).random_(120, 125, generator=generator)
            elif recipe == "normalized_route_weights":
                temp = torch.rand(shape, dtype=torch.float32, device=device, generator=generator) + 0.125
                value.copy_(temp / temp.sum(dim=-1, keepdim=True))
            elif recipe == "log_positive_decay":
                temp = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
                value.copy_((1 + 15 * temp).log())
            elif recipe == "inverse_softplus_dt":
                temp = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
                value.copy_((0.001 + 0.099 * temp).expm1().log())
            elif recipe == "unit_scale":
                value.fill_(1)
            elif recipe == "zero_workspace":
                value.zero_()
            elif recipe == "finite_normal":
                scale = 0.05 if node["name"] == "initial_state" else 0.25
                if str(dt).startswith("torch.float8_"):
                    temp = torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * scale
                    value.copy_(temp.to(dt))
                else:
                    value.normal_(std=scale, generator=generator)
            else:
                raise ValueError(f"unsupported numeric recipe: {recipe}")
            for name, attribute in node.get("tensor_attrs", {}).items():
                setattr(value, name, attribute)
            return value
        if isinstance(node, dict) and "sequence" in node:
            values = [build(item) for item in node["items"]]
            return tuple(values) if node["sequence"] == "tuple" else values
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    for key in data["shared"]:
        if key not in shared:
            shared[key] = build(data["shared"][key])
    records = []
    for record in data["records"]:
        records.append({"sig": record["source_sig"], "regime": record["regime"],
                        "args": build(record["args"]), "kwargs": build(record["kwargs"]),
                        "kwargs_before": build(record["kwargs_before"]), "output": None})
    return {"shared": shared, "records": records}


def tensor_leaves(value, torch):
    if type(value) is torch.Tensor:
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in tensor_leaves(item, torch)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in tensor_leaves(item, torch)]
    return []


def clone_tree(value, torch):
    storages = {}
    def clone(node):
        if type(node) is torch.Tensor:
            storage = node.untyped_storage()
            key = (str(node.device), storage.data_ptr(), storage.nbytes())
            if key not in storages:
                raw = torch.empty(0, dtype=torch.uint8, device=node.device)
                raw.set_(storage, 0, (storage.nbytes(),), (1,))
                storages[key] = raw.clone().untyped_storage()
            value = torch.empty(0, dtype=node.dtype, device=node.device)
            value.set_(storages[key], node.storage_offset(), node.shape, node.stride())
            value.__dict__.update(node.__dict__)
            return value
        if isinstance(node, dict):
            return {key: clone(item) for key, item in node.items()}
        if isinstance(node, (list, tuple)):
            values = [clone(item) for item in node]
            return tuple(values) if isinstance(node, tuple) else values
        return node
    return clone(value)


def encode_output(value, torch):
    if type(value) is torch.Tensor:
        cpu = value.detach().cpu().contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        return {"tensor": True, **tensor_signature(value), "data": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [encode_output(item, torch) for item in value]}
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError("worker returned an unsupported output tree")


def decode_output(value, torch):
    dt = dtype(torch, value["dtype"])
    shape = value["shape"]
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid worker output shape")
    expected = math.prod(shape) * torch.empty((), dtype=dt).element_size()
    if expected > 256 << 20:
        raise ValueError("worker output exceeds its declared contract bound")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != expected:
        raise ValueError("worker output byte count mismatch")
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape)


def compare_output(observed, expected, tol, torch, rms=None):
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(observed, dict) or not observed.get("tensor"):
            return False
        if any(observed.get(key) != expected[key] for key in ("shape", "stride", "dtype", "storage_offset", "device")):
            return False
        left, right = decode_output(observed, torch), decode_output(expected, torch)
        if not right.is_floating_point():
            return bool(torch.equal(left, right))
        left, right = left.float(), right.float()
        floor = right.pow(2).mean().sqrt().clamp_min(1e-6) if rms is None else max(float(rms), 1e-6)
        return bool(((left - right).abs() <= float(tol) * floor + float(tol) * right.abs()).all())
    if isinstance(expected, dict) and expected.get("state"):
        if (not isinstance(observed, dict) or not observed.get("state")
                or observed.get("signature") != expected["signature"]
                or observed.get("indices") != expected["indices"]
                or observed.get("unselected_rows_unchanged") is not True):
            return False
        return compare_output(observed["selected_rows"], expected["selected_rows"], tol, torch,
                              rms=expected["full_state_rms"])
    if isinstance(expected, dict) and "sequence" in expected:
        return (isinstance(observed, dict) and observed.get("sequence") == expected["sequence"]
                and len(observed.get("items", [])) == len(expected["items"])
                and all(compare_output(a, b, tol, torch) for a, b in zip(observed["items"], expected["items"])))
    return type(observed) is type(expected) and observed == expected


def require_output_contract(encoded, expected):
    """Both workers must retain the captured output layout, even if both change."""
    if isinstance(expected, dict) and expected.get("tensor"):
        signature = encoded.get("signature") if isinstance(encoded, dict) and encoded.get("state") else encoded
        if not isinstance(signature, dict) or any(signature.get(key) != expected[key]
                                                  for key in ("shape", "stride", "dtype", "storage_offset")):
            raise RuntimeError("worker changed the captured output tensor contract")
    elif isinstance(expected, dict) and "sequence" in expected:
        if (not isinstance(encoded, dict) or encoded.get("sequence") != expected["sequence"]
                or len(encoded.get("items", [])) != len(expected["items"])):
            raise RuntimeError("worker changed the captured output tuple contract")
        for value, spec in zip(encoded["items"], expected["items"]):
            require_output_contract(value, spec)
    elif type(encoded) is not type(expected) or encoded != expected:
        raise RuntimeError("worker changed the captured non-tensor output contract")
