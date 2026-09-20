"""Protected Kimi generated-input contracts; no persistent numerical oracles."""
from __future__ import annotations
import ast
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import zlib


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(ut):
    ut = Path(ut)
    metadata = json.loads((ut / "meta.json").read_text())
    declaration = metadata["generated_inputs"]
    path = ut / declaration["contract_file"]
    if digest(path) != declaration["contract_sha256"]:
        raise RuntimeError("generated structural contract hash mismatch")
    value = json.loads(path.read_text())
    if (value["schema_version"] != 1 or value["case_count"] != len(value["records"])
            or value["contains_numeric_input_values"] or value["contains_reference_output_values"]):
        raise RuntimeError("invalid structural-only contract")
    return value


def dtype(torch, value):
    result = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(result, torch.dtype):
        raise TypeError(f"unsupported captured dtype: {value}")
    return result


def payload_bytes(payload):
    compressed = base64.b64decode(payload["data"], validate=True)
    limit = int(payload["raw_bytes"])
    if limit < 0 or limit > 64 << 20:
        raise ValueError("invalid structural payload bound")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, limit + 1)
    if len(raw) != limit or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("structural payload length differs")
    if hashlib.sha256(raw).hexdigest() != payload["sha256"]:
        raise ValueError("structural tensor checksum differs")
    return raw


def restore_primitive(value):
    if isinstance(value, dict) and "sequence" in value:
        items = [restore_primitive(item) for item in value["items"]]
        return tuple(items) if value["sequence"] == "tuple" else items
    if isinstance(value, dict):
        return {key: restore_primitive(item) for key, item in value.items()}
    return value


def build_tensors(torch, device, seed):
    """Return a builder retaining captured storage aliases, offsets and strides."""
    storages = {}
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def build(desc):
        if not isinstance(desc, dict) or not desc.get("tensor"):
            return restore_primitive(desc)
        dt = dtype(torch, desc["dtype"])
        group = desc["storage_group"]
        if group not in storages:
            count = int(desc["storage_bytes"])
            itemsize = torch.empty((), dtype=dt).element_size()
            if count % itemsize:
                raise ValueError("captured storage is not dtype-aligned")
            storage = torch.zeros(count, dtype=torch.uint8, device=device)
            if desc["recipe"] == "normal_std_0.1":
                normal = torch.randn(count // itemsize, generator=generator, dtype=torch.float32,
                                     device=device).mul_(0.1).to(dt)
                storage.copy_(normal.view(torch.uint8))
            storages[group] = storage.untyped_storage()
        tensor = torch.empty(0, dtype=dt, device=device)
        tensor.set_(storages[group], int(desc["storage_offset"]), tuple(desc["shape"]), tuple(desc["stride"]))
        recipe = desc["recipe"]
        if recipe == "captured_integer":
            raw = payload_bytes(desc["payload"])
            if len(raw) != tensor.numel() * tensor.element_size():
                raise ValueError("structural tensor byte count differs from shape")
            tensor.copy_(torch.frombuffer(bytearray(raw), dtype=dt).reshape(desc["shape"]).to(device))
        elif recipe == "generated_routing_weights":
            tensor.copy_((torch.rand(desc["shape"], generator=generator, device=device) + 0.5)
                         .div_(int(desc["topk"])).to(dt))
            mask = torch.frombuffer(bytearray(payload_bytes(desc["zero_mask"])), dtype=torch.uint8)
            tensor.masked_fill_(mask.reshape(desc["shape"]).to(device).bool(), 0)
        elif recipe != "normal_std_0.1":
            raise ValueError(f"unknown tensor recipe {recipe}")
        return tensor
    return build


def attention_record(record, seed, torch, device):
    build = build_tensors(torch, device, seed)
    k = build(record["k"])
    v = k[:, :, :record["v_head_dim"]] if record["v_is_slice"] else build(record["v"])
    slots = {"k": k, "v": v, "kv_indices": build(record["kv_indices"])}
    positional, output_slots = [], {}
    for index, node in enumerate(record["pos"]):
        if isinstance(node, dict) and "__slot__" in node:
            name = node["__slot__"]
            if name in ("att_out", "att_lse"):
                desc = record["output_contract"][name]
                output_slots[index] = (tuple(desc["shape"]), dtype(torch, desc["dtype"]))
                positional.append(None)
            else:
                positional.append(slots[name])
        else:
            positional.append(build(node))
    return {"pos": positional, "kw": {key: build(value) for key, value in record["kw"].items()},
            "out_slots": output_slots}


def load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted helper alias names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _NoCachedReferences(ast.NodeTransformer):
    """Retain the original replay/allocation code, remove its in-process goldens."""
    def visit_Expr(self, node):
        if any(isinstance(part, ast.Name) and part.id == "BASELINE_FN" for part in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "ref_out":
                return None
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == "ref":
                return None
        return self.generic_visit(node)

    def visit_Delete(self, node):
        if any(isinstance(target, ast.Name) and target.id == "ref_out" for target in node.targets):
            return None
        return node

    def visit_Return(self, node):
        if (isinstance(node.value, ast.Dict)
                and any(isinstance(key, ast.Constant) and key.value == "capture_idx" for key in node.value.keys)):
            node.value.keys.append(ast.Constant("state_inputs"))
            node.value.values.append(ast.Name(id="st_args", ctx=ast.Load()))
        return self.generic_visit(node)


def legacy_definitions(ut, torch, device, compact, seed=None):
    """Reuse protected recipes without importing the archive-dependent UT module."""
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())
    h = sys.modules.get("harness_lib") or load_module("harness_lib", ut / "harness_lib.py")
    geo = meta["geometry"]
    scope = {"__file__": str(ut / "unittest.py"), "__name__": "_kimi_generated_recipes",
             "META": meta, "GEO": geo, "torch": torch, "h": h, "DEV": device,
             "REGIME": meta["regime"], "TOPK": geo.get("topk"),
             "MODEL_DIM": geo.get("model_dim"), "INTER_DIM": geo.get("inter_dim"),
             "E": geo.get("num_experts"), "MEDIAN_LAUNCHES": int(meta.get("median_launches", 1)),
             "I_Q": 0, "I_K": 1, "I_V": 2, "I_AO": 3, "I_AL": 4, "I_IPTR": 5, "I_IDX": 6, "I_NKS": 7}
    if compact["kind"] == "attention":
        wanted = {"_make_call", "_num_kv_splits", "_synth", "_online_buckets", "random_shapes"}
    else:
        wanted = {"_as", "_rand_bytes", "build_inputs", "_out_of", "_invoke", "_invoke_median",
                  "make_call", "timing_cases", "build_replay"}
        scope["ROUTING"] = {}
        for index, record in enumerate(compact["records"]):
            build = build_tensors(torch, "cpu", (0 if seed is None else seed) + index)
            scope["ROUTING"][record["sig"]] = {key: build(value) for key, value in record["routing"].items()}
    source = ast.parse((ut / "unittest.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    nodes = [_NoCachedReferences().visit(node) if node.name == "build_replay" else node for node in nodes]
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    if any(isinstance(node, ast.Name) and node.id == "BASELINE_FN" for node in ast.walk(tree)):
        raise RuntimeError("generated replay still references an in-process golden")
    exec(compile(tree, str(ut / "unittest.py"), "exec"), scope)
    if compact["kind"] == "moe" and seed is not None:
        original = scope["build_inputs"]
        indices = {item["sig"]: index for index, item in enumerate(meta["case_specs"])}
        def generated_inputs(spec, gen=None):
            generator = gen or torch.Generator(device=device).manual_seed(seed + indices[spec["sig"]])
            return original(spec, generator)
        scope["build_inputs"] = generated_inputs
    return scope


def bind(ut, meta, reference):
    """Reference workers never import an editable kernel module."""
    ut = Path(ut)
    if meta.get("entry_attr"):
        helper = load_module("_kimi_generated_package", ut / "flydsl_package.py")
        root = ut / ("baseline_src/flydsl" if reference else "kernel_src/flydsl")
        module = helper.load_moe_package(root, "_kimi_generated_baseline" if reference else "_kimi_generated_candidate",
                                        frozen_manifest=ut / "dependency_manifest.json" if reference else None)
        return getattr(module, meta["entry_attr"])
    helper = load_module("_kimi_generated_bindings", ut / "bindings.py")
    if reference:
        module = helper._load("_kimi_generated_frozen_attention", str(ut / "baseline_ref/decode_attention.py.orig"))
        return module._decode_grouped_att_m_fwd
    return helper.resolve_pair(str(ut))[1]


def tensor_leaves(value, torch):
    if type(value) is torch.Tensor:
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in tensor_leaves(item, torch)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in tensor_leaves(item, torch)]
    return []


def encode_output(value, torch):
    if type(value) is torch.Tensor:
        cpu = torch.Tensor.contiguous(torch.Tensor.cpu(torch.Tensor.detach(value)))
        raw = torch.Tensor.numpy(torch.Tensor.view(cpu, torch.uint8)).tobytes()
        return {"tensor": True, "shape": list(value.shape), "stride": list(torch.Tensor.stride(value)),
                "dtype": str(value.dtype), "device": str(value.device), "storage_offset": torch.Tensor.storage_offset(value),
                "data": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [encode_output(item, torch) for item in value]}
    if value is None or type(value) in (int, bool, float, str):
        return value
    raise TypeError("unsupported candidate output type")


def decode_output(value, torch):
    dt = dtype(torch, value["dtype"])
    shape = value["shape"]
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("invalid output shape")
    size = math.prod(shape) * torch.empty((), dtype=dt).element_size()
    if size > 2 << 30:
        raise ValueError("output exceeds the largest captured contract bound")
    if not isinstance(value["data"], str) or len(value["data"]) != 4 * ((size + 2) // 3):
        raise ValueError("encoded output length differs from its fixed shape")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != size:
        raise ValueError("output byte count differs from its fixed shape")
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape)


def compare_output(observed, expected, tol, torch):
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(observed, dict) or not observed.get("tensor"):
            return False
        if any(observed.get(key) != expected[key] for key in ("shape", "stride", "dtype", "device", "storage_offset")):
            return False
        left, right = decode_output(observed, torch), decode_output(expected, torch)
        if not right.is_floating_point():
            return torch.equal(left, right)  # integer and bool outputs are exact
        left, right = left.float(), right.float()
        atol = float(tol) * right.pow(2).mean().sqrt().clamp_min(1e-6)
        return bool(((left - right).abs() <= atol + float(tol) * right.abs()).all())
    if isinstance(expected, dict) and "sequence" in expected:
        return (isinstance(observed, dict) and observed.get("sequence") == expected["sequence"]
                and len(observed.get("items", [])) == len(expected["items"])
                and all(compare_output(a, b, tol, torch) for a, b in zip(observed["items"], expected["items"])))
    return type(observed) is type(expected) and observed == expected
