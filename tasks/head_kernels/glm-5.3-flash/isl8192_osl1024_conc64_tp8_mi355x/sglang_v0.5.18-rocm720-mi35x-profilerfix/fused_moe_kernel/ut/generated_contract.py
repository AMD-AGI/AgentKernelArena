"""Protected deterministic GLM input recipes and parent-only wire comparison."""
from __future__ import annotations
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(ut):
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())
    settings = meta["generated_inputs"]
    path = ut / settings["contract_file"]
    if digest(path) != settings["contract_sha256"]:
        raise RuntimeError("compact GLM structure checksum mismatch")
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1
            or value.get("source_reference_sha256") != meta["archival_capture"]["reference_io_sha256"]
            or value.get("correctness_case_count") != meta["num_cases"]
            or value.get("correctness_case_ids") != settings["captured_case_ids"]):
        raise RuntimeError("compact GLM contract changed the captured case identities")
    return value


def dtype(torch, name):
    value = getattr(torch, name.removeprefix("torch."), None)
    if value is None or not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {name!r}")
    return value


def build_blob(compact, seed, torch, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    storages = {}
    initialized = set()

    def build(node):
        if isinstance(node, dict) and node.get("tensor"):
            if node["recipe"] == "runtime_reference":
                return dict(node)  # Metadata only. No expected numerical answers in a worker input.
            dt = dtype(torch, node["dtype"])
            size = int(node["storage_nbytes"])
            group = node["storage_group"]
            if size < 0 or size > 2**34:
                raise ValueError("invalid captured storage extent")
            if group not in storages:
                storages[group] = torch.empty(size, dtype=torch.uint8, device=device).untyped_storage()
            storage = storages[group]
            if storage.nbytes() != size:
                raise ValueError("inconsistent aliased storage extent")
            itemsize = torch.empty((), dtype=dt).element_size()
            raw = torch.empty(0, dtype=dt, device=device)
            raw.set_(storage, 0, (size // itemsize,), (1,))
            if group not in initialized:
                recipe = node["recipe"]
                if recipe == "finite_fp8_weight":
                    # Generate representable finite E4M3 weights in bounded chunks.
                    for start in range(0, raw.numel(), 1 << 20):
                        values = torch.empty(min(1 << 20, raw.numel()-start), dtype=torch.float32, device=device)
                        values.normal_(0, 32, generator=generator).clamp_(-448, 448)
                        raw[start:start+values.numel()].copy_(values.to(dt))
                elif recipe == "positive_block_scale":
                    raw.uniform_(2**-8, 2**-6, generator=generator)
                elif recipe == "generated_numeric":
                    if compact["task"] == "elementwise_copy_cluster":
                        raw.uniform_(0.001, 0.051, generator=generator)
                    else:
                        raw.normal_(0, float(compact["activation_std"]), generator=generator)
                elif recipe == "captured_routing":
                    raw.zero_()
                else:
                    raise ValueError(f"unknown GLM input recipe {recipe!r}")
                initialized.add(group)
            value = raw.as_strided(tuple(node["shape"]), tuple(node["stride"]), int(node["storage_offset"]))
            if node["recipe"] == "captured_routing":
                expected = math.prod(node["shape"]) * itemsize
                if expected > 16 << 20:
                    raise ValueError("routing tensor exceeds compact contract bound")
                compressed = base64.b64decode(node["data"], validate=True)
                decoder = zlib.decompressobj()
                data = decoder.decompress(compressed, expected + 1)
                if len(data) != expected or not decoder.eof or decoder.unused_data:
                    raise ValueError("invalid compact routing byte count")
                source = torch.frombuffer(bytearray(data), dtype=dt).reshape(node["shape"])
                value.copy_(source.to(device))
            for name, attr in node["tensor_attrs"].items():
                setattr(value, name, attr)
            value.requires_grad_(bool(node.get("requires_grad", False)))
            return value
        if isinstance(node, dict) and "sequence" in node:
            values = [build(x) for x in node["items"]]
            return tuple(values) if node["sequence"] == "tuple" else values
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return node

    return build(compact["blob"])


def prepare_cases(cases, ut, seed, torch, device="cuda"):
    compact = load_contract(ut)
    cases._C.clear()
    cases._C["blob"] = build_blob(compact, seed, torch, device)
    if compact["task"] == "fused_moe_kernel":
        cases._C["act_std"] = compact["activation_std"]
    return compact


def recorded_cases(cases, compact):
    blob = cases._blob()
    rows = []
    if compact["task"] == "elementwise_copy_cluster":
        for index, record in enumerate(blob["records"]):
            kwargs = record.get("kwargs") or {}
            scale = kwargs["scale"] if "scale" in kwargs else record["args"][0]
            regime = record.get("regime") or ""
            rows.append({"sig": f"oracle{index}_m{scale.shape[0]}_{regime or 'na'}", "regime": regime,
                         "args": {"scale": scale}, "output_contract": record["output"]})
    else:
        for record in blob["cases"]:
            regime = record.get("regime") or ""
            rows.append({"sig": f"oracle_m{record['m']}_{regime or 'na'}", "regime": regime,
                         "args": cases._build_args(record["hidden_states"], record["topk_weights"], record["topk_ids"]),
                         "output_contract": record["output"]})
    return rows


def reference_copy(args):
    scale = args["scale"]
    return scale.t().contiguous().t() if scale.dim() == 2 else scale


def alias_contract(output, inputs, torch):
    left = tensor_leaves(output, torch)
    right = tensor_leaves(inputs, torch)
    return [[i, j] for i, value in enumerate(left) for j, source in enumerate(right)
            if value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()]


def encode_output(value, torch):
    """Use typed byte JSON, not pickle, across the untrusted output boundary."""
    if type(value) is torch.Tensor:
        if value.__dict__:
            raise TypeError("GLM output tensor has undeclared attributes or shadowed methods")
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
    """Use mixed floating tolerance and exact integer/bool equality for every output."""
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(observed, dict) or not observed.get("tensor"):
            return False
        if any(observed[k] != expected[k] for k in ("shape", "stride", "dtype", "storage_offset", "device")):
            return False
        left, right = decode_output(observed, torch), decode_output(expected, torch)
        # Discrete outputs are exact; floating outputs retain the mixed tolerance.
        if not left.is_floating_point() or not right.is_floating_point():
            return left.dtype == right.dtype and bool(torch.equal(left, right))
        left, right = left.float(), right.float()
        atol = float(tol) * right.pow(2).mean().sqrt().clamp_min(1e-6)
        return bool(((left - right).abs() <= atol + float(tol) * right.abs()).all())
    if isinstance(expected, dict) and "sequence" in expected:
        return (isinstance(observed, dict) and observed.get("sequence") == expected["sequence"]
                and len(observed.get("items", [])) == len(expected["items"])
                and all(compare_output(a, b, tol, torch) for a, b in zip(observed["items"], expected["items"])))
    return type(observed) is type(expected) and observed == expected


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


def require_inputs_unchanged(saved, torch, writable=()):
    allowed = {id(value) for value in writable}
    for value, signature, copy in saved:
        current = (list(value.shape), list(value.stride()), str(value.dtype),
                   value.storage_offset(), value.data_ptr(), dict(value.__dict__))
        if current != signature:
            raise RuntimeError("candidate changed a GLM input tensor contract")
        if id(value) not in allowed and not torch.equal(value.contiguous().view(torch.uint8), copy.contiguous().view(torch.uint8)):
            raise RuntimeError("candidate changed a read-only GLM input or its tensor contract")


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


class ExactTimingUnavailable(RuntimeError):
    """An observed required workload lacks a retained structural call record."""


def require_complete_timing(meta):
    scoring = meta['generated_inputs'].get('workload_scoring', {})
    if scoring.get('enabled') is False:
        raise ExactTimingUnavailable(scoring['reason'])
    if not meta['generated_inputs']['profiles'].get('timed'):
        raise ExactTimingUnavailable('no qualified scored timing cases are declared')


def semantic_moe_call_cases(blob, build_args, meta):
    """Unscored initialization probes with retained routing, never served-workload cases."""
    rows = []
    for spec in meta['generated_inputs']['semantic_call_probes']:
        if spec['status'] != 'retained_initialization_postcall_structure':
            continue
        record = blob['cases'][spec['record_index']]
        if (record['m'] != spec['m'] or record['regime'] != spec['regime']
                or record['sig'].split('|')[7] != 'True'):
            raise ValueError('retained call does not prove its declared in-place timing contract')
        args = build_args(record['hidden_states'], record['topk_weights'], record['topk_ids'])
        args['inplace'] = True  # Raw captured argument; the old oracle explicitly forced False.
        rows.append({'sig': spec['case_id'], 'regime': spec['regime'], 'm': spec['m'],
                     'args': args, 'mutable_inputs': ['hidden_states'],
                     'routing_source': spec['source_case_id'], 'routing_resampled': False, 'scored': False, 'case_role': 'semantic_initialization_probe'})
    return rows


def scored_moe_timing_cases(blob, build_args, meta):
    require_complete_timing(meta)
    return semantic_moe_call_cases(blob, build_args, meta)
