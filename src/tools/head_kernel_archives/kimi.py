#!/usr/bin/env python3
"""Extract Kimi structure only: shapes, aliases, paging/routing integers and zero masks.

No floating-point activation, expert-weight, routing-weight or output values are
serialized. The original archives are verified and opened read-only by the
transport wrapper; this module never imports task or candidate code.
"""
from __future__ import annotations
import base64
import json
import hashlib
import zlib

MAX_STRUCTURAL_ELEMENTS = 4_000_000


def packed_bytes(tensor, torch):
    # The restricted metadata environment intentionally has no NumPy.
    raw = bytes(tensor.detach().cpu().contiguous().clone().view(torch.uint8).untyped_storage())
    return {"encoding": "zlib+base64", "raw_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def tensor_description(tensor, groups):
    storage = tensor.untyped_storage()
    group = groups.setdefault((storage.data_ptr(), storage.nbytes()), f"storage_{len(groups)}")
    return {"tensor": True, "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
            "storage_bytes": storage.nbytes(), "storage_group": group}


def primitive(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [primitive(item) for item in value]}
    if isinstance(value, dict):
        return {key: primitive(item) for key, item in value.items()}
    raise TypeError(f"non-JSON captured scalar: {type(value).__name__}")


def output_contract(value, torch):
    if torch.is_tensor(value):
        out = tensor_description(value, {})
        out.pop("storage_group")
        out.pop("storage_bytes")
        return out
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [output_contract(item, torch) for item in value]}
    if isinstance(value, dict):
        return {key: output_contract(item, torch) for key, item in value.items()}
    return primitive(value)


def extract_attention(blob, meta, torch):
    records = blob["records"]
    expected = [case for case in meta["cases"] if not case.get("timing", True)]
    if len(records) != len(expected):
        raise ValueError("attention archive does not contain every captured correctness case")
    result = []
    for index, record in enumerate(records):
        groups = {}

        def encode(value, numeric=False):
            if torch.is_tensor(value):
                desc = tensor_description(value, groups)
                if numeric:
                    if not value.is_floating_point():
                        raise TypeError("numeric attention operand is not floating point")
                    desc["recipe"] = "normal_std_0.1"
                else:
                    if value.is_floating_point() or value.numel() > MAX_STRUCTURAL_ELEMENTS:
                        raise ValueError("unexpected structural attention operand")
                    desc.update(recipe="captured_integer", payload=packed_bytes(value, torch))
                return desc
            if isinstance(value, dict) and value.get("__tensor__"):
                desc = encode(value["data"], numeric=numeric)
                for key in ("shape", "stride", "dtype", "storage_offset"):
                    if key in value and value[key] != desc[key]:
                        # These older Kimi captures store the physical tensor directly.
                        raise ValueError(f"attention snapshot view disagrees with data: {key}")
                return desc
            return primitive(value)

        pos = []
        for slot, value in enumerate(record["pos"]):
            if isinstance(value, dict) and "__slot__" in value:
                pos.append(dict(value))
            else:
                pos.append(encode(value, numeric=slot == 0))
        kwargs = {name: encode(value, numeric=False) for name, value in record["kw"].items()}
        ref = record["ref"]
        item = {"sig": record["sig"], "case_id": expected[index]["sig"],
                "regime": record.get("regime", "decode"), "pos": pos, "kw": kwargs,
                "k": encode(record["k"], numeric=True),
                "kv_indices": encode(record["kv_indices"]),
                "v_is_slice": bool(record["v_is_slice"]), "v_head_dim": int(record["v_head_dim"]),
                "output_contract": output_contract(ref, torch)}
        if not item["v_is_slice"]:
            item["v"] = encode(record["v"], numeric=True)
        q = pos[0]
        if q["shape"][0] != int(expected[index]["bs"]):
            raise ValueError("captured attention batch disagrees with metadata")
        if item["kv_indices"]["shape"] != [int(expected[index]["bs"]) * int(expected[index]["ctx_per_seq"])]:
            raise ValueError("captured attention page count disagrees with metadata")
        result.append(item)
    return result


def extract_moe(blob, meta, torch):
    records = blob["cases"]
    expected = meta["case_specs"]
    if [row["spec"]["sig"] for row in records] != [row["sig"] for row in expected]:
        raise ValueError("MoE archive does not preserve the complete case ordering")
    result = []
    for record, spec in zip(records, expected):
        if record["spec"] != spec:
            raise ValueError("MoE archive case specification differs from metadata")
        groups = {}
        routing = {}
        for name, value in record["routing"].items():
            if not torch.is_tensor(value) or value.numel() > MAX_STRUCTURAL_ELEMENTS:
                raise ValueError(f"unexpected MoE routing field {name}")
            desc = tensor_description(value, groups)
            if name == "sorted_weights":
                if not value.is_floating_point():
                    raise TypeError("routing weights must be floating point")
                if not torch.isfinite(value).all():
                    raise ValueError("nonfinite captured routing weights need an explicit policy")
                desc.update(recipe="generated_routing_weights", topk=int(meta["geometry"]["topk"]),
                            zero_mask=packed_bytes((value == 0).to(torch.uint8), torch))
            else:
                if value.is_floating_point():
                    raise TypeError(f"structural routing index is floating point: {name}")
                desc.update(recipe="captured_integer", payload=packed_bytes(value, torch))
            routing[name] = desc
        if not {"sorted_token_ids", "sorted_expert_ids", "num_valid_ids"}.issubset(routing):
            raise ValueError("incomplete captured expert routing")
        result.append({"sig": spec["sig"], "spec": primitive(spec), "routing": routing,
                       "output_contract": output_contract(record["ref"], torch)})
    return result


def extract_blob(blob, meta, task_name, torch):
    kind = "attention" if task_name == "fwd_grouped_kernel_stage1" else "moe"
    records = (extract_attention(blob, meta, torch) if kind == "attention"
               else extract_moe(blob, meta, torch))
    return {"schema_version": 1, "kind": kind, "task": task_name,
            "input_policy": "generated_numeric_values_at_captured_structure",
            "source_reference_sha256": meta["reference_io_sha256"],
            "case_count": len(records), "records": records,
            "contains_numeric_input_values": False, "contains_reference_output_values": False}
