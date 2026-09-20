"""Extract compact DeepSeek tensor contracts from verified CPU-only archives."""

from __future__ import annotations
import base64
import json
import math
import zlib


def packed_tensor(tensor, torch):
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    compressor = zlib.compressobj(9)
    parts = []
    for offset in range(0, flat.numel(), 1 << 20):
        parts.append(
            compressor.compress(bytes(flat[offset : offset + (1 << 20)].tolist()))
        )
    parts.append(compressor.flush())
    return {
        "codec": "zlib-base64",
        "bytes": flat.numel(),
        "data": base64.b64encode(b"".join(parts)).decode("ascii"),
    }


def extract_blob(blob, meta, names, torch):
    records = blob["records"]
    if len(records) != int(meta["num_cases"]):
        raise ValueError("record count differs from task metadata")
    tensors = {}
    storage_ids = {}
    output = []
    shared = blob.get("shared") or {}

    def resolve(node):
        while isinstance(node, dict) and set(node) == {"__shared__"}:
            node = shared[node["__shared__"]]
        return node

    def encode(node, name, record_index, result=False):
        node = resolve(node)
        if torch.is_tensor(node):
            node = {"__tensor__": True, "data": node}
        if isinstance(node, dict) and node.get("__dsa_kv__"):
            rows = node["rows"]
            shape = list(node["shape"])
            return {
                "dsa_kv": True,
                "shape": shape,
                "dtype": str(node["dtype"]),
                "stride0": int(node.get("stride0") or math.prod(shape[1:])),
                "rows_shape": list(rows.shape),
                "rows_dtype": str(rows.dtype),
                "rows_idx": encode(node["rows_idx"], "rows_idx", record_index),
                "recipe": "dsa_fp8_nope448_bf16_rope64_e8m0_scales7",
                "block_payload_bytes": shape[1] * 576,
                "block_scale_bytes": shape[1] * 8,
                "source_node_fields": sorted(node),
            }
        if isinstance(node, dict) and node.get("__tensor__"):
            value = node["data"]
            logical_dtype = str(node.get("dtype", value.dtype))
            storage = value.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            group = storage_ids.setdefault(
                storage_key, "storage_" + str(len(storage_ids))
            )
            desc = {
                "tensor": True,
                "shape": list(node.get("shape", value.shape)),
                "stride": list(node.get("stride", value.stride())),
                "dtype": logical_dtype,
                "storage_offset": int(
                    node.get("storage_offset", value.storage_offset())
                ),
                "storage_group": group,
                "storage_bytes": storage.nbytes(),
                "tensor_attrs": node.get("tensor_attrs") or node.get("attrs") or {},
                "bitcast": bool(node.get("bitcast")),
                "capture_data_dtype": str(value.dtype),
            }
            if result:
                desc["recipe"] = "fresh_frozen_baseline_output"
                return desc
            if name in {"q", "a", "inter_states", "hidden_states"}:
                desc["recipe"] = "generated_activation"
            elif name in {"w1", "w2"}:
                desc["recipe"] = "generated_mxfp4_weight"
            elif "scale" in name and value.numel() > 1:
                desc["recipe"] = "generated_block_scale"
            elif name in {"attn_sink", "sink"}:
                desc["recipe"] = "generated_sink"
            elif name == "out":
                desc["recipe"] = "zero_output"
            else:
                if value.numel() > 16_000_000:
                    raise ValueError("unexpected large structural tensor: " + name)
                desc["recipe"] = "captured_structure"
                desc["payload"] = packed_tensor(value, torch)
            return desc
        if isinstance(node, dict):
            if "__repr__" in node:
                return {"repr": node["__repr__"]}
            return {
                key: encode(value, key, record_index, result)
                for key, value in node.items()
            }
        if isinstance(node, (list, tuple)):
            return {
                "sequence": type(node).__name__,
                "items": [encode(value, name, record_index, result) for value in node],
            }
        if node is None or type(node) in (str, int, float, bool):
            return node
        raise TypeError(type(node).__name__)

    for index, record in enumerate(records):
        args = record.get("args", [])
        kwargs = record.get("kwargs", {})
        item = {
            "sig": record["sig"],
            "regime": record.get("regime", ""),
            "args": [encode(value, names[i], index) for i, value in enumerate(args)],
            "kwargs": {key: encode(value, key, index) for key, value in kwargs.items()},
            "output_contract": encode(record["output"], "output", index, True),
        }
        if "case_specs" in meta:
            out = record["output"][0]
            out = resolve(out)
            if isinstance(out, dict):
                out = out["data"]
            mask = torch.isnan(out.float()).reshape(out.shape[0], -1)
            all_rows = mask.all(dim=1)
            any_rows = mask.any(dim=1)
            partial = (any_rows & ~all_rows).nonzero().reshape(-1)
            item["undefined_rows"] = all_rows.nonzero().reshape(-1).tolist()
            item["undefined_partial_rows"] = partial.tolist()
            item["undefined_partial_payload"] = packed_tensor(mask[partial], torch)
            item["undefined_shape"] = list(out.shape)
            lse = resolve(record["output"][1])
            lse = lse["data"] if isinstance(lse, dict) else lse
            lse_mask = torch.isnan(lse)
            item["undefined_lse"] = {
                "shape": list(lse.shape),
                "undefined_count": int(lse_mask.sum()),
                "mask": packed_tensor(lse_mask, torch),
            }
        output.append(item)
    return {
        "schema_version": 1,
        "task_kind": (
            "dsa"
            if "case_specs" in meta
            else ("moe2" if meta.get("inplace_out_arg") == 6 else "moe1")
        ),
        "input_policy": "generated_numeric_values_at_captured_structural_contract",
        "source_reference_sha256": meta["reference_io_sha256"],
        "records": output,
    }
