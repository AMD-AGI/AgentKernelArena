"""Extract compact public GLM input structure; never serialize golden values."""
from __future__ import annotations
import ast
import base64
import zlib


def extract_blob(blob, meta, task, torch):
    groups = {}
    structural = []

    def scalar(value):
        if isinstance(value, dict) and set(value) == {"__repr__"}:
            return ast.literal_eval(value["__repr__"])
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        raise TypeError(f"unsupported captured scalar {type(value).__name__}")

    def encode(value, path=(), output=False):
        if isinstance(value, dict) and value.get("__tensor__"):
            tensor, envelope = value["data"], value
        elif torch.is_tensor(value):
            tensor, envelope = value, {}
        else:
            tensor = None
        if tensor is not None:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            group = groups.setdefault(key, f"storage_{len(groups)}")
            shape = list(envelope.get("shape", tensor.shape))
            stride = list(envelope.get("stride", tensor.stride()))
            offset = int(envelope.get("storage_offset", tensor.storage_offset()))
            nbytes = storage.nbytes()
            desc = {"tensor": True, "shape": shape, "stride": stride,
                    "dtype": str(envelope.get("dtype", tensor.dtype)), "storage_offset": offset,
                    "storage_group": group, "storage_nbytes": nbytes,
                    "tensor_attrs": envelope.get("tensor_attrs") or envelope.get("attrs") or dict(tensor.__dict__),
                    "requires_grad": bool(tensor.requires_grad)}
            name = next((str(x) for x in reversed(path) if not isinstance(x, int)), "")
            if output:
                desc["recipe"] = "runtime_reference"
            elif name in ("topk_ids", "topk_weights"):
                if tensor.numel() > 2_000_000:
                    raise ValueError("structural routing tensor exceeds extraction bound")
                data = bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
                desc.update(recipe="captured_routing", encoding="zlib-base64-le",
                            data=base64.b64encode(zlib.compress(data, 9)).decode("ascii"))
                structural.append({"path": "/" + "/".join(map(str, path)), "bytes": len(data)})
            elif name in ("w1_scale", "w2_scale"):
                desc["recipe"] = "positive_block_scale"
            elif name in ("w1", "w2"):
                desc["recipe"] = "finite_fp8_weight"
            elif name in ("hidden_states", "scale") or task == "elementwise_copy_cluster":
                desc["recipe"] = "generated_numeric"
            else:
                raise ValueError(f"unclassified numerical tensor: {path}")
            return desc
        if isinstance(value, dict) and "__repr__" not in value:
            return {str(k): encode(v, path + (k,), output or k in ("output", "ref")) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {"sequence": type(value).__name__, "items": [encode(v, path + (i,), output) for i, v in enumerate(value)]}
        return scalar(value)

    if task == "elementwise_copy_cluster":
        records = blob["records"]
        if len(records) != meta["num_cases"]:
            raise ValueError("copy record count differs from metadata")
        selected = {"records": records, "shared": blob.get("shared", {})}
        ids = [r["sig"] for r in records]
    elif task == "fused_moe_kernel":
        records = blob["cases"]
        if [(r["m"], r["regime"]) for r in records] != [(r["m"], r["regime"]) for r in meta["cases"]]:
            raise ValueError("MoE captured case order/geometry changed")
        selected = {k: blob[k] for k in ("shared", "static", "cases", "routing_pool")}
        ids = [r["sig"] for r in meta["cases"]]
    else:
        raise ValueError(task)
    encoded = encode(selected)
    result = {"schema_version": 1, "task": task, "input_policy": "generated_numeric_preserved_capture_structure",
              "source_reference_sha256": meta["reference_io_sha256"], "correctness_case_ids": ids,
              "correctness_case_count": len(records), "blob": encoded, "structural_values": structural,
              "generator_version": 1}
    if task == "fused_moe_kernel":
        result["activation_std"] = float(records[0]["hidden_states"].float().std().clamp_min(1e-3))
    return result
