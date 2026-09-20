"""Extract only tensor layouts and real structural values from verified Qwen archives."""
from __future__ import annotations

import math


def integer_encoding(values):
    """Lossless affine runs keep large page permutations compact and inspectable."""
    runs = []
    index = 0
    while index < len(values):
        start = values[index]
        step = values[index + 1] - start if index + 1 < len(values) else 0
        end = index + 1
        while end < len(values) and values[end] == start + (end - index) * step:
            end += 1
        runs.append([start, step, end - index])
        index = end
    return {"encoding": "affine_runs", "count": len(values), "runs": runs}


def extract_blob(blob, meta, torch, names=()):
    if len(blob.get("records") or []) != int(meta["num_cases"]):
        raise ValueError("capture case count differs from metadata")
    groups = {}
    statistics = {"structural_elements": 0, "numeric_tensor_elements": 0, "factory_elements": 0}

    def encode(value, name, output=False):
        if torch.is_tensor(value):
            value = {"__tensor__": True, "data": value}
        if isinstance(value, dict) and (value.get("__tensor__") or value.get("__tensor_factory__")):
            data = value.get("data")
            factory = data is None
            shape = list(value["shape"] if factory else value.get("shape", data.shape))
            stride = list(value["stride"] if factory else value.get("stride", data.stride()))
            dtype = str(value["dtype"] if factory else value.get("dtype", data.dtype))
            offset = int(value.get("storage_offset", 0 if factory else data.storage_offset()))
            dt = getattr(torch, dtype.removeprefix("torch."))
            itemsize = torch.empty((), dtype=dt).element_size()
            extent = offset + sum(max(int(n) - 1, 0) * int(s) for n, s in zip(shape, stride)) + (1 if math.prod(shape) else 0)
            storage = None if factory else data.untyped_storage()
            key = ("factory", name) if factory else (storage.data_ptr(), storage.nbytes())
            group = groups.setdefault(key, "storage_" + str(len(groups)))
            desc = {"tensor": True, "shape": shape, "stride": stride, "dtype": dtype,
                    "storage_offset": offset, "storage_group": group,
                    "storage_nbytes": max(extent * itemsize, 0 if factory else storage.nbytes()),
                    "tensor_attrs": value.get("tensor_attrs") or value.get("attrs") or {},
                    "name": name}
            if output:
                desc["recipe"] = "runtime_reference"
            elif factory:
                desc["recipe"] = "zero_workspace"
                statistics["factory_elements"] += math.prod(shape)
            elif dtype in {"torch.int8", "torch.int16", "torch.int32", "torch.int64", "torch.bool"}:
                if data.numel() > 2_000_000:
                    raise ValueError(f"unexpected large structural tensor: {name}")
                desc["recipe"] = "captured_structure"
                desc["values"] = integer_encoding([int(x) for x in data.reshape(-1).tolist()])
                statistics["structural_elements"] += data.numel()
            else:
                leaf = name.rsplit(".", 1)[-1]
                if dtype == "torch.float4_e2m1fn_x2":
                    recipe = "packed_fp4"
                elif dtype == "torch.float8_e8m0fnu" or (dtype == "torch.uint8" and leaf in {"w1_scale", "w2_scale"}):
                    recipe = "positive_e8m0_scale"
                elif leaf in {"topk_weight", "topk_weights"}:
                    recipe = "normalized_route_weights"
                elif leaf == "A_log":
                    recipe = "log_positive_decay"
                elif leaf == "dt_bias":
                    recipe = "inverse_softplus_dt"
                elif leaf in {"k_scale", "v_scale"}:
                    recipe = "unit_scale"
                elif dtype == "torch.uint8":
                    raise ValueError(f"unclassified byte tensor requires an explicit recipe: {name}")
                else:
                    recipe = "finite_normal"
                desc["recipe"] = recipe
                statistics["numeric_tensor_elements"] += math.prod(shape)
            return desc
        if isinstance(value, (tuple, list)):
            return {"sequence": type(value).__name__,
                    "items": [encode(x, name + f"[{i}]", output) for i, x in enumerate(value)]}
        if isinstance(value, dict):
            return {key: encode(item, str(key), output) for key, item in value.items()}
        if value is None or type(value) in {bool, int, float, str}:
            return value
        raise ValueError(f"unsupported capture value: {name}: {type(value).__name__}")

    shared = {key: encode(value, key) for key, value in (blob.get("shared") or {}).items()}
    records = []
    for index, row in enumerate(blob["records"]):
        records.append({"source_sig": row.get("sig"), "regime": row.get("regime"),
                        "args": {"sequence": "tuple", "items": [
                            encode(value, names[i]) for i, value in enumerate(row.get("args") or ())]},
                        "kwargs": encode(row.get("kwargs") or {}, "kwargs"),
                        "kwargs_before": encode(row.get("kwargs_before") or {}, "kwargs_before"),
                        "output_contract": encode(row.get("output"), "output", True)})
    return {"schema_version": 1, "input_policy": "generated_values_at_captured_contract",
            "source_reference_sha256": meta["reference_io_sha256"], "task_id": meta["task_id"],
            "correctness_case_count": len(records), "shared": shared, "records": records,
            "statistics": statistics}
