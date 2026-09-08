"""Version-1 input policy. Builder repairs may edit this file only.

Random tensors are synthesized, not reconstructed captured inputs. Preserve
all definition shapes, dtypes, scalar literals and operator semantics.
"""

from __future__ import annotations

import hashlib

import torch

from scripts.task_api import dimensions, dtype, shape_of


def make_inputs(definition, row, policy, device="cuda"):
    seed = int.from_bytes(hashlib.sha256(
        f"{policy['seed']}:{row['workload']['uuid']}".encode()).digest()[:8], "little")
    generator = torch.Generator(device=device).manual_seed(seed)
    axes = dimensions(definition, row)
    if definition["op_type"] == "moe":
        return _moe(definition, row, axes, generator, device)
    if definition["op_type"] != "gemm":
        raise NotImplementedError("Implement this operator's input policy from its declared contract")
    result = {}
    for name, spec in definition["inputs"].items():
        desc = row["workload"]["inputs"][name]
        if desc["type"] == "scalar":
            result[name] = desc["value"]
        else:
            kind = dtype(spec["dtype"])
            if kind not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise NotImplementedError(f"No generic random policy for {name}: {kind}")
            result[name] = torch.randn(shape_of(spec, axes), generator=generator, device=device, dtype=kind)
    return result


def _moe(definition, row, axes, generator, device):
    required = {"num_tokens", "model_dim", "num_experts", "topk", "w1_rows", "w1_cols",
                "w2_cols", "w1_scale_cols", "w2_scale_cols"}
    if not required <= axes.keys() or "quantization:per_1x32" not in definition.get("tags", []):
        raise NotImplementedError("Only declared per_1x32 MXFP4 MoE inputs have a built-in policy")
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

    e, d, rows, m, k = (axes[n] for n in ("num_experts", "model_dim", "w1_rows", "num_tokens", "topk"))
    if k > e or rows % 2 or axes["w1_cols"] * 2 != d or axes["w2_cols"] * 2 != rows // 2:
        raise ValueError("Inconsistent MXFP4 MoE dimensions")
    result = {}
    for name, shape in (("w1", (e, rows, d)), ("w2", (e, d, rows // 2))):
        raw = 0.125 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        packed, scales = dynamic_mxfp4_quant(raw.reshape(-1, shape[-1]))
        packed = packed.reshape(shape[0], shape[1], -1)
        scales = e8m0_shuffle(scales).reshape(shape[0], shape[1], -1)
        result[name] = shuffle_weight(packed.contiguous(), (16, 16)).view(dtype(definition["inputs"][name]["dtype"]))
        result[name].is_shuffled = True
        result[name + "_scale"] = scales.view(dtype(definition["inputs"][name + "_scale"]["dtype"]))
    result["hidden_states"] = 0.25 * torch.randn((m, d), device=device, dtype=torch.bfloat16, generator=generator)
    scores = torch.rand((m, e), device=device, generator=generator)
    ids = scores.topk(k, dim=-1).indices
    result["topk_ids"] = ids.to(torch.int32).contiguous()
    result["topk_weights"] = scores.gather(1, ids).softmax(-1).contiguous()
    for name, desc in row["workload"]["inputs"].items():
        if desc["type"] == "scalar":
            result[name] = desc["value"]
    return result
