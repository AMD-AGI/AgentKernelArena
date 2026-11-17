"""Routed weights counted from the generated routing: only experts or groups that receive rows are read."""
from __future__ import annotations

from typing import Any

from ..generated import register_data_model
from ..interface import CaseShapes
from . import Estimate, model_for


def _with_used_experts(operator: str, experts_axis: str, used):
    static = model_for(operator)

    def estimate(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
        result = static.estimate(shapes)  # FLOPs do not depend on the routing
        experts = shapes.symbols[experts_axis]
        count = used(inputs)
        total = 0
        for tensor in shapes.inputs:
            if tensor.role == "weight" and tensor.shape and tensor.shape[0] == experts:
                total += tensor.nbytes // experts * count
            elif not (tensor.mutable or tensor.role == "state" or tensor.maybe_padded):
                total += tensor.nbytes
        result.bytes_override = total + sum(t.nbytes for t in shapes.outputs)
        result.notes = [n for n in result.notes if not n.startswith("bytes:")]
        result.notes.append(f"data_dependent: {count} of {experts} experts or groups receive rows")
        return result
    return estimate


def _distinct_ids(name: str):
    return lambda inputs: len({int(v) for v in _flat(inputs[name].tolist()) if int(v) >= 0})


def _nonempty(name: str):
    return lambda inputs: sum(1 for v in inputs[name].tolist() if int(v) > 0)


def _nonempty_spans(name: str):
    """Groups with at least one row, from cumulative row offsets."""
    def count(inputs):
        offsets = [int(v) for v in inputs[name].tolist()]
        return sum(1 for start, end in zip(offsets, offsets[1:]) if end > start)
    return count


def _flat(values):
    for value in values:
        if isinstance(value, list):
            yield from _flat(value)
        else:
            yield value


register_data_model("grouped_gemm", "routing_data/grouped_gemm@v1",
                    _with_used_experts("grouped_gemm", "num_groups", _nonempty("group_sizes")))
register_data_model("jagged_dense_bmm_broadcast_add", "routing_data/jagged_dense_bmm_broadcast_add@v1",
                    _with_used_experts("jagged_dense_bmm_broadcast_add", "n_groups", _nonempty_spans("seq_offsets")))
for _operator in ("fused_moe", "fused_moe_tkw1", "fused_moe_mxfp8", "fused_moe_fp8_blockscale", "fused_moe_mxfp4",
                  "fused_moe_a16w4", "fused_moe_a8w4", "fused_moe_wna16", "fused_moe_mxfp4_gemm"):
    register_data_model(_operator, f"routing_data/{_operator}@v1",
                        _with_used_experts(_operator, "num_experts", _distinct_ids("topk_ids")))
