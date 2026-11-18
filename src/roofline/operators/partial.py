"""Memory-bound operators whose generic byte rule is wrong in a fixed, data-independent way.

The generic rule G2 counts index tensors and mutable state as zero bytes. For the operators below
every index entry is read, or every state slot is read and written back, whatever the inputs are,
so the byte count is fixed by the shapes. Each model states why each tensor it counts must be
moved. FLOPs are not counted: the arithmetic is elementwise or a per-slot matrix-vector product
whose intensity is far below the vector-peak to bandwidth ratio.
"""
from __future__ import annotations

from ..interface import CaseShapes
from . import Estimate, OperatorModel, register


def _sum(tensors) -> int:
    return sum(t.nbytes for t in tensors)


def count_expert_tokens(shapes: CaseShapes) -> Estimate:
    """Every topk_ids entry may be valid and changes one count, so each is read; the histogram is written."""
    return Estimate(bytes_override=shapes.tensor("topk_ids").nbytes + _sum(shapes.outputs),
                    notes=["bytes: every routing entry read, histogram written"])


def moe_sorting(shapes: CaseShapes) -> Estimate:
    """Each (token, slot) id decides where its entry goes and each weight is copied to sorted_weights, so
    topk_ids and topk_weights are read whole; every output (sorted tables and the zeroed moe_buf) is
    compared whole, so written whole."""
    inputs = shapes.tensor("topk_ids").nbytes + shapes.tensor("topk_weights").nbytes
    return Estimate(bytes_override=inputs + _sum(shapes.outputs),
                    notes=["bytes: routing ids and weights read, every output written"])


def fused_mxfp4_quant_moe_sort(shapes: CaseShapes) -> Estimate:
    """Every row of x is quantized into x_fp4; the scale row at sorted position j is the scale of the row
    named by sorted_ids[j] (or zero for a sentinel), and the reference uses every entry, so x and
    sorted_ids are read whole. num_valid_ids (8 bytes) is not counted."""
    inputs = shapes.tensor("x").nbytes + shapes.tensor("sorted_ids").nbytes
    return Estimate(bytes_override=inputs + _sum(shapes.outputs),
                    notes=["bytes: x and every sorted id read, outputs written"])


def moe_silu_and_mul_mxfp4_quant_sorted(shapes: CaseShapes) -> Estimate:
    """Every row of x is activated and quantized into out at its own position, so x is read whole. The
    reference ignores sorted_ids entries past num_valid_ids[0], which the shapes do not reveal, so
    sorted_ids is not counted (4 bytes per row against 4096 bytes of x)."""
    return Estimate(bytes_override=shapes.tensor("x").nbytes + _sum(shapes.outputs),
                    notes=["bytes: x read, outputs written; sorted_ids not counted"])


def _whole_state_update(shapes: CaseShapes) -> Estimate:
    """The slot index is a built-in permutation of all slots (declared ``distribution: permutation``), so
    every state slot is read (the output depends on it) and written back (it is compared whole and
    every slot changes). Every other input enters every output element or slot and is read once; the
    output is written once."""
    state = [t for t in shapes.inputs if t.mutable]
    return Estimate(bytes_override=_sum(shapes.inputs) + _sum(state) + _sum(shapes.outputs),
                    notes=["bytes: every state slot read and written (permutation of the slots)"])


# Share of the state write a correct kernel cannot skip. A review of the generated inputs found about
# 3.7% of the 64-byte state lines of selective_state_update change by less than the baseline-relative
# tolerance (dt is often about 1e-4), so a kernel could leave them unwritten; 10% is left out.
NEGLIGIBLE_UPDATE_MARGIN = 0.9


def selective_state_update(shapes: CaseShapes) -> Estimate:
    """As ``_whole_state_update`` (every slot read, every input read once), except that only 90% of the
    state write is counted: lines whose update is below the tolerance need not be written."""
    state = sum(t.nbytes for t in shapes.inputs if t.mutable)
    return Estimate(bytes_override=_sum(shapes.inputs) + int(state * NEGLIGIBLE_UPDATE_MARGIN) + _sum(shapes.outputs),
                    notes=["bytes: every state slot read, 90% of the state write counted",
                           "negligible_update_margin"])


register("count_expert_tokens", OperatorModel("partial/count_expert_tokens", 1, count_expert_tokens))
register("moe_sorting", OperatorModel("partial/moe_sorting", 1, moe_sorting))
register("fused_mxfp4_quant_moe_sort", OperatorModel("partial/fused_mxfp4_quant_moe_sort", 1,
                                                     fused_mxfp4_quant_moe_sort))
register("moe_silu_and_mul_mxfp4_quant_sorted", OperatorModel("partial/moe_silu_and_mul_mxfp4_quant_sorted", 1,
                                                              moe_silu_and_mul_mxfp4_quant_sorted))
register("selective_state_update", OperatorModel("partial/selective_state_update", 2, selective_state_update))
register("lightning_attention_decode", OperatorModel("partial/lightning_attention_decode", 1, _whole_state_update))
