"""Routed mixture-of-experts layers (Tier 3 work, bounded without the routing data).

Every generator of these tasks routes each token to ``top_k`` distinct experts (``topk`` over the
router logits), so whatever the routing:

- FLOPs: ``n_tokens * top_k`` token-expert pairs each run stage 1 (``rows(w1) x hidden``) and
  stage 2 (``hidden x intermediate``): ``2 * T * k * (rows(w1) * H + H * I)``.
- Bytes: hidden states, routing ids and weights and the output are touched whole; at least
  ``top_k`` distinct experts are used, so at least ``top_k`` experts' slices of every
  per-expert weight tensor are read. Other experts may be skipped entirely.

Each stage is charged at the peak of its lowest-precision operand, the faster peak. Where the
operator quantizes its activations before the products (MXFP4, MXFP8 or fp8, as the task
description states), the model gives the activations' bits after that quantization.
"""
from __future__ import annotations

from ..interface import CaseShapes, Tensor
from . import Estimate, OperatorModel, register
from .attention import _first
from .gemm import add_product, mm


def _expert_bytes(shapes: CaseShapes, experts: int, used: int) -> int:
    """Per-expert weights for ``used`` experts, other unpadded inputs whole, outputs whole."""
    total = 0
    for tensor in shapes.inputs:
        if tensor.role == "weight" and tensor.shape and tensor.shape[0] == experts:
            total += tensor.nbytes // experts * used
        elif not (tensor.mutable or tensor.role == "state" or tensor.maybe_padded):
            total += tensor.nbytes
    return total + sum(t.nbytes for t in shapes.outputs)


def _logical_rows(tensor: Tensor) -> int:
    return tensor.shape[1]


def two_stage(w1: str, w2: str, **bits: int):
    def estimate(shapes: CaseShapes) -> Estimate:
        tokens = _first(shapes, "num_tokens", "n_tokens")
        top_k = _first(shapes, "top_k", "topk")
        hidden = shapes.symbols["hidden_size"]
        inter = _first(shapes, "intermediate_size", "inter_size")
        experts = shapes.symbols["num_experts"]
        pairs = tokens * top_k
        result = Estimate(notes=["routed_pairs_fixed_by_axes"])
        stage1 = 2 * pairs * _logical_rows(shapes.tensor(w1)) * hidden
        add_product(shapes, result, mm("1", "1", "1", "hidden_states", w1, **bits), stage1)
        add_product(shapes, result, mm("1", "1", "1", "hidden_states", w2, **bits), 2 * pairs * hidden * inter)
        result.bytes_override = _expert_bytes(shapes, experts, min(top_k, experts))
        result.notes.append("bytes: top_k experts' weights")
        return result
    return estimate


def single_stage(shapes: CaseShapes) -> Estimate:
    """fused_moe_mxfp4_gemm: one routed GEMM, a[T, H] against b[e, I, H] for each of the T * k pairs."""
    tokens, top_k = shapes.symbols["n_tokens"], shapes.symbols["topk"]
    hidden, inter, experts = (shapes.symbols[n] for n in ("hidden_size", "intermediate_size", "num_experts"))
    result = Estimate(notes=["routed_pairs_fixed_by_axes", "bytes: top_k experts' weights"])
    add_product(shapes, result, mm("1", "1", "1", "a", "b"), 2 * tokens * top_k * inter * hidden)
    result.bytes_override = _expert_bytes(shapes, experts, min(top_k, experts))
    return result


MODELS = {
    "fused_moe": two_stage("w13", "w2"),
    "fused_moe_tkw1": two_stage("w13", "w2"),
    "fused_moe_mxfp8": two_stage("w13", "w2", hidden_states=8),           # tokens quantized to MXFP8
    "fused_moe_fp8_blockscale": two_stage("w1", "w2", hidden_states=8),   # tokens quantized to fp8
    "fused_moe_mxfp4": two_stage("w1", "w2", hidden_states=4),            # tokens quantized to MXFP4
    "fused_moe_a16w4": two_stage("w1", "w2"),                             # bfloat16 tokens, not quantized
    # fp8 / MXFP8 tokens against MXFP4 weights. TraceLens's class for this kernel charges the FP4 matrix
    # peak; nothing shows the product needs the slower 8-bit rate, so the faster peak is used.
    "fused_moe_a8w4": two_stage("w1", "w2", hidden_states=4),
    "fused_moe_wna16": two_stage("w13_qweight", "w2_qweight", w13_qweight=4, w2_qweight=4),
    "fused_moe_mxfp4_gemm": single_stage,
}
for _name, _function in MODELS.items():
    register(_name, OperatorModel(f"moe/{_name}", 1, _function))
