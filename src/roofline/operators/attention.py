"""Attention with shapes fixed by the case (Tier 2).

Per (query, key) pair a head needs QKᵀ (2*d FLOPs) and PV (2*d FLOPs) in the forward pass;
masked pairs need neither. Counts:

- Causality is not an axis of the task contract (some references fix it with a module constant).
  A scenario whose description says "non-causal" counts every pair ``S^2``; every other scenario
  of a causal-capable operator counts the causal pairs ``S(S+1)/2``.
- Varlen batches: with ``T`` tokens split into ``b`` sequences the causal pairs
  ``sum L_i(L_i+1)/2`` are smallest when the lengths are equal, ``T^2/(2b) + T/2``.
- The backward pass needs five products per pair: recomputing QKᵀ (P is not an input),
  dV = Pᵀ dO, dP = dO Vᵀ, dQ = dS K and dK = dSᵀ Q.
- Softmax, scaling and masking are vector work and ignored.
"""
from __future__ import annotations

from ..interface import CaseShapes
from . import Estimate, OperatorModel, register
from .gemm import add_product, mm

CAUSAL_NOTE = "causal_pairs_counted"


def causal_pairs(length: int) -> int:
    return length * (length + 1) // 2


def is_non_causal(shapes: CaseShapes) -> bool:
    return "non-causal" in shapes.description


def self_attention_pairs(shapes: CaseShapes, length: int) -> tuple[int, str]:
    """(query, key) pairs of one sequence attending to itself, and the note that says which count."""
    if is_non_causal(shapes):
        return length * length, "all_pairs_counted: non-causal scenario"
    return causal_pairs(length), CAUSAL_NOTE


def window_pairs(length: int, window_left: int) -> int:
    """Pairs (i, j) with i - window_left <= j <= i, 0 <= j, i < length."""
    full = min(length, window_left + 1)
    return causal_pairs(full) + (length - full) * (window_left + 1)


def _first(shapes: CaseShapes, *names: str) -> int:
    for name in names:
        if name in shapes.symbols:
            return shapes.symbols[name]
    raise KeyError(f"{shapes.task_id}: none of {names} is an axis")


def _charge(shapes: CaseShapes, estimate: Estimate, flops: int, a: str = "q", b: str = "k|v") -> None:
    add_product(shapes, estimate, mm("1", "1", "1", a, b), flops)


def flash_attention_fwd(shapes: CaseShapes) -> Estimate:
    heads, dim = _first(shapes, "n_heads", "num_heads"), shapes.symbols["head_dim"]
    if "seq_len" in shapes.symbols or "seqlen" in shapes.symbols:
        per_sequence, note = self_attention_pairs(shapes, _first(shapes, "seq_len", "seqlen"))
        estimate = Estimate(notes=[note])
        pairs = shapes.symbols["batch"] * per_sequence
    else:
        tokens, batch = shapes.symbols["n_tokens"], shapes.symbols["batch"]
        estimate = Estimate(notes=[CAUSAL_NOTE, "varlen_equal_split_lower_bound"])
        pairs = tokens * tokens // (2 * batch) + tokens // 2
    _charge(shapes, estimate, 4 * dim * heads * pairs)
    return estimate


def sage_attention_fwd(shapes: CaseShapes) -> Estimate:
    """SageAttention quantizes inside the operator, as its description states: Q and K to int8 (v1) or
    MXFP4 (v2), V to fp8. QKᵀ is charged in the Q/K format and PV in fp8."""
    heads, dim = _first(shapes, "n_heads", "num_heads"), shapes.symbols["head_dim"]
    pairs = shapes.symbols["batch"] * causal_pairs(shapes.symbols["seqlen"])
    qk_key = "matrix_fp4" if "MXFP4" in shapes.description else "matrix_int8"
    estimate = Estimate(notes=[CAUSAL_NOTE, f"in_operator_quantization: qk {qk_key}, pv matrix_fp8"])
    estimate.add_matrix(qk_key, 2 * dim * heads * pairs)
    estimate.add_matrix("matrix_fp8", 2 * dim * heads * pairs)
    return estimate


def sliding_window_attention(shapes: CaseShapes) -> Estimate:
    heads, dim = shapes.symbols["n_heads"], shapes.symbols["head_dim"]
    pairs = shapes.symbols["batch"] * window_pairs(shapes.symbols["seq_len"], int(shapes.scalars["window_left"]))
    estimate = Estimate()
    _charge(shapes, estimate, 4 * dim * heads * pairs)
    return estimate


def flash_attention_bwd(shapes: CaseShapes) -> Estimate:
    heads, dim = shapes.symbols["n_heads"], shapes.symbols["head_dim"]
    per_sequence, note = self_attention_pairs(shapes, shapes.symbols["seqlen"])
    pairs = shapes.symbols["batch"] * per_sequence
    estimate = Estimate(notes=[note, "bytes: saved o and lse excluded (a kernel may recompute them)"])
    _charge(shapes, estimate, 5 * 2 * dim * heads * pairs, "q", "do")
    estimate.bytes_override = (sum(t.nbytes for t in shapes.inputs if t.name not in ("o", "lse"))
                               + sum(t.nbytes for t in shapes.outputs))
    return estimate


def causal_self_attention(shapes: CaseShapes) -> Estimate:
    """minGPT block: QKV projection, causal attention over E = heads * head_dim, output projection."""
    batch, length, embed = (shapes.symbols[n] for n in ("batch", "seq_len", "n_embd"))
    estimate = Estimate()
    add_product(shapes, estimate, mm("batch * seq_len", "3 * n_embd", "n_embd", "x", "c_attn_weight"))
    _charge(shapes, estimate, 4 * embed * batch * causal_pairs(length), "x", "c_attn_weight")
    add_product(shapes, estimate, mm("batch * seq_len", "n_embd", "n_embd", "x", "c_proj_weight"))
    return estimate


def vision_attention_block(shapes: CaseShapes) -> Estimate:
    """Self-attention over hw positions, unmasked, with input and output projections."""
    batch, positions, embed = (shapes.symbols[n] for n in ("batch", "hw", "embed_dim"))
    estimate = Estimate()
    add_product(shapes, estimate, mm("batch * hw", "3 * embed_dim", "embed_dim", "x", "in_proj_weight"))
    _charge(shapes, estimate, 4 * embed * batch * positions * positions, "x", "in_proj_weight")
    add_product(shapes, estimate, mm("batch * hw", "embed_dim", "embed_dim", "x", "out_proj_weight"))
    return estimate


def nonlocal_attention_dot(shapes: CaseShapes) -> Estimate:
    """Non-local block: E = elu(qᵀk)/n, y = v E, out = W_g y + b_g.

    The products are counted in the reference's order: the query, key and value projections, the
    energy qᵀk, the aggregation v E and the output projection. Merging the value and output maps
    into one C x C map is equal in real arithmetic, but run through the evaluator the merged order
    was not shown to pass in every case, so it is not counted.
    """
    batch, channels, positions, qk = (shapes.symbols[n] for n in ("batch", "channels", "hw", "qk_channels"))
    projections = 2 * 2 * batch * positions * channels * qk + 2 * 2 * batch * positions * channels * channels
    energy = 2 * batch * positions * positions * qk
    aggregate = 2 * batch * positions * positions * channels
    estimate = Estimate()
    _charge(shapes, estimate, projections + energy + aggregate, "x", "query_weight")
    return estimate


for _name, _function in {
    "flash_attention_fwd": flash_attention_fwd, "sage_attention_fwd": sage_attention_fwd,
    "sliding_window_attention": sliding_window_attention, "flash_attention_bwd": flash_attention_bwd,
    "causal_self_attention": causal_self_attention, "vision_attention_block": vision_attention_block,
    "nonlocal_attention_dot": nonlocal_attention_dot,
}.items():
    register(_name, OperatorModel(f"attention/{_name}", 1, _function))
