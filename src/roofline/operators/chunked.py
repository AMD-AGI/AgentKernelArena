"""Chunked linear-attention recurrences (Gated DeltaNet, KDA) on one sequence of ``n_tokens``.

The tokens form consecutive chunks of ``chunk_size`` (the last may be partial). Within a chunk of
``n`` tokens a strictly lower-triangular product has ``n(n-1)/2`` pairs and a causal one
``n(n+1)/2``; only those pairs are counted.

- A per-head scalar gate ``exp(g_i - g_j)`` or ``beta_i`` factors out of a product, so Q Kᵀ and
  K Kᵀ are counted once per *key* head where key heads are shared by several value heads.
- KDA gates are per channel, so its products are counted per head.
- Triangular inverses (``solve_tril``) are not counted: their cost is far below the bytes moved.

Bytes: inputs and outputs whole, except the chunked ``A`` matrices, of which only the triangle the
reference uses is counted (``solve_tril`` ignores the diagonal and above; a causal ``A`` only the
part on and below the diagonal).
"""
from __future__ import annotations

from ..interface import CaseShapes
from . import Estimate, OperatorModel, register
from .gemm import add_product, mm


def chunk_pairs(tokens: int, chunk: int, *, strict: bool) -> int:
    def pairs(n: int) -> int:
        return n * (n - 1) // 2 if strict else n * (n + 1) // 2
    return (tokens // chunk) * pairs(chunk) + pairs(tokens % chunk)


def _key_heads(shapes: CaseShapes) -> int:
    for name in ("num_key_heads", "num_k_heads"):
        if name in shapes.symbols:
            return shapes.symbols[name]
    return shapes.symbols["num_heads"]


def _bytes(shapes: CaseShapes, triangles: dict[str, bool]) -> int:
    """Whole unpadded inputs and all outputs, with each named [T, H, C] input counted as its triangle."""
    tokens, chunk, heads = (shapes.symbols[n] for n in ("n_tokens", "chunk_size", "num_heads"))
    total = 0
    for tensor in shapes.inputs:
        if tensor.name in triangles:
            element = tensor.nbytes // max(1, tokens * heads * chunk)
            total += heads * chunk_pairs(tokens, chunk, strict=triangles[tensor.name]) * element
        elif not (tensor.mutable or tensor.role == "state" or tensor.maybe_padded):
            total += tensor.nbytes
    return total + sum(t.nbytes for t in shapes.outputs)


def _common(shapes: CaseShapes):
    tokens, chunk = shapes.symbols["n_tokens"], shapes.symbols["chunk_size"]
    return tokens, chunk, chunk_pairs(tokens, chunk, strict=True), chunk_pairs(tokens, chunk, strict=False)


def scaled_dot_kkt(shapes: CaseShapes) -> Estimate:
    """A = beta_i exp(g_i - g_j) k_i·k_j for j < i: one strict K Kᵀ per key head."""
    _, _, strict, _ = _common(shapes)
    estimate = Estimate(notes=["per_key_head_products"])
    add_product(shapes, estimate, mm("1", "1", "1", "k", "k"), 2 * shapes.symbols["head_dim"] * _key_heads(shapes)
                * strict)
    return estimate


def kda_chunk_kkt(shapes: CaseShapes) -> Estimate:
    """A (strict, from k) and Aqk (causal, from q) per head with per-channel gates."""
    _, _, strict, causal = _common(shapes)
    estimate = Estimate()
    add_product(shapes, estimate, mm("1", "1", "1", "q", "k"),
                2 * shapes.symbols["head_dim"] * shapes.symbols["num_heads"] * (strict + causal))
    return estimate


def recompute(key_dim: str, value_dim: str, matrix: str, strict_input: bool):
    def estimate(shapes: CaseShapes) -> Estimate:
        """w = A (beta k ...) and u = A (beta v): a causal [n x n] times [n x d] product per head."""
        _, _, _, causal = _common(shapes)
        width = shapes.symbols[key_dim] + shapes.symbols[value_dim]
        result = Estimate(notes=[f"bytes: triangle of {matrix}"])
        add_product(shapes, result, mm("1", "1", "1", "k", "v"), 2 * width * shapes.symbols["num_heads"] * causal)
        result.bytes_override = _bytes(shapes, {matrix: strict_input})
        return result
    return estimate


def solve_tril(shapes: CaseShapes) -> Estimate:
    result = Estimate(notes=["bytes: strict lower triangle of A", "triangular_inverse_not_counted"])
    result.bytes_override = _bytes(shapes, {"A": True})
    return result


def gdn_chunk_fwd_h(shapes: CaseShapes) -> Estimate:
    """Per chunk and head: v_new = u - w Sᵀ (every token) and S += kᵀ v_new (all but the last chunk)."""
    tokens, chunk = shapes.symbols["n_tokens"], shapes.symbols["chunk_size"]
    dk, dv, heads = (shapes.symbols[n] for n in ("head_dim_k", "head_dim_v", "num_heads"))
    estimate = Estimate(notes=["last_chunk_state_update_not_counted"])
    flops = 2 * tokens * dk * dv * heads + 2 * max(0, tokens - chunk) * dk * dv * heads
    add_product(shapes, estimate, mm("1", "1", "1", "w", "u"), flops)
    return estimate


def gdn_chunk_fwd_o(shapes: CaseShapes) -> Estimate:
    """o = q Sᵀ (per head) + causal(q kᵀ, per key head) v (per head)."""
    tokens, _, _, causal = _common(shapes)
    dk, dv, heads = (shapes.symbols[n] for n in ("head_dim_k", "head_dim_v", "num_heads"))
    estimate = Estimate(notes=["per_key_head_products"])
    flops = 2 * tokens * dk * dv * heads + 2 * causal * (dk * _key_heads(shapes) + dv * heads)
    add_product(shapes, estimate, mm("1", "1", "1", "q", "v"), flops)
    return estimate


def kda_chunk_fwd_o(shapes: CaseShapes) -> Estimate:
    """o = (q exp2(g)) Sᵀ + causal(A) v per head; only A's causal triangle is read."""
    tokens, _, _, causal = _common(shapes)
    dk, dv, heads = (shapes.symbols[n] for n in ("head_dim_k", "head_dim_v", "num_heads"))
    estimate = Estimate(notes=["bytes: causal triangle of A"])
    add_product(shapes, estimate, mm("1", "1", "1", "q", "v"), 2 * tokens * dk * dv * heads + 2 * causal * dv * heads)
    estimate.bytes_override = _bytes(shapes, {"A": False})
    return estimate


def recurrent_attention(tokens_axes: tuple[str, ...]):
    def estimate(shapes: CaseShapes) -> Estimate:
        """A whole linear-attention prefill with a [d x d] state per head (lightning, KDA).

        Every algorithm (recurrent, or chunked with any chunk size) does at least two dense products
        per head over all T tokens: the queries read a state that holds the input state (o_t = S_t q_t,
        [T x d] by [d x d]), and the final state is the input state plus T rank-1 updates
        ([d x T] by [T x d]). That is 4 T d^2 per head. The intra-chunk products, the delta rule's
        S k (KDA) and the gates are left out, so this is below every chunked count. A kernel that drops
        history whose decay falls below the tolerance can do less work, but the floor of these shapes
        is memory-bound by a wide margin (4 d^2 FLOPs against at least 8 d bytes per token and head).
        """
        tokens = 1
        for axis in tokens_axes:
            tokens *= shapes.symbols[axis]
        dim, heads = shapes.symbols["head_dim"], shapes.symbols["num_heads"]
        result = Estimate(notes=["decay_truncation_sensitive"])
        add_product(shapes, result, mm("1", "1", "1", "q", "k"), 4 * tokens * dim * dim * heads)
        return result
    return estimate


for _name, _function in {
    "chunk_scaled_dot_kkt": scaled_dot_kkt, "chunk_cumsum_kkt": scaled_dot_kkt, "kda_chunk_kkt": kda_chunk_kkt,
    "recompute_w_u": recompute("key_dim", "value_dim", "A", False),
    "kda_recompute_w_u": recompute("head_dim_k", "head_dim_v", "A", False),
    "solve_tril_recompute_w_u": recompute("key_dim", "value_dim", "A_raw", True),
    "solve_tril": solve_tril, "gdn_chunk_fwd_h": gdn_chunk_fwd_h, "gdn_chunk_fwd_o": gdn_chunk_fwd_o,
    "kda_chunk_fwd_o": kda_chunk_fwd_o,
    "kda_chunk_attention": recurrent_attention(("n_seqs", "seq_len")),
    "lightning_attention_prefill": recurrent_attention(("n_tokens",)),
}.items():
    register(_name, OperatorModel(f"chunked/{_name}", 1, _function))
