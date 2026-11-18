"""Composite blocks and chunked state-space products with fixed totals (Tier 2).

Chunk boundaries (``cu_chunk_seqlens``) come from the generated inputs, but the token total is
an axis. A per-chunk cost quadratic in the chunk length is smallest when the ``n_chunks`` chunks
are equally long (convexity), so ``sum n_c^2 >= T^2 / n_chunks`` bounds every split.
Work that only some chunks do (the inter-chunk state term of a chunk that continues a sequence)
is counted as zero.
"""
from __future__ import annotations

from ..bytes_model import generic_bytes
from ..interface import CaseShapes
from . import Estimate, OperatorModel, attention, register
from .attention import causal_pairs
from .gemm import add_product, mm


def mingpt_block(shapes: CaseShapes) -> Estimate:
    """LN -> causal self-attention (QKV, scores, PV, projection) -> LN -> MLP (4E hidden)."""
    batch, length, embed = (shapes.symbols[n] for n in ("batch", "seq_len", "n_embd"))
    estimate = Estimate()
    rows = "batch * seq_len"
    add_product(shapes, estimate, mm(rows, "3 * n_embd", "n_embd", "x", "c_attn_weight"))
    add_product(shapes, estimate, mm("1", "1", "1", "x", "c_attn_weight"), 4 * embed * batch * causal_pairs(length))
    add_product(shapes, estimate, mm(rows, "n_embd", "n_embd", "x", "attn_proj_weight"))
    add_product(shapes, estimate, mm(rows, "4 * n_embd", "n_embd", "x", "c_fc_weight"))
    add_product(shapes, estimate, mm(rows, "n_embd", "4 * n_embd", "x", "mlp_proj_weight"))
    return estimate


def _mhc(weight: str, mixes: str):
    def estimate(shapes: CaseShapes) -> Estimate:
        """Only the mix projection of the hc_mult residual streams counts; the stream mixing is vector work.
        The projection weight is float32 and the residual bfloat16; the product is charged at the
        residual's 16-bit peak, as TraceLens's mHC classes charge it."""
        result = Estimate()
        add_product(shapes, result, mm("tokens", mixes, "hc_hidden", "residual", weight, **{weight: 16}))
        return result
    return estimate


def _square_sum_lower_bound(total: int, parts: int) -> int:
    return total * total // parts


def ssd_bmm_chunk(shapes: CaseShapes) -> Estimate:
    """out[c, g] = C_c B_cᵀ over each chunk's n_c tokens: 2 * n_c^2 * dstate per group."""
    tokens, chunks = shapes.symbols["n_tokens"], shapes.symbols["n_chunks"]
    flops = 2 * shapes.symbols["ngroups"] * shapes.symbols["dstate"] * _square_sum_lower_bound(tokens, chunks)
    estimate = Estimate(notes=["equal_chunk_split_lower_bound"])
    add_product(shapes, estimate, mm("1", "1", "1", "C", "B"), flops)
    return estimate


def ssd_chunk_scan(shapes: CaseShapes) -> Estimate:
    """Intra-chunk causal product (weights x x) per head; the inter-chunk C·state term is not counted."""
    tokens, chunks = shapes.symbols["n_tokens"], shapes.symbols["n_chunks"]
    pairs = tokens * tokens // (2 * chunks) + tokens // 2
    flops = 2 * shapes.symbols["num_heads"] * shapes.symbols["head_dim"] * pairs
    estimate = Estimate(notes=["equal_chunk_split_lower_bound", "inter_chunk_term_not_counted"])
    add_product(shapes, estimate, mm("1", "1", "1", "cb", "x"), flops)
    # Bytes: x, C, D and out whole; of the chunk-padded inputs only each chunk's real tokens, and of
    # cb only the causal triangle; states are skipped (the last chunk's state is never read, and a
    # chunk that starts a sequence reads none).
    whole = sum(shapes.tensor(n).nbytes for n in ("x", "C", "D", "out"))
    cb = shapes.tensor("cb")
    triangle = shapes.symbols["num_groups"] * pairs * (cb.nbytes // max(1, _count(cb.shape)))
    per_token = sum(shapes.symbols["num_heads"] * tokens * (shapes.tensor(n).nbytes // _count(shapes.tensor(n).shape))
                    for n in ("dt", "dA_cumsum"))
    estimate.bytes_override = whole + triangle + per_token
    estimate.notes.append("bytes: chunk padding, the causal half of cb and states excluded")
    return estimate


def _count(shape: tuple[int, ...]) -> int:
    count = 1
    for extent in shape:
        count *= extent
    return count


def fused_bmm_rope_kv_cache(shapes: CaseShapes) -> Estimate:
    """Per q head: q_nope (quantized to fp8 in the kernel) times w_kᵀ; rope and the cache write are bytes.

    Bytes beyond the generic rule: every head's w_k and its scale are operands of the product, and
    each token writes one [k_nope | rotated k_pe] row into the cache (the generator draws the slots
    without replacement). The cos / sin tables are read only at the tokens' positions and stay out.
    """
    estimate = Estimate(notes=["bytes: w_k and one cache row per token added to G2"])
    add_product(shapes, estimate, mm("num_q_heads * n_tokens", "kv_lora_dim", "qk_nope_head_dim", "w_k", "w_k"))
    cache = shapes.tensor("kv_cache")
    row_bytes = cache.nbytes // max(1, cache.shape[0])
    estimate.bytes_override = (generic_bytes(shapes).bytes + shapes.tensor("w_k").nbytes
                               + shapes.tensor("w_k_scale").nbytes + shapes.symbols["n_tokens"] * row_bytes)
    return estimate


def gather_kv_b_proj(shapes: CaseShapes) -> Estimate:
    """Every gathered latent row (total_kv of them) is projected by the (proj_rows x kv_lora) weight."""
    estimate = Estimate(notes=["bytes: gathered k_buffer rows only"])
    add_product(shapes, estimate, mm("total_kv", "proj_rows", "kv_lora_dim", "k_buffer", "kv_proj_weight"))
    # Only the total_kv rows named by kv_indices are read from the (much larger) block pool. The
    # per-request offsets (metadata) are not counted: output row i depends only on kv_indices[i].
    k_buffer = shapes.tensor("k_buffer")
    row_bytes = k_buffer.nbytes // max(1, k_buffer.shape[0] * k_buffer.shape[1])
    others = sum(t.nbytes for t in shapes.inputs
                 if t.name != "k_buffer" and t.role != "metadata" and not t.maybe_padded)
    estimate.bytes_override = (shapes.symbols["total_kv"] * row_bytes + others
                               + sum(t.nbytes for t in shapes.outputs))
    return estimate


def jagged_dense_bmm(shapes: CaseShapes) -> Estimate:
    """out[rows of group g] = jagged rows @ dense[g] + bias[g]; the row total is an axis.

    A group with no rows needs no dense or bias read. The rows are in some group, so at least one
    group's weights are read; the generated offsets say how many groups have rows
    (``routing_data``).
    """
    estimate = Estimate(notes=["bytes: one group's dense weights and bias"])
    add_product(shapes, estimate, mm("total_rows", "n", "k", "jagged", "dense"))
    groups = shapes.symbols["n_groups"]
    per_group = (shapes.tensor("dense").nbytes + shapes.tensor("bias").nbytes) // groups
    estimate.bytes_override = (shapes.tensor("jagged").nbytes + shapes.tensor("seq_offsets").nbytes + per_group
                               + sum(t.nbytes for t in shapes.outputs))
    return estimate


# The embedded-Gaussian non-local block differs from the dot-product one only in the energy
# nonlinearity (softmax over j instead of elu / n): the same products are necessary. Registered here
# so attention.py is untouched.
register("nonlocal_attention_gaussian",
         OperatorModel("attention/nonlocal_attention_gaussian", 1, attention.nonlocal_attention_dot))

for _name, _function in {
    "jagged_dense_bmm_broadcast_add": jagged_dense_bmm,
    "mingpt_block": mingpt_block,
    "mhc_fused_post_pre": _mhc("fn", "mix_dim"),
    "mhc_fused_post_pre_rmsnorm": _mhc("proj_weight", "mixes"),
    "ssd_bmm_chunk": ssd_bmm_chunk, "ssd_chunk_scan": ssd_chunk_scan,
    "fused_bmm_rope_kv_cache": fused_bmm_rope_kv_cache, "gather_kv_b_proj": gather_kv_b_proj,
}.items():
    register(_name, OperatorModel(f"composite/{_name}", 1, _function))
