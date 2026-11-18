"""Decode attention over paged or ragged KV caches, counted from the generated context lengths.

For a sequence with ``L`` visible keys, each query head needs QKᵀ (2*d_qk) and PV (2*d_v) per key,
and each KV head's ``L`` keys and values must be read once; queries and outputs are read and
written whole. Cache pages or slots no sequence uses are not counted, nor are scales, block
tables, worklists or other metadata.

The second half of the module counts prefill, extend and sparse (top-k) attention the same way:
FLOPs only for the (query, key) pairs the contract makes visible, bytes only for the keys and
values some visible pair uses. Every model below states its lower-bound argument; the shared
rules are:

- Products are charged at the peak of the lowest-precision operand (``add_product``).
- Softmax, soft-capping, SiLU, ALiBi, sinks and scaling are vector work and ignored.
- Sparse selections count each distinct selected key once per query for FLOPs (a repeated
  entry contributes the same term, so a kernel could weight it instead of recomputing it) and
  once per batch for bytes (rows shared by several queries need to be read only once).
- Queries of rows that select nothing need not be read; outputs are written whole because the
  evaluator compares them whole. Index, length and schedule tensors are not counted.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from ..generated import register_data_model
from ..interface import CaseShapes
from . import Estimate
from .gemm import add_product, mm


def _ints(value: Any) -> list[int]:
    return [int(v) for v in value.tolist()]


def context_lengths(inputs: dict[str, Any]) -> list[int]:
    """Visible keys per sequence: an explicit length tensor, or the differences of a token indptr."""
    for name in ("context_lengths", "seq_lens", "seqused_k"):
        if name in inputs:
            return _ints(inputs[name])
    if "kv_indptr" in inputs:
        bounds = _ints(inputs["kv_indptr"])
        return [b - a for a, b in zip(bounds, bounds[1:])]
    raise KeyError("no context-length input")


def _element(shapes: CaseShapes, name: str) -> int:
    tensor = shapes.tensor(name)
    count = 1
    for extent in tensor.shape:
        count *= extent
    return tensor.nbytes // max(1, count)


def _decode(query: str, key: str, value: str | None, heads: str, kv_heads: str, dim_qk: str, dim_v: str,
            window: str | None = None, length_names: tuple[str, ...] | None = None):
    def estimate(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
        lengths = context_lengths({n: inputs[n] for n in (length_names or inputs) if n in inputs})
        if window is not None:
            lengths = [min(length, int(shapes.scalars[window])) for length in lengths]
        keys = sum(lengths)
        q_heads, h_kv = shapes.value(heads), shapes.value(kv_heads)
        d_qk, d_v = shapes.value(dim_qk), shapes.value(dim_v)
        result = Estimate(notes=["data_dependent: generated context lengths"])
        add_product(shapes, result, mm("1", "1", "1", query, key), 2 * (d_qk + d_v) * q_heads * keys)
        kv_bytes = keys * h_kv * d_qk * _element(shapes, key)
        if value is not None:
            kv_bytes += keys * h_kv * d_v * _element(shapes, value)
        result.bytes_override = kv_bytes + shapes.tensor(query).nbytes + sum(t.nbytes for t in shapes.outputs)
        return result
    return estimate


def _mla(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """MLA decode: one latent row per key (head_size = kv_lora_rank + rope) serves as both K and V."""
    lengths = context_lengths({n: inputs[n] for n in ("seq_lens", "kv_indptr") if n in inputs})
    keys = sum(lengths)
    heads, latent, lora = (shapes.value(n) for n in ("num_heads", "head_size", "kv_lora_rank"))
    result = Estimate(notes=["data_dependent: generated context lengths"])
    add_product(shapes, result, mm("1", "1", "1", "q", "kv_cache"), 2 * (latent + lora) * heads * keys)
    result.bytes_override = (keys * latent * _element(shapes, "kv_cache") + shapes.tensor("q").nbytes
                             + sum(t.nbytes for t in shapes.outputs))
    return result


for _operator, _model in {
    "paged_attention_decode": _decode("query", "key_cache", "value_cache", "num_query_heads", "num_kv_heads",
                                      "head_size", "head_size", length_names=("context_lengths", "seq_lens")),
    "paged_attention_decode_swa": _decode("query", "key_cache", "value_cache", "num_query_heads", "num_kv_heads",
                                          "head_size", "head_size", window="sliding_window",
                                          length_names=("context_lengths",)),
    "decode_attention": _decode("q", "k_buffer", "v_buffer", "n_heads", "n_kv_heads", "head_dim", "head_dim",
                                length_names=("kv_indptr",)),
    "paged_attention_ragged": _decode("query", "key_cache", "value_cache", "num_query_heads", "num_kv_heads",
                                      "head_size", "head_size", length_names=("kv_indptr",)),
    "mla_decode": _mla,
}.items():
    register_data_model(_operator, f"attention_data/{_operator}@v1", _model)


# --- Prefill, extend and sparse attention --------------------------------------------------------

TASKS_DIR = Path(__file__).resolve().parents[3] / "tasks"


def _per_row(tensor) -> int:
    """Bytes of one leading-axis row of a declared tensor."""
    return tensor.nbytes // max(1, tensor.shape[0])


def bottom_right_pairs(q_len: int, kv_len: int, window: int | None = None) -> int:
    """Visible pairs when query row r (of q_len) sits at key position kv_len - q_len + r and sees every
    key at or before it (with a window, only those fewer than ``window`` positions before it)."""
    total = 0
    for r in range(q_len):
        visible = min(kv_len, max(0, kv_len - q_len + r + 1))
        total += visible if window is None else min(window, visible)
    return total


def bottom_right_keys(q_len: int, kv_len: int, window: int | None = None) -> int:
    """Keys some row of ``bottom_right_pairs`` sees: all kv_len, or with a window the last q_len + window - 1."""
    if q_len <= 0:
        return 0
    return max(0, kv_len if window is None else min(kv_len, q_len + window - 1))


def merged_length(intervals: list[tuple[int, int]]) -> int:
    """Size of the union of half-open intervals [a, b)."""
    total, end = 0, None
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if end is None or a >= end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


@lru_cache(maxsize=None)
def contract_window(task_id: str) -> int | None:
    """The sliding window of a task: ``sliding window N`` in its contract description and/or ``swaN``
    in its task ID (both must agree when both are present), or None when the task has no window.

    Without either, any other sign of a window (the word in the description, a ``WINDOW``
    constant or "last N tokens/keys" wording in the reference) raises, so the case stays
    unestimated instead of counting unwindowed pairs, which would over-count.
    """
    import yaml
    task = TASKS_DIR / task_id
    raw = yaml.safe_load((task / "task.yaml").read_text())
    described = re.search(r"sliding window (\d+)", raw["description"])
    named = re.search(r"swa(\d+)", task_id)
    windows = {int(m.group(1)) for m in (described, named) if m}
    if len(windows) > 1:
        raise ValueError(f"{task_id}: the task ID and the description state different windows {sorted(windows)}")
    if windows:
        return windows.pop()
    reference = (task / raw["reference"]["file"]).read_text()
    if ("window" in raw["description"].lower() or re.search(r"^\s*WINDOW\s*=", reference, re.M)
            or re.search(r"last \w+ (tokens|keys)", reference)):
        raise ValueError(f"{task_id}: a window is implied but neither 'sliding window N' nor 'swaN' states it")
    return None


def _unified_attention(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Causal (bottom-right aligned) paged attention, optionally windowed.

    Sequence s has q_s = cu_seqlens_q[s+1] - cu_seqlens_q[s] query rows and L_s = seqused_k[s] keys;
    each query head needs 4*d FLOPs per visible pair (``bottom_right_pairs``). Each KV head's keys
    and values at positions some row sees (``bottom_right_keys``) are read once; page slots past
    L_s, positions older than every window and block-table entries are not counted; q is read and
    out written whole. The window is the one the contract description states (``contract_window``).
    """
    starts, lengths = _ints(inputs["cu_seqlens_q"]), _ints(inputs["seqused_k"])
    window = contract_window(shapes.task_id)
    q_heads, kv_heads, dim = (shapes.value(n) for n in ("num_q_heads", "num_kv_heads", "head_dim"))
    pairs = keys = 0
    for seq, kv_len in enumerate(lengths):
        q_len = starts[seq + 1] - starts[seq]
        pairs += bottom_right_pairs(q_len, kv_len, window)
        keys += bottom_right_keys(q_len, kv_len, window)
    cache = "kv_cache" if "kv_cache" in {t.name for t in shapes.inputs} else "k_cache"
    notes = ["data_dependent: generated sequence lengths"] + ([f"contract_window:{window}"] if window else [])
    result = Estimate(notes=notes)
    add_product(shapes, result, mm("1", "1", "1", "q", cache), 4 * dim * q_heads * pairs)
    result.bytes_override = (keys * kv_heads * 2 * dim * _element(shapes, cache) + shapes.tensor("q").nbytes
                             + sum(t.nbytes for t in shapes.outputs))
    return result


def _batch_prefill_paged(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Paged extend attention, bottom-right causal. Sequence b has qo_b query rows (cu_seqlens_q) and
    kv_b = (pages_b - 1) * page_size + kv_last_page_lens[b] keys (kv_indptr counts pages): 4*d FLOPs
    per query head and visible pair. Each KV head's kv_b keys and values are read once (unused slots
    of a last page and unused pages excluded); q is read and out written whole."""
    starts, page_starts = _ints(inputs["cu_seqlens_q"]), _ints(inputs["kv_indptr"])
    last = _ints(inputs["kv_last_page_lens"])
    page_size = shapes.value("page_size")
    q_heads, kv_heads, dim = (shapes.value(n) for n in ("num_q_heads", "num_kv_heads", "head_size"))
    pairs = keys = 0
    for b, last_len in enumerate(last):
        pages = page_starts[b + 1] - page_starts[b]
        kv_len = (pages - 1) * page_size + last_len if pages > 0 else 0
        q_len = starts[b + 1] - starts[b]
        pairs += bottom_right_pairs(q_len, kv_len)
        keys += bottom_right_keys(q_len, kv_len)
    result = Estimate(notes=["data_dependent: generated sequence lengths"])
    add_product(shapes, result, mm("1", "1", "1", "q", "k_cache"), 4 * dim * q_heads * pairs)
    kv_bytes = keys * kv_heads * dim * (_element(shapes, "k_cache") + _element(shapes, "v_cache"))
    result.bytes_override = kv_bytes + shapes.tensor("q").nbytes + sum(t.nbytes for t in shapes.outputs)
    return result


def _prefix_prefill(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Prefix-prefill: sequence b has q_b new tokens (b_start_loc) after c_b = b_seq_len[b] - q_b cached
    ones; new token i sees the c_b cached keys and new keys 0..i, q_b*c_b + q_b(q_b+1)/2 pairs at
    4*d FLOPs per query head. Bytes: the new q, k, v whole (the last new token sees every new key),
    each KV head's c_b cached keys and values (unused slots of a last page excluded), the ALiBi
    slopes if declared, and out."""
    starts, totals = _ints(inputs["b_start_loc"]), _ints(inputs["b_seq_len"])
    q_heads, dim = shapes.tensor("q").shape[1], shapes.tensor("q").shape[2]
    kv_heads = shapes.tensor("k").shape[1]
    pairs = cached = 0
    for b, total in enumerate(totals):
        q_len = starts[b + 1] - starts[b]
        pairs += bottom_right_pairs(q_len, total)
        cached += max(0, total - q_len)
    result = Estimate(notes=["data_dependent: generated sequence lengths"])
    add_product(shapes, result, mm("1", "1", "1", "q", "k_cache"), 4 * dim * q_heads * pairs)
    names = {t.name for t in shapes.inputs}
    dense = sum(shapes.tensor(n).nbytes for n in ("q", "k", "v", "alibi_slopes") if n in names)
    cache_bytes = cached * kv_heads * dim * (_element(shapes, "k_cache") + _element(shapes, "v_cache"))
    result.bytes_override = dense + cache_bytes + sum(t.nbytes for t in shapes.outputs)
    return result


LEAN_BLOCK = 16 # tokens per block of the lean-attention block tables (fixed by the contract)


def _lean_paged_decode(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Unmasked decode with per-head block tables: request r owns entries [ends[r-1], ends[r]) of every
    head's table and query_rows / batch query rows. Per head and request the distinct listed blocks
    give the keys, 4*d FLOPs per query row and key. Bytes: every head's distinct blocks of keys and
    values over the used table prefix, plus q and out."""
    ends = _ints(inputs["batch_num_block_n"])
    table = inputs["kv_block_tables"].tolist()
    _, rows, dim = shapes.tensor("q").shape
    rows_per_request = rows // max(1, len(ends))
    pairs = blocks = 0
    for head_table in table:
        start = 0
        for end in ends:
            pairs += rows_per_request * LEAN_BLOCK * len(set(head_table[start:end]))
            start = end
        blocks += len(set(head_table[:ends[-1]])) if ends else 0
    result = Estimate(notes=["data_dependent: generated block tables"])
    add_product(shapes, result, mm("1", "1", "1", "q", "k"), 4 * dim * pairs)
    kv_bytes = blocks * LEAN_BLOCK * dim * (_element(shapes, "k") + _element(shapes, "v"))
    result.bytes_override = kv_bytes + shapes.tensor("q").nbytes + sum(t.nbytes for t in shapes.outputs)
    return result


def hstu_pairs(length: int, contextual: int, targets: int) -> int:
    """Visible pairs of one HSTU sequence under the contract's position rule.

    ids_i = clamp(i - contextual + 1, min=0) when contextual > 0 (else i), then clamp(max=max_ids), with
    max_ids = length - targets (- contextual + 1 when contextual > 0). Row i sees j == i and every j
    with ids_j < ids_i; with contextual > 0 a row with ids_i == 0 also sees every j with ids_j < max_ids.
    """
    max_ids = length - targets - (contextual - 1 if contextual > 0 else 0)
    ids = [min(max(i - contextual + 1, 0) if contextual > 0 else i, max_ids) for i in range(length)]
    first: dict[int, int] = {}
    for i, value in enumerate(ids):  # ids never decrease, so a value's first index counts the ids below it
        first.setdefault(value, i)
    below_max = sum(1 for value in ids if value < max_ids)
    total = 0
    for value in ids:
        if contextual > 0 and value == 0:
            total += below_max + (0 if value < max_ids else 1)  # {i} united with {j : ids_j < max_ids}
        else:
            total += first[value] + 1
    return total


def _hstu(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """HSTU (pointwise SiLU) attention over a jagged batch: 4*d FLOPs per head and visible pair
    (``hstu_pairs``); masked pairs are the only work a kernel may skip. Every token's q, k and v are
    used (each row sees itself), so they are read whole and out is written whole; the sort order
    only schedules work and is not counted."""
    offsets, targets = _ints(inputs["seq_offsets"]), _ints(inputs["num_targets"])
    contextual = int(shapes.scalars.get("contextual_seq_len", 0))
    heads, dim = shapes.value("n_heads"), shapes.value("head_dim")
    pairs = sum(hstu_pairs(offsets[b + 1] - offsets[b], contextual, targets[b]) for b in range(len(targets)))
    result = Estimate(notes=["data_dependent: generated sequence lengths"])
    add_product(shapes, result, mm("1", "1", "1", "q", "k"), 4 * dim * heads * pairs)
    result.bytes_override = (sum(shapes.tensor(n).nbytes for n in ("q", "k", "v"))
                             + sum(t.nbytes for t in shapes.outputs))
    return result


def _paged_mqa_logits(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Indexer logits over a paged key cache. Query row n of sequence b (next_n rows per sequence,
    L_b = context_lens[b] keys) scores keys 0 .. L_b - next_n + n: 2*d FLOPs per head and scored pair
    (one product, no PV). The key codes have the query's format (fp8 codes in a uint8 container, or
    packed fp4), so the product is charged at the query's precision. Bytes: every key some row
    scores (L_b per sequence) with its scale bytes, the queries and head weights of rows that score
    a key (query scales too unless their extent is padded), and the whole logits output (its -inf
    columns included: the evaluator compares whole rows)."""
    lengths, next_n = _ints(inputs["context_lens"]), shapes.value("next_n")
    heads, dim = shapes.value("num_heads"), shapes.value("head_size")
    names = {t.name for t in shapes.inputs}
    query = "q" if "q" in names else "q_fp4"
    slots = shapes.value("num_blocks") * shapes.value("block_size")
    per_key = sum(shapes.tensor(n).nbytes for n in ("kv_cache", "kv_scale") if n in names) // slots
    pairs = keys = live = 0
    for length in lengths:
        visible = [max(0, min(length, length - next_n + n + 1)) for n in range(next_n)]
        pairs += sum(visible)
        keys += max(visible)
        live += sum(1 for v in visible if v > 0)
    row_tensors = [query, "weights"]
    if "q_scale" in names and not shapes.tensor("q_scale").maybe_padded:
        row_tensors.append("q_scale")
    row_bytes = sum(shapes.tensor(n).nbytes for n in row_tensors) // shapes.value("num_rows")
    result = Estimate(notes=["data_dependent: generated context lengths"])
    add_product(shapes, result, mm("1", "1", "1", query, query), 2 * dim * heads * pairs)
    result.bytes_override = keys * per_key + live * row_bytes + sum(t.nbytes for t in shapes.outputs)
    return result


def _fp8_mqa_logits(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Indexer logits with per-row key windows [ks_i, ke_i) clipped to the keys: 2*d FLOPs per head and
    windowed pair. Bytes: the keys and key scales in the union of the windows, the queries and head
    weights of rows with a non-empty window, and the whole logits output."""
    kv_len = shapes.tensor("kv").shape[0]
    windows = [(max(0, a), min(kv_len, b))
               for a, b in zip(_ints(inputs["cu_seqlen_ks"]), _ints(inputs["cu_seqlen_ke"]))]
    heads, dim = shapes.value("num_heads"), shapes.value("head_size")
    pairs = sum(max(0, b - a) for a, b in windows)
    live = sum(1 for a, b in windows if b > a)
    per_key = _per_row(shapes.tensor("kv")) + _per_row(shapes.tensor("kv_scales"))
    per_row = _per_row(shapes.tensor("q")) + _per_row(shapes.tensor("weights"))
    result = Estimate(notes=["data_dependent: generated key windows"])
    add_product(shapes, result, mm("1", "1", "1", "q", "kv"), 2 * dim * heads * pairs)
    result.bytes_override = (merged_length(windows) * per_key + live * per_row
                             + sum(t.nbytes for t in shapes.outputs))
    return result


def _sparse_estimate(shapes: CaseShapes, pool: str, selected: list[int], pool_bytes: int, d_qk: int, d_v: int,
                     heads: int, extra_inputs: tuple[str, ...]) -> Estimate:
    """Shared tail of the sparse models: ``selected`` holds the distinct key count of every query row
    (rows of ``q``), ``pool_bytes`` the bytes of the distinct pool rows used by the whole batch."""
    result = Estimate(notes=["data_dependent: generated sparse indices"])
    add_product(shapes, result, mm("1", "1", "1", "q", pool), 2 * (d_qk + d_v) * heads * sum(selected))
    names = {t.name for t in shapes.inputs}
    q_row = shapes.tensor("q").nbytes // max(1, len(selected))
    result.bytes_override = (pool_bytes + sum(1 for n in selected if n) * q_row
                             + sum(shapes.tensor(n).nbytes for n in extra_inputs if n in names)
                             + sum(t.nbytes for t in shapes.outputs))
    return result


def _sparse_mla_prefill(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Ragged CSR top-k attention over latent rows that are both key and value (d_qk = d_v = head_dim):
    query i uses the distinct entries of indices[indptr[i]:indptr[i+1]] inside [0, num_kv). Bytes:
    each distinct used latent row once, the queries of rows that select something, the sink logits
    if declared, and out."""
    bounds, flat = _ints(inputs["indptr"]), _ints(inputs["indices"])
    kv = shapes.tensor("kv")
    rows, used = [], set()
    for a, b in zip(bounds, bounds[1:]):
        chosen = {s for s in flat[a:b] if 0 <= s < kv.shape[0]}
        rows.append(len(chosen))
        used |= chosen
    dim = shapes.value("head_dim")
    return _sparse_estimate(shapes, "kv", rows, len(used) * _per_row(kv), dim, dim, shapes.value("num_heads"),
                            ("attn_sink",))


def _sparse_mla_attention(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Top-k MLA over a paged latent cache: token t uses the distinct entries of topk_indices[t] inside
    the pool; QKᵀ spans the full head_size row, PV its first kv_lora_rank values. Bytes: each distinct
    used cache row (head_size values) once, the queries of tokens that select something, and out."""
    kv = shapes.tensor("kv")
    capacity = kv.shape[0] * kv.shape[1]
    rows, used = [], set()
    for row in inputs["topk_indices"].tolist():
        chosen = {s for s in row if 0 <= s < capacity}
        rows.append(len(chosen))
        used |= chosen
    return _sparse_estimate(shapes, "kv", rows, len(used) * (kv.nbytes // capacity), shapes.value("head_size"),
                            shapes.value("kv_lora_rank"), shapes.value("num_heads"), ())


def _flash_mla_sparse_decode(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Sparse decode over one or two packed FP8 latent pools; a slot's 512-wide row is both K and V.

    Query row (b, t) uses the distinct nonnegative in-pool slots among the first lens[b] entries of
    each pool's index table; entries of both pools add up. Each distinct slot of each pool is read
    once per batch, at its record size minus the scale record's pad byte (576 payload bytes and 7
    exponents of 584). The products are charged at fp8, the cache codes' precision (the bf16 rope
    tail and the bf16 query are not cheaper). Queries of rows without entries are not read; the
    sinks are read and output and lse written whole.
    """
    names = {t.name for t in shapes.inputs}
    query_length = shapes.value("query_length")
    rows: list[int] = []
    pool_bytes = 0
    for prefix in ("", "extra_"):
        pool = prefix + "kv_cache"
        if pool not in names:
            continue
        cache = shapes.tensor(pool)
        capacity = cache.shape[0] * cache.shape[1]
        record = cache.nbytes // capacity
        lengths = _ints(inputs[prefix + "sparse_lens"])
        table = inputs[prefix + "sparse_indices"].tolist()
        counts, used = [], set()
        for b, length in enumerate(lengths):
            for t in range(query_length):
                chosen = {s for s in table[b][t][:length] if 0 <= s < capacity}
                counts.append(len(chosen))
                used |= chosen
        rows = counts if not rows else [a + c for a, c in zip(rows, counts)]
        pool_bytes += len(used) * (record - 1)
    dim = shapes.value("head_dim")
    return _sparse_estimate(shapes, "kv_cache", rows, pool_bytes, dim, dim, shapes.value("heads"), ("sinks",))


for _operator, _model in {
    "unified_attention": _unified_attention,
    "batch_prefill_paged": _batch_prefill_paged,
    "prefix_prefill_attention": _prefix_prefill,
    # The ALiBi bias changes the scores, not which pairs are visible: every causal pair is counted.
    "prefix_prefill_attention_alibi": _prefix_prefill,
    "lean_attention_paged_decode": _lean_paged_decode,
    "hstu_attention": _hstu,
    "paged_mqa_logits": _paged_mqa_logits,
    "fp8_mqa_logits": _fp8_mqa_logits,
    "sparse_mla_prefill": _sparse_mla_prefill,
    "sparse_mla_attention": _sparse_mla_attention,
    "flash_mla_sparse_decode_fp8": _flash_mla_sparse_decode,
}.items():
    register_data_model(_operator, f"attention_data/{_operator}@v1", _model)
