"""Tier 3 models of non-attention operators whose work depends on the generated routing or lengths.

Each model counts, from the generated inputs, only the rows, slots and weights that the reference
uses, and only bytes a correct kernel cannot avoid: an input row is counted when some output value
depends on it, a cache or state slot when it must be read (an output depends on it) or rewritten
(it is compared and its value changes), and an index entry when the entry alone decides whether or
where a row goes. Outputs are compared whole, so they are always counted whole. Tiny per-head
parameters and metadata are left out; leaving bytes out only lowers the floor.

Where an entry is repeated (two tokens on one cache slot, one route listed twice), only its last
writer in the reference's order is counted.
"""
from __future__ import annotations

import math
from typing import Any

from ..generated import register_data_model
from ..interface import CaseShapes, Tensor
from . import Estimate
from .gemm import add_product, mm


def _flat(values) -> list[int]:
    out: list[int] = []
    stack = [values]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(reversed(value))
        else:
            out.append(int(value))
    return out


def _ints(tensor: Any) -> list[int]:
    return _flat(tensor.tolist())


def _rows(tensor: Any) -> list[list[int]]:
    return [_flat(row) if isinstance(row, list) else [int(row)] for row in tensor.tolist()]


def _float_rows(tensor: Any) -> list[list[float]]:
    """Each index of the leading axis flattened to floats."""
    rows = []
    for row in tensor.tolist():
        flat, stack = [], [row]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(reversed(value))
            else:
                flat.append(float(value))
        rows.append(flat)
    return rows


def _row_bytes(tensor: Tensor) -> int:
    """Bytes of one index of the leading axis."""
    return tensor.nbytes // max(1, tensor.shape[0])


def _outputs(shapes: CaseShapes) -> int:
    return sum(t.nbytes for t in shapes.outputs)


def _element(tensor: Tensor) -> int:
    count = 1
    for extent in tensor.shape:
        count *= extent
    return tensor.nbytes // max(1, count)


# --- routed GEMMs --------------------------------------------------------------------------------

def batched_moe_gemm(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """C[e, :n_e] = A[e, :n_e] B[e]ᵀ. FLOPs 2 * sum(n_e) * N * K. Bytes: the n_e valid rows of A, the
    weights of experts with n_e > 0 (an empty expert's B affects nothing), C whole (its padding rows
    are compared as zeros)."""
    a, b = shapes.tensor("A"), shapes.tensor("B")
    max_tokens, hidden = a.shape[1], a.shape[2]
    counts = [max(0, min(n, max_tokens)) for n in _ints(inputs["expert_num_tokens"])]
    rows, used = sum(counts), sum(1 for n in counts if n > 0)
    result = Estimate(notes=[f"data_dependent: {rows} valid rows, {used} of {len(counts)} experts used"])
    add_product(shapes, result, mm("1", "1", "1", "A", "B"), 2 * rows * b.shape[1] * hidden)
    result.bytes_override = (rows * hidden * _element(a) + used * _row_bytes(b)
                             + shapes.tensor("expert_num_tokens").nbytes + _outputs(shapes))
    return result


def fused_moe_lora(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """LoRA delta of the routes the reference writes (listed, enabled slots; blocks with expert >= 0;
    route ids < n_tokens * top_k). Each distinct (token, slot, expert) of a written route needs, per
    slice, the shrink x lora_aᵀ (2*H*R) and the expand (2*R*N); two routes of one token on the same
    pair would share both, so they are counted once. Bytes: the
    hidden rows of tokens with a written route, the lora_a and lora_b slices of every (slot, expert)
    pair that is the last writer of some route, the delta whole."""
    s = shapes.symbols
    top_k, block = s["top_k"], s["block_size"]
    lora_a, lora_b = shapes.tensor("lora_a"), shapes.tensor("lora_b")
    slices, max_loras = lora_a.shape[0], lora_a.shape[1]
    rank, hidden, inter = lora_a.shape[3], lora_a.shape[4], lora_b.shape[3]
    routes_total = shapes.tensor("hidden_states").shape[0] * top_k
    enabled = _ints(inputs["adapter_enabled"])
    padded = _ints(inputs["num_tokens_post_padded"])
    sorted_ids, expert_ids = _rows(inputs["sorted_token_ids"]), _rows(inputs["expert_ids"])
    writer: dict[int, tuple[int, int]] = {}
    for slot in sorted(set(_ints(inputs["lora_ids"]))):
        if slot < 0 or slot >= max_loras or enabled[slot] == 0:
            continue
        n_blocks = min(-(-padded[slot] // block), len(expert_ids[slot]))
        chosen = []
        for i in range(n_blocks * block):
            route, expert = sorted_ids[slot][i], expert_ids[slot][i // block]
            if 0 <= route < routes_total and expert >= 0:
                chosen.append((expert, route))
        for expert, route in sorted(chosen):  # the reference writes expert by expert, in ascending order
            writer[route] = (slot, expert)
    routes, pairs = len(writer), set(writer.values())
    tokens = len({route // top_k for route in writer})
    products = len({(route // top_k, pair) for route, pair in writer.items()})
    result = Estimate(notes=[f"data_dependent: {routes} routes, {len(pairs)} (slot, expert) pairs"])
    add_product(shapes, result, mm("1", "1", "1", "hidden_states", "lora_a"),
                products * slices * 2 * rank * (hidden + inter))
    per_pair = slices * rank * (hidden + inter) * _element(lora_a)
    result.bytes_override = (tokens * _row_bytes(shapes.tensor("hidden_states")) + len(pairs) * per_pair
                             + _outputs(shapes))
    return result


def _lora(weight: str):
    def estimate(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
        """out[t] = x[t] W[slot(t)]ᵀ for the tokens of groups with slot >= 0 (2 * R * H FLOPs each, the
        last group listing a token wins). Bytes: x rows of those tokens, the weights of slots that
        serve at least one of them, out whole (other rows are compared as zeros). The index tensors
        are not counted."""
        w = shapes.tensor(weight)
        slot_of: dict[int, int] = {}
        order = _ints(inputs["token_indices_sorted"])
        starts, counts = _ints(inputs["lora_token_start_loc"]), _ints(inputs["num_tokens_per_lora"])
        for group, slot in enumerate(_ints(inputs["lora_ids"])):
            if slot < 0:
                continue
            for token in order[starts[group]:starts[group] + counts[group]]:
                slot_of[token] = slot
        tokens, slots = len(slot_of), len(set(slot_of.values()))
        result = Estimate(notes=[f"data_dependent: {tokens} routed tokens, {slots} adapters used"])
        add_product(shapes, result, mm("1", "1", "1", "x", weight), 2 * tokens * w.shape[1] * w.shape[2])
        result.bytes_override = tokens * _row_bytes(shapes.tensor("x")) + slots * _row_bytes(w) + _outputs(shapes)
        return result
    return estimate


# --- gathers, masks and scatters -----------------------------------------------------------------

def ep_gather(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """out[t] = sum over valid k (id >= 0) of w[t, k] * input[index[t, k]]. Every id is read (it decides
    validity); for each valid assignment its weight and index are read and its input row (distinct
    per assignment; a repeated row is counted once); out is written whole. The weighted sum is
    vector work, ignored."""
    ids, index = _ints(inputs["recv_topk_ids"]), _ints(inputs["input_index"])
    valid = [i for i, expert in enumerate(ids) if expert >= 0]
    rows = len({index[i] for i in valid})
    result = Estimate(notes=[f"data_dependent: {len(valid)} of {len(ids)} assignments valid"])
    result.bytes_override = (rows * _row_bytes(shapes.tensor("input_tensor")) + shapes.tensor("recv_topk_ids").nbytes
                             + len(valid) * (_element(shapes.tensor("recv_topk_weight"))
                                             + _element(shapes.tensor("input_index")))
                             + _outputs(shapes))
    return result


def silu_and_mul_fp8_quant_masked(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Only the first tokens_per_expert[e] rows of each expert are activated; the others are written as
    zeros without reading y. Bytes: the valid rows of y, the counts, both outputs whole."""
    y = shapes.tensor("y")
    capacity = y.shape[1]
    rows = sum(max(0, min(n, capacity)) for n in _ints(inputs["tokens_per_expert"]))
    result = Estimate(notes=[f"data_dependent: {rows} of {y.shape[0] * capacity} rows valid"])
    result.bytes_override = (rows * (y.nbytes // (y.shape[0] * capacity))
                             + shapes.tensor("tokens_per_expert").nbytes + _outputs(shapes))
    return result


def reshape_and_cache(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """cache[slot[t]] = key[t] | value[t] for slot >= 0. Every slot entry is read (it decides whether and
    where token t goes); for every distinct written slot the last token's key and value rows are read
    and the cache row is written (key + value bytes each way). Slots no token writes are untouched."""
    slots = _ints(inputs["slot_mapping"])
    written = len({slot for slot in slots if slot >= 0})
    row = _row_bytes(shapes.tensor("key")) + _row_bytes(shapes.tensor("value"))
    result = Estimate(notes=[f"data_dependent: {written} of {len(slots)} tokens write a slot"])
    result.bytes_override = 2 * written * row + shapes.tensor("slot_mapping").nbytes + _outputs(shapes)
    return result


def fused_rope_reshape_cache(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """q_out, k_out = RoPE(q, k) for every token, then key_cache[slot[t]] = k_out[t] and
    value_cache[slot[t]] = v[t] for slot >= 0. Bytes: q, k, positions and every slot entry (it decides
    whether and where token t goes) whole, both outputs whole; for every distinct written slot the
    last writer's v row and the write of one key and one value cache slot (both are compared and
    change: they start as zeros and receive normal draws). v rows of padded tokens, and v rows
    overwritten by a later token on the same slot, are not counted. The cos/sin tables are not
    counted (a kernel may compute them from the position), nor the RoPE arithmetic."""
    v, key_cache = shapes.tensor("v"), shapes.tensor("key_cache")
    slot_bytes = key_cache.nbytes // max(1, key_cache.shape[0] * key_cache.shape[1])
    slots = _ints(inputs["slot_mapping"])
    written = len({slot for slot in slots if slot >= 0})
    whole = sum(shapes.tensor(name).nbytes for name in ("q", "k", "positions", "slot_mapping"))
    result = Estimate(notes=[f"data_dependent: {written} of {len(slots)} tokens write a cache slot"])
    result.bytes_override = (whole + written * (_row_bytes(v) + 2 * slot_bytes) + _outputs(shapes))
    return result


# --- recurrent state ------------------------------------------------------------------------------

# A (sequence, head) whose decay exp(g) is below this keeps less than 1e-4 of its old state, which is
# below the comparison tolerance, so a kernel may skip reading that state slice.
NEGLIGIBLE_DECAY = 1e-4
SOFTPLUS_THRESHOLD = 20.0


def _flat_values(tensor: Any) -> list[float]:
    values: list[float] = []
    stack = [tensor.tolist()]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(reversed(value))
        else:
            values.append(float(value))
    return values


def gdn_decays(a_rows: list[list[float]], dt_bias: list[float], a_log: list[float]) -> list[list[float]]:
    """exp(g) = exp(-exp(A_log[h]) * softplus(a[b, h] + dt_bias[h])) per sequence and value head."""
    rates = [math.exp(v) for v in a_log]
    decays = []
    for row in a_rows:
        out = []
        for h, value in enumerate(row):
            pre = value + dt_bias[h]
            softplus = math.log1p(math.exp(pre)) if pre <= SOFTPLUS_THRESHOLD else pre
            out.append(math.exp(-rates[h] * softplus))
        decays.append(out)
    return decays


def recurrent_decode(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """One gated-delta decode step per sequence on its pool slot; a negative slot marks padding (zero
    output, state untouched). Bytes: the slot index whole, the per-sequence activations (tensors with
    a leading batch axis) of unpadded sequences, out whole, and per distinct used slot:

    - the write of every head's state slice (compared, and every slice changes);
    - the read of a head's slice only when some unpadded sequence on that slot has decay
      exp(g) >= 1e-4: below that the old state contributes less than the tolerance, so a kernel may
      skip it. The decay is computed from the generated a, dt_bias and A_log.

    The per-slot matrix-vector work (about 4 FLOPs per float32 state element moved) is far below the
    vector-peak to bandwidth ratio and ignored."""
    index = next(t for t in shapes.inputs if t.role == "index")
    state = next(t for t in shapes.inputs if t.mutable)
    batch, heads = index.shape[0], state.shape[1]
    slots = _ints(inputs[index.name])
    live = [(row, slot) for row, slot in enumerate(slots) if slot >= 0]
    rows = sum(_row_bytes(t) for t in shapes.inputs
               if t.role == "activation" and not t.mutable and t.shape and t.shape[0] == batch)
    distinct = len({slot for _, slot in live})
    a_rows = _float_rows(inputs["a"])
    decays = gdn_decays(a_rows, _flat_values(inputs["dt_bias"]), _flat_values(inputs["A_log"]))
    read = {(slot, h) for row, slot in live for h in range(heads) if decays[row][h] >= NEGLIGIBLE_DECAY}
    slice_bytes = _row_bytes(state) // heads
    result = Estimate(notes=[f"data_dependent: {len(live)} of {batch} sequences unpadded",
                             f"data_dependent: {len(read)} of {distinct * heads} state slices read "
                             f"(decay >= {NEGLIGIBLE_DECAY:g})"])
    result.bytes_override = (index.nbytes + len(live) * rows + distinct * _row_bytes(state)
                             + len(read) * slice_bytes + _outputs(shapes))
    return result


def ssd_chunk_state_varlen(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Final state of sequence b = sum over its tokens in its last chunk (positions k0..L-1) of
    scale * x ⊗ B, plus a decayed carried state. Per such token every head needs a P x N outer
    product (2*H*P*N FLOPs, bf16 operands); the tokens of different sequences are distinct. Bytes:
    x and B rows, dt and dA_cumsum entries (one per head) of those tokens, one carried H x P x N
    state per sequence (chunk_states of its last chunk or its initial state; only one sequence can
    carry a given chunk's state), cu_seqlens, the states whole. The boundary dA entry is not counted.
    Known margin: a kernel could rebuild some dA_cumsum entries from dt and the other entries instead
    of reading them; that saving is below 1.5% of these bytes and is not subtracted."""
    s = shapes.symbols
    heads, head_dim, state_dim, size = s["num_heads"], s["head_dim"], s["state_dim"], s["chunk_size"]
    bounds = _ints(inputs["cu_seqlens"])
    tokens = 0
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        chunk_start = (end - 1) // size * size
        tokens += end - max(start, chunk_start)
    sequences = len(bounds) - 1
    result = Estimate(notes=[f"data_dependent: {tokens} tokens in the sequences' last chunks"])
    add_product(shapes, result, mm("1", "1", "1", "x", "B"), 2 * tokens * heads * head_dim * state_dim)
    per_token = (_row_bytes(shapes.tensor("x")) + _row_bytes(shapes.tensor("B"))
                 + heads * (_element(shapes.tensor("dt")) + _element(shapes.tensor("dA_cumsum"))))
    carried = heads * head_dim * state_dim * _element(shapes.tensor("initial_states"))
    result.bytes_override = (tokens * per_token + sequences * carried + shapes.tensor("cu_seqlens").nbytes
                             + _outputs(shapes))
    return result


# --- selection and reductions --------------------------------------------------------------------

def topk_transform_paged(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """Row b with L = seq_lens[b] valid scores: for L > k any unread score could be among the k largest,
    so all L are read; for L <= k the selection is 0..L-1 and no score is needed. The selected
    positions span at least ceil(min(L, k) / page_size) pages, whose page-table entries are read.
    seq_lens and the output are counted whole. The selection itself is not counted."""
    topk = shapes.symbols["topk"]
    page = int(shapes.scalars["page_size"])
    scores, pages = shapes.tensor("scores"), shapes.tensor("page_tables")
    lengths = _ints(inputs["seq_lens"])
    read = sum(length for length in lengths if length > topk)
    entries = sum(-(-min(length, topk) // page) for length in lengths if length > 0)
    result = Estimate(notes=[f"data_dependent: {read} scores in rows longer than k"])
    result.bytes_override = (read * _element(scores) + entries * _element(pages)
                             + shapes.tensor("seq_lens").nbytes + _outputs(shapes))
    return result


def cross_entropy_label_smoothing(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """A row whose target equals ignore_index (when ignore_index >= 0) has loss 0 and is left out of the
    mean, so its logits need not be read. Bytes: the logits of the other rows, every target (it
    decides whether a row is ignored), both outputs. The class weights are not counted."""
    logits, target = shapes.tensor("x"), shapes.tensor("target")
    ignore = int(shapes.scalars["ignore_index"])
    labels = _ints(inputs["target"])
    kept = sum(1 for label in labels if label != ignore) if ignore >= 0 else len(labels)
    result = Estimate(notes=[f"data_dependent: {kept} of {len(labels)} rows not ignored"])
    result.bytes_override = kept * _row_bytes(logits) + target.nbytes + _outputs(shapes)
    return result


def index_select_rows(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """out[i] = table[idx[i]]: a row selected several times is read from the table once. Bytes: the
    distinct selected rows, every index (it alone decides which row goes where) and the output."""
    table, idx = shapes.tensor("table"), shapes.tensor("idx")
    selected = _ints(inputs["idx"])
    distinct = len(set(selected))
    result = Estimate(notes=[f"data_dependent: {distinct} distinct rows among {len(selected)} selections"])
    result.bytes_override = distinct * _row_bytes(table) + idx.nbytes + _outputs(shapes)
    return result


def apply_temperature(shapes: CaseShapes, inputs: dict[str, Any]) -> Estimate:
    """logits[i] /= temperature[expanded_idx_mapping[i]] in place; the contract leaves rows whose
    temperature is exactly 0 or 1 unmodified. Bytes: each modified row read and rewritten, every
    mapping entry, and the temperatures the mapping names."""
    logits, mapping = shapes.tensor("logits"), shapes.tensor("expanded_idx_mapping")
    requests = _ints(inputs["expanded_idx_mapping"])
    temperatures = [float(value) for value in inputs["temperature"].tolist()]
    modified = sum(1 for request in requests if temperatures[request] not in (0.0, 1.0))
    result = Estimate(notes=[f"data_dependent: {modified} of {len(requests)} rows modified"])
    result.bytes_override = (2 * modified * _row_bytes(logits) + mapping.nbytes
                             + len(set(requests)) * _element(shapes.tensor("temperature")))
    return result


for _operator, _model in {
    "batched_moe_gemm": batched_moe_gemm,
    "fused_moe_lora": fused_moe_lora,
    "lora_expand": _lora("lora_b"),
    "lora_shrink": _lora("lora_a"),
    "ep_gather": ep_gather,
    "silu_and_mul_fp8_quant_masked": silu_and_mul_fp8_quant_masked,
    "reshape_and_cache_flash": reshape_and_cache,
    "reshape_and_cache_flash_diffkv": reshape_and_cache,
    "fused_rope_reshape_cache": fused_rope_reshape_cache,
    "gdn_fused_recurrent_decode": recurrent_decode,
    "ssd_chunk_state_varlen": ssd_chunk_state_varlen,
    "topk_transform_paged": topk_transform_paged,
    "cross_entropy_label_smoothing": cross_entropy_label_smoothing,
    "index_select_rows": index_select_rows,
    "apply_temperature": apply_temperature,
}.items():
    register_data_model(_operator, f"partial_data/{_operator}@v1", _model)
