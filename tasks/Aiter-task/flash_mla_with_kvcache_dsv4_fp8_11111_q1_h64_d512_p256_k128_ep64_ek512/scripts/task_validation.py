"""Independent known answers and comparator controls for task validation only.

These small fixtures are validation evidence, never additional scored cases.
They do not call the production baseline, initializer or candidate. Expected
answers are literal FP8/BF16 format facts and scalar Python attention, not
outputs of task_reference.
"""
from __future__ import annotations

import math

import torch

PAYLOAD_BYTES, NOPE_DIMS, SCALE_BYTES, RECORD_BYTES = 576, 448, 8, 584
# E4M3FN codes: 0x38 = 1, 0x40 = 2, 0xB8 = -1, 0x30 = 0.5. A scale byte of 127
# leaves its 64-dimension group unscaled; 128 doubles and 126 halves it.
ONE, TWO, MINUS_ONE, HALF = 0x38, 0x40, 0xB8, 0x30


def _decoded_code(code, scale):
    """The format's literal BF16 assembly of one E4M3 code under its scale byte.

    sign = code bit 7, exponent = code bits 3..6 + scale - 7, mantissa = code
    bits 0..2 as the top BF16 mantissa bits. A zero code therefore decodes to
    2**(scale - 134), not to zero.
    """
    exponent = ((code >> 3) & 15) + scale - 7
    value = 2.0 ** (exponent - 127) * (1 + (code & 7) / 8)
    return -value if code & 128 else value


def _decoded(record):
    """Full 512-dimension decoded vector of a packed slot record."""
    scales = record.get("scales", [127] * 7)
    nope = record.get("nope", {})
    values = [_decoded_code(nope.get(dim, 0), scales[dim // 64]) for dim in range(NOPE_DIMS)]
    rope = record.get("rope", {})
    return values + [float(rope.get(dim, 0.0)) for dim in range(64)]


def _pack(slots, page_size, device):
    """Pack slots as pages of payloads followed by their scale slots.

    ``slots`` holds per slot ``nope`` {dim: E4M3 code}, ``scales`` (seven
    bytes, one per 64 nope dimensions) and ``rope`` {dim: BF16 value}.
    """
    pages = -(-len(slots) // page_size)
    raw = torch.zeros((pages, page_size * RECORD_BYTES), dtype=torch.uint8)
    for slot, record in enumerate(slots):
        page, within = divmod(slot, page_size)
        payload = within * PAYLOAD_BYTES
        for dim, code in record.get("nope", {}).items():
            raw[page, payload + dim] = code
        for dim, value in record.get("rope", {}).items():
            raw[page, payload + NOPE_DIMS + 2 * dim:payload + NOPE_DIMS + 2 * dim + 2] = (
                torch.tensor([value], dtype=torch.bfloat16).view(torch.uint8))
        scales = record.get("scales", [127] * 7)
        start = page_size * PAYLOAD_BYTES + within * SCALE_BYTES
        raw[page, start:start + 7] = torch.tensor(scales, dtype=torch.uint8)
    return raw.view(pages, page_size, 1, RECORD_BYTES).view(torch.float8_e4m3fn).to(device)


def decode_known_answer(reference, device):
    record = {"nope": {0: ONE, 64: TWO, 128: MINUS_ONE, 447: HALF},
              "scales": [127, 128, 126, 127, 127, 127, 125], "rope": {0: 1.5, 63: -2.0}}
    cache = _pack([{}, record], page_size=2, device=device)
    expected = torch.tensor(_decoded(record))
    # Literal facts, independent of the assembly formula: scaled codes and an
    # unset code under its group's scale.
    literal = {0: 1.0, 64: 4.0, 128: -0.5, 447: 0.125, 1: 2.0**-7, 65: 2.0**-6, 446: 2.0**-9,
               448: 1.5, 511: -2.0, 449: 0.0}
    if any(float(expected[d]) != v for d, v in literal.items()):
        raise AssertionError("scalar decode oracle disagrees with literal format facts")
    got = reference._decode_slots(cache, torch.tensor([1], device=device))
    torch.testing.assert_close(got.cpu(), expected[None], rtol=0, atol=0)
    return {"oracle": "literal E4M3 BF16 assembly with per-64-dimension exponent scales and BF16 rope",
            "checked_dims": {str(d): v for d, v in literal.items()}}


def _attention(vectors, query, scale, sink):
    """Shared K=V scalar attention with the sink in the softmax denominator only."""
    if not vectors:
        return [0.0] * 512, math.inf
    logits = [sum(q * k for q, k in zip(query, vector)) * scale for vector in vectors]
    weights = [math.exp(v) for v in logits]
    total = sum(weights)
    denominator = total + math.exp(sink)
    output = [sum(w * vector[d] for w, vector in zip(weights, vectors)) / denominator for d in range(512)]
    return output, math.log(total)


def attention_known_answer(reference, inputs_module, device):
    has_extra = "extra" in inputs_module.POOL_NAMES
    # Main pool slots 0..3 hold 1, 2, -1, 0.5 in dimension 0; extra pool slots
    # 0..1 hold 4 (code 2 under a doubled scale) and -1, with a halved second
    # group. Unset codes decode to small nonzero values under their scales.
    main_records = [{"nope": {0: code}, "rope": {5: 0.25 * (i + 1)}}
                    for i, code in enumerate((ONE, TWO, MINUS_ONE, HALF))]
    extra_records = [{"nope": {0: TWO}, "scales": [128, 126] + [127] * 5},
                     {"nope": {0: MINUS_ONE}, "rope": {7: -1.0}}]
    main_values = [_decoded(r) for r in main_records]
    extra_values = [_decoded(r) for r in extra_records]
    main = _pack(main_records, 2, device)
    extra = _pack(extra_records, 2, device)
    # Rows: main only; a -1 inside the main prefix; legal tails past both
    # prefixes; both pools empty; both pools contributing; a short history
    # whose one extra entry is -1.
    main_rows = [([0, 1, -1, -1], 2), ([0, -1, 2, -1], 3), ([3, 0, 1, 2], 1),
                 ([-1, -1, -1, -1], 0), ([1, 2, -1, -1], 2), ([0, 3, -1, -1], 2)]
    extra_rows = [([-1] * 4, 0), ([-1] * 4, 0), ([1, 0, 1, 0], 0),
                  ([-1] * 4, 0), ([0, 1, -1, -1], 2), ([-1, 0, 0, 0], 1)]
    queries, sinks, scale = [1.0, -2.0], [0.0, 1.0], 0.5
    batch, heads = len(main_rows), len(queries)
    q = torch.zeros((batch, 1, heads, 512), dtype=torch.bfloat16)
    q[:, 0, :, 0] = torch.tensor(queries, dtype=torch.bfloat16)
    operands = {
        "q": q.to(device), "kv_cache": main,
        "sparse_indices": torch.tensor([[r[0]] for r in main_rows], dtype=torch.int32, device=device),
        "sparse_lens": torch.tensor([r[1] for r in main_rows], dtype=torch.int32, device=device),
        "sm_scale": scale, "sinks": torch.tensor(sinks, dtype=torch.float32, device=device),
    }
    if has_extra:
        operands.update(
            extra_kv_cache=extra,
            extra_sparse_indices=torch.tensor([[r[0]] for r in extra_rows], dtype=torch.int32, device=device),
            extra_sparse_lens=torch.tensor([r[1] for r in extra_rows], dtype=torch.int32, device=device))
    expected_output = torch.zeros((batch, 1, heads, 512))
    expected_lse = torch.zeros((batch, 1, heads))
    selected = []
    for row in range(batch):
        chosen = [("main", i) for i in main_rows[row][0][:main_rows[row][1]] if i >= 0]
        if has_extra:
            chosen += [("extra", i) for i in extra_rows[row][0][:extra_rows[row][1]] if i >= 0]
        selected.append(chosen)
        vectors = [(main_values if pool == "main" else extra_values)[i] for pool, i in chosen]
        for head in range(heads):
            query = [queries[head]] + [0.0] * 511
            value, lse = _attention(vectors, query, scale, sinks[head])
            expected_output[row, 0, head] = torch.tensor(value)
            expected_lse[row, 0, head] = lse
    output, lse = inputs_module.named_outputs(
        reference.run(**{name: operands[name] for name in inputs_module.INPUTS}))
    # Bounds cover BF16 output rounding and FP32 arithmetic of this fixture only.
    torch.testing.assert_close(output.float().cpu(), expected_output, rtol=2**-7, atol=1e-6)
    if not torch.equal(torch.isposinf(lse.cpu()), torch.isposinf(expected_lse)):
        raise AssertionError("LSE +inf rows differ from the empty rows")
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(lse.cpu()[finite], expected_lse[finite], rtol=1e-5, atol=1e-5)
    return {"oracle": "scalar Python softmax with sink over literally decoded shared K/V vectors",
            "selected_slots": [[f"{pool}:{i}" for pool, i in row] for row in selected],
            "expected_output_dim0": expected_output[:, 0, :, 0].tolist(),
            "observed_output_dim0": output[:, 0, :, 0].float().cpu().tolist(),
            "expected_lse": [[v if math.isfinite(v) else "inf" for v in row]
                             for row in expected_lse[:, 0].tolist()]}


def comparator_controls(compare, device):
    output = torch.ones((2, 1, 2, 4), dtype=torch.bfloat16, device=device)
    output[1] = 0
    lse = torch.tensor([[[0.5, 0.25]], [[math.inf, math.inf]]], dtype=torch.float32, device=device)
    expected = (output, lse)
    near_output, near_lse = output.clone(), lse.clone()
    near_output[0] = 1.0078125
    near_lse[0] += 5e-4
    for accepted in ((output.clone(), lse.clone()), (near_output, near_lse)):
        compare.run(accepted, expected)

    def changed(new_output=None, new_lse=None):
        return (output.clone() if new_output is None else new_output,
                lse.clone() if new_lse is None else new_lse)

    far_output, far_lse = output.clone(), lse.clone()
    far_output[0] = 1.03125
    far_lse[0] += 3e-3
    finite_empty, infinite_row = lse.clone(), lse.clone()
    finite_empty[1] = 0.0
    infinite_row[0] = math.inf
    rejected = {
        "output_outside_gate": changed(far_output),
        "lse_outside_gate": changed(new_lse=far_lse),
        "finite_lse_on_empty_row": changed(new_lse=finite_empty),
        "infinite_lse_on_nonempty_row": changed(new_lse=infinite_row),
        "nonfinite_output": changed(output * float("nan")),
        "nan_lse": changed(new_lse=lse * float("nan")),
        "wrong_output_dtype": changed(output.float()),
        "wrong_output_shape": changed(output[:, :, :1]),
        "missing_lse": output.clone(),
    }
    observations = {}
    for name, wrong in rejected.items():
        try:
            compare.run(wrong, expected)
        except AssertionError as error:
            observations[name] = str(error)
        else:
            raise RuntimeError(f"Comparator accepted negative control: {name}")
    return {"oracle": "BF16 output additive 1e-2, FP32 LSE additive 1e-3, exact +inf rows",
            "accepted_inexact": {"output": 1.0078125, "lse_offset": 5e-4},
            "rejected_controls": observations}


def run_controls(measure, *, device="cuda", records=None):
    records = [] if records is None else records
    checks = [("mla_fp8_decode", lambda: decode_known_answer(measure.task_reference, device)),
              ("mla_attention_known_answer",
               lambda: attention_known_answer(measure.task_reference, measure.task_inputs, device)),
              ("comparator_positive_and_negative", lambda: comparator_controls(measure.task_compare, device))]
    for name, check in checks:
        record = {"name": name, "device": str(device), "status": "FAIL"}
        records.append(record)
        try:
            record["evidence"] = check()
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
            raise RuntimeError(f"Task validation control failed: {name}: {error}") from error
        record["status"] = "PASS"
    return records
