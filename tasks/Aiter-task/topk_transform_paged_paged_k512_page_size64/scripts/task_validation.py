"""Independent known answers and comparator controls for task validation only.

These small fixtures are validation evidence, never additional scored cases.
They do not call the production baseline, initializer or candidate. Expected
answers come from scalar Python selection and page mapping written from the
definition, not from outputs of task_reference.
"""
from __future__ import annotations

import torch


def _oracle(scores, lengths, tables, page_size, k):
    """Short rows keep positions 0..len-1 in order; long rows take the k best scores."""
    rows = []
    for values, length, table in zip(scores, lengths, tables):
        if length <= k:
            positions = list(range(length))
        else:
            positions = sorted(range(length), key=lambda p: values[p], reverse=True)[:k]
        slots = [table[p // page_size] * page_size + p % page_size for p in positions]
        rows.append(slots + [-1] * (k - len(slots)))
    return rows


def _fixture(batch, width, pages, page_size, lengths, score_stride, table_stride):
    # Coprime strides make each score row a permutation (tie-free) and each
    # page-table row a permutation of the pages, both differing by row.
    scores = [[((score_stride * j + 3 * r) % width) / width for j in range(width)] for r in range(batch)]
    tables = [[(table_stride * p + r) % pages for p in range(pages)] for r in range(batch)]
    return scores, tables


def _operands(scores, lengths, tables, page_size, device):
    batch = len(scores)
    metadata = torch.zeros((batch + 1, 2), dtype=torch.int32, device=device)
    metadata[0, 0] = 2**31 - 1
    return {"scores": torch.tensor(scores, dtype=torch.float32, device=device),
            "seq_lens": torch.tensor(lengths, dtype=torch.int32, device=device),
            "metadata": metadata, "page_size": page_size,
            "page_tables": torch.tensor(tables, dtype=torch.int32, device=device)}


def _assert_selection(got, expected, lengths, k):
    """Short rows exactly; long rows as sets with no padding."""
    for row, length in enumerate(lengths):
        observed, wanted = got[row].tolist(), expected[row]
        if length <= k:
            if observed != wanted:
                raise AssertionError(f"row {row} (length {length}): {observed} != {wanted}")
        elif sorted(observed) != sorted(wanted) or -1 in observed:
            raise AssertionError(f"row {row} (length {length}): selected set differs")


def small_known_answer(reference, device):
    # Lengths cover empty, short, exactly k, and long rows that span pages.
    k, page_size, width, pages = 4, 4, 16, 4
    lengths = [0, 3, 4, 11]
    scores, tables = _fixture(4, width, pages, page_size, lengths, score_stride=7, table_stride=3)
    expected = _oracle(scores, lengths, tables, page_size, k)
    got = reference._paged_reference(**_operands(scores, lengths, tables, page_size, device), k=k)
    _assert_selection(got, expected, lengths, k)
    return {"oracle": "scalar Python prefix order / top-k by score, then page-table slot mapping",
            "k": k, "page_size": page_size, "lengths": lengths,
            "expected": expected, "observed": got.cpu().tolist()}


def production_known_answer(reference, inputs_module, device):
    # The bundle's bound callback at the task's k and page size, around the
    # 512-slot boundary: one row exactly k long, one longer.
    k, page_size = inputs_module.K, inputs_module.PAGE_SIZE
    width, pages = 1024, 16
    lengths = [k, k + 188]
    scores, tables = _fixture(2, width, pages, page_size, lengths, score_stride=37, table_stride=5)
    expected = _oracle(scores, lengths, tables, page_size, k)
    got = reference.run(**_operands(scores, lengths, tables, page_size, device))
    _assert_selection(got, expected, lengths, k)
    return {"oracle": "scalar Python selection at the bound k and page size",
            "k": k, "page_size": page_size, "lengths": lengths,
            "observed_first_slots": [row[:8] for row in got.cpu().tolist()]}


def comparator_controls(compare, device):
    # The small fixture's answer: empty row, short row with padding, exactly-k
    # row, and a long row whose selection order is unspecified.
    k, page_size, width, pages = 4, 4, 16, 4
    lengths = [0, 3, 4, 11]
    scores, tables = _fixture(4, width, pages, page_size, lengths, score_stride=7, table_stride=3)
    rows = _oracle(scores, lengths, tables, page_size, k)
    expected = torch.tensor(rows, dtype=torch.int32, device=device)
    accepted = {"exact": expected.clone(), "long_row_reordered": expected.clone()}
    accepted["long_row_reordered"][3] = expected[3].flip(0)
    for name, value in accepted.items():
        compare.run(value, expected)
    unselected = next(table * page_size + offset for table in range(pages) for offset in range(page_size)
                      if table * page_size + offset not in rows[3])
    rejected = {name: expected.clone() for name in (
        "wrong_long_selection", "short_row_reordered", "padding_moved", "missing_padding",
        "duplicated_slot")}
    rejected["wrong_long_selection"][3, 0] = unselected
    rejected["short_row_reordered"][1, :2] = expected[1, :2].flip(0)
    rejected["padding_moved"][1] = torch.tensor([-1, *rows[1][1:3], rows[1][0]], dtype=torch.int32)
    rejected["missing_padding"][0, 0] = rows[2][0]
    rejected["duplicated_slot"][3, 1] = expected[3, 0]
    rejected["wrong_dtype"] = expected.long()
    rejected["wrong_shape"] = expected[:, :k - 1]
    observations = {}
    for name, wrong in rejected.items():
        try:
            compare.run(wrong, expected)
        except AssertionError as error:
            observations[name] = str(error)
        else:
            raise RuntimeError(f"Comparator accepted negative control: {name}")
    return {"oracle": "exact per-row sets, reference padding positions, fixed short-row order",
            "accepted_controls": sorted(accepted), "rejected_controls": observations}


def run_controls(measure, *, device="cuda", records=None):
    records = [] if records is None else records
    checks = [("topk_small_known_answer", lambda: small_known_answer(measure.task_reference, device)),
              ("topk_production_known_answer",
               lambda: production_known_answer(measure.task_reference, measure.task_inputs, device)),
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
