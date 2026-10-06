"""Independent known answers and comparator controls for task validation only.

These small fixtures are validation evidence, never additional scored cases.
They do not call the production baseline, initializer or candidate. Expected
answers come from scalar Python index arithmetic and block-scaled dot products
written from the definition, not from outputs of task_reference.
"""
from __future__ import annotations

import torch

M, N, K, BLOCK = 2, 16, 256, 128


def _shuffle_position(row, col, k):
    """Physical element of logical weight (row, col) under AITER's 16x16 shuffle.

    Rows group in 16s; each 32-column block splits into two 16-column halves;
    a 16x16 tile is stored row by row, tiles ordered by row group, column block
    and half.
    """
    row_group, row_in = divmod(row, 16)
    col_block, rest = divmod(col, 32)
    half, col_in = divmod(rest, 16)
    return (((row_group * (k // 32) + col_block) * 2 + half) * 16 + row_in) * 16 + col_in


def _fixture():
    # Exactly representable E4M3 integers and power-of-two scales: every partial
    # sum is exact in FP32, so the only rounding is the final BF16 conversion.
    a = [[((i * 3 + t * 5) % 7) - 3 for t in range(K)] for i in range(M)]
    b = [[((j * 7 + t * 3) % 5) - 2 for t in range(K)] for j in range(N)]
    a_scale = [[0.5, 1.0], [2.0, 0.25]]
    b_scale = [[0.25, 2.0]]
    return a, b, a_scale, b_scale


def block_scaled_known_answer(reference, inputs_module, device):
    a, b, a_scale, b_scale = _fixture()
    storage = inputs_module.A_SCALE_STORAGE
    expected = [[sum(a[i][t] * a_scale[i][t // BLOCK] * b[j][t] * b_scale[j // BLOCK][t // BLOCK]
                     for t in range(K)) for j in range(N)] for i in range(M)]
    physical_b = [0] * (N * K)
    for row in range(N):
        for col in range(K):
            physical_b[_shuffle_position(row, col, K)] = b[row][col]
    blocks = K // BLOCK
    if storage == "raw":
        # The (m, sk) buffer holds the column-major payload: element t * m + i is scale (i, t).
        flat = [a_scale[i][t] for t in range(blocks) for i in range(M)]
        stored_scale = [flat[r * blocks:(r + 1) * blocks] for r in range(M)]
    else:
        stored_scale = a_scale
    got = reference.run(
        a=torch.tensor(a, dtype=torch.float32, device=device).to(torch.float8_e4m3fn),
        b=torch.tensor(physical_b, dtype=torch.float32, device=device).reshape(N, K).to(torch.float8_e4m3fn),
        a_scale=torch.tensor(stored_scale, dtype=torch.float32, device=device),
        b_scale=torch.tensor(b_scale, dtype=torch.float32, device=device))
    wanted = torch.tensor(expected, dtype=torch.float64).to(torch.bfloat16)
    torch.testing.assert_close(got.cpu(), wanted, rtol=0, atol=0)
    return {"oracle": ("scalar 16x16 weight shuffle positions, declared activation-scale storage and "
                       "128x128 block-scaled dot products"),
            "a_scale_storage": storage, "shape": [M, N, K],
            "expected_row0": [float(v) for v in wanted[0, :4]],
            "observed_row0": [float(v) for v in got[0, :4].float().cpu()]}


def comparator_controls(compare, device):
    expected = torch.ones((2, 4), dtype=torch.bfloat16, device=device)
    # Positive control with nonzero error; this is not just a self-compare.
    accepted_value, rejected_value = 1.0078125, 1.03125
    compare.run(torch.full_like(expected, accepted_value), expected)
    rejected = {
        "outside_numerical_gate": torch.full_like(expected, rejected_value),
        "wrong_sign": -expected,
        "wrong_shape": expected[:, :1],
        "wrong_dtype": expected.float(),
        "nonfinite_output": torch.full_like(expected, float("nan")),
    }
    observations = {}
    for name, wrong in rejected.items():
        try:
            compare.run(wrong, expected)
        except AssertionError as error:
            observations[name] = str(error)
        else:
            raise RuntimeError(f"Comparator accepted negative control: {name}")
    return {"oracle": "BF16 0.01 either tolerance", "reference_value": 1.,
            "accepted_inexact_value": accepted_value, "rejected_inexact_value": rejected_value,
            "rejected_controls": observations}


def run_controls(measure, *, device="cuda", records=None):
    records = [] if records is None else records
    checks = [("blockwise_scaled_known_answer",
               lambda: block_scaled_known_answer(measure.task_reference, measure.task_inputs, device)),
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
