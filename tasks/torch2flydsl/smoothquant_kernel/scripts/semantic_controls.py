# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Untimed zero-row contract controls for both runtime roles.

The timed random workloads are unchanged. Each finite control calls the actual
operator on base, changed and base-again inputs, and checks both returned values
against an independent analytic answer with the original task comparator.
"""
CONTROL_CASES = [
    {"test_case_id": "control_zero_rows", "params": {"m": 3, "n": 5120, "control": "all_zero_rows"}, "checks": ["correctness"]},
    {"test_case_id": "control_mixed_zero_rows", "params": {"m": 3, "n": 5120, "control": "mixed_zero_rows"}, "checks": ["correctness"]},
]


def make_draw(control, changed=False, device="cuda"):
    import torch

    if control not in {c["params"]["control"] for c in CONTROL_CASES}:
        raise ValueError("Unknown SmoothQuant semantic control")
    if control == "all_zero_rows":
        values = [1., 0., -2.] if changed else [0., 0., 0.]
    else:
        values = [-1., 0., 2.] if changed else [0., 2., -1.]
    inp = torch.tensor(values, dtype=torch.bfloat16, device=device)[:, None].repeat(1, 5120)
    x_scale = torch.tensor([1., 2.], dtype=torch.float32, device=device).repeat(2560)
    codes = []
    scales = []
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        codes.append([63 * sign, 127 * sign] * 2560)
        scales.append([abs(value) * 2. / 127. if value else 1.])
    expected = (torch.tensor(codes, dtype=torch.int8, device=device),
                torch.tensor(scales, dtype=torch.float32, device=device))
    return (inp, x_scale), expected


def run_controls(operator, compare, unchanged, *, device="cuda"):
    import torch

    records = []
    for case in CONTROL_CASES:
        inputs = None
        for changed in (False, True, False):
            draw, expected = make_draw(case["params"]["control"], changed, device)
            if inputs is None:
                inputs = draw
            else:
                for value, new_value in zip(inputs, draw):
                    value.copy_(new_value)
            originals = tuple(value.clone() for value in inputs)
            with torch.no_grad():
                actual = operator(*inputs)
            unchanged(inputs, originals)
            compare(actual, expected, inputs)
            wrong = (expected[0].clone(), expected[1] + 1.)
            try:
                compare(wrong, expected, inputs)
            except AssertionError:
                pass
            else:
                raise AssertionError("Semantic comparator accepted deliberately incorrect output")
        records.append({"control": case["test_case_id"], "actual_operator_draws": 3,
                        "known_answer": "PASS", "changed_input": "PASS",
                        "read_only": "PASS", "negative_output": "rejected", "timed": False})
    return records
