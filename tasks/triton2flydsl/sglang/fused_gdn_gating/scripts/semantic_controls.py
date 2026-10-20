# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Untimed actual-operator softplus branch/parameter controls.

The default branch tests linear inputs including a value whose naive exp would
overflow FP32. The explicit finite beta/threshold case separates a stable
always-nonlinear implementation from the public threshold contract at the
unchanged g tolerance. Neither control changes the scored nine workloads.
"""
import math

CONTROL_CASES = [
    {"test_case_id": "control_softplus_default_linear", "params": {"batch": 1, "num_heads": 4, "control": "default_linear", "dtype": "bfloat16"}, "checks": ["correctness"], "dtype": "bfloat16"},
    {"test_case_id": "control_softplus_explicit_threshold", "params": {"batch": 1, "num_heads": 4, "control": "explicit_threshold", "beta": 2., "threshold": 1., "dtype": "bfloat16"}, "checks": ["correctness"], "dtype": "bfloat16"},
]


def make_draw(control, changed=False, device="cuda"):
    import torch

    if control == "default_linear":
        a = [25., 19., 50., 90.] if changed else [20., 21., 30., 100.]
        kwargs = {}  # Exercise the public defaults, not only positional values.
        beta, threshold = 1., 20.
    elif control == "explicit_threshold":
        a = [1., .5, 0., -1.] if changed else [-1., 0., .5, 1.]
        beta, threshold = 2., 1.
        kwargs = {"beta": beta, "threshold": threshold}
    else:
        raise ValueError("Unknown GDN semantic control")
    b = [1., -1., 2., 0.] if changed else [-2., 0., 1., 2.]
    inputs = (torch.zeros(4, dtype=torch.float32, device=device),
              torch.tensor([a], dtype=torch.bfloat16, device=device),
              torch.tensor([b], dtype=torch.bfloat16, device=device),
              torch.zeros(4, dtype=torch.float32, device=device))
    # Independent scalar piecewise formula. Avoid computing the unused exp(x)
    # at x=100; the public branch selects x directly there.
    expected_g = [-math.log1p(math.exp(beta * value)) / beta
                  if beta * value <= threshold else -value for value in a]
    expected_beta = [1. / (1. + math.exp(-value)) for value in b]
    expected = (torch.tensor([[expected_g]], dtype=torch.float32, device=device),
                torch.tensor([[expected_beta]], dtype=torch.bfloat16, device=device).float())
    return inputs, kwargs, expected


def run_controls(operator, compare, unchanged, *, device="cuda", start_shape_id=10):
    import torch

    records = []
    for index, case in enumerate(CONTROL_CASES):
        inputs = None
        for changed in (False, True, False):
            draw, kwargs, expected = make_draw(case["params"]["control"], changed, device)
            if inputs is None:
                inputs = draw
            else:
                for value, new_value in zip(inputs, draw):
                    value.copy_(new_value)
            originals = tuple(value.clone() for value in inputs)
            with torch.no_grad():
                actual = operator(*inputs, **kwargs)
            unchanged(inputs, originals)
            compare(actual, expected, inputs[1].dtype)
            wrong = (expected[0] + 100., expected[1].clone())
            try:
                compare(wrong, expected, inputs[1].dtype)
            except AssertionError:
                pass
            else:
                raise AssertionError("Semantic comparator accepted deliberately incorrect output")
        records.append({"shape_id": start_shape_id + index, "shape": [1, 4],
                        "control": case["test_case_id"], "passed": True, "known_answer": "PASS",
                        "actual_operator_draws": 3, "changed_input": "PASS",
                        "read_only": "PASS", "negative_output": "rejected", "timed": False})
    return records
