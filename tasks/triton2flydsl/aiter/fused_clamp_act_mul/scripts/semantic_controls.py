# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Untimed GELU/GELU-tanh controls of the actual fused operator.

Keep the original SiLU random cases and timed workloads unchanged. Both public
GELU modes are exercised with and without clamping/weights; the weighted draws
also require use of the caller-owned output. The original numerical gate is
retained, so the controls do not demand distinction where two approximations
are already equivalent within that gate.
"""
CONTROL_CASES = [
    {"test_case_id": "control_gelu", "params": {"M": 1, "D": 2048, "limit": 0., "wm": "none", "activation": "gelu", "provided_out": False}, "checks": ["correctness"]},
    {"test_case_id": "control_gelu_weighted_out", "params": {"M": 3, "D": 2048, "limit": 2., "wm": "broadcast", "activation": "gelu", "provided_out": True}, "checks": ["correctness"]},
    {"test_case_id": "control_gelu_tanh", "params": {"M": 1, "D": 2048, "limit": 0., "wm": "none", "activation": "gelu_tanh", "provided_out": False}, "checks": ["correctness"]},
    {"test_case_id": "control_gelu_tanh_weighted_out", "params": {"M": 3, "D": 2048, "limit": 2., "wm": "elementwise", "activation": "gelu_tanh", "provided_out": True}, "checks": ["correctness"]},
]


def make_draw(params, changed=False, device="cuda"):
    import torch

    rows, columns = params["M"], params["D"] // 2
    gate = torch.tensor([-4., -2., -1., 0., 1., 2., 4., 8.], dtype=torch.bfloat16, device=device).repeat(rows, columns // 8)
    up = torch.tensor([-3., -2., -1., .5, 1., 2., 3., 4.], dtype=torch.bfloat16, device=device).repeat(rows, columns // 8)
    if changed:
        up.neg_()
    inp = torch.cat((gate, up), dim=-1).contiguous()
    if params["wm"] == "broadcast":
        weights = torch.tensor([[1.], [-.5], [2.]], dtype=torch.float32, device=device)
    elif params["wm"] == "elementwise":
        weights = torch.tensor([1., -.5], dtype=torch.float32, device=device).repeat(rows, columns // 2)
    else:
        weights = None
    return inp, weights


def reference(inp, limit, weights, activation):
    import torch
    import torch.nn.functional as F

    gate, up = inp.float().chunk(2, dim=-1)
    if limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    if activation not in {"gelu", "gelu_tanh"}:
        raise ValueError("Unknown semantic control activation")
    value = F.gelu(gate, approximate="tanh" if activation == "gelu_tanh" else "none") * up
    if weights is not None:
        value = value * weights
    return value.to(inp.dtype)


def run_controls(operator, compare, unchanged, *, device="cuda"):
    import torch

    records = []
    for case in CONTROL_CASES:
        params = case["params"]
        inp, weights = None, None
        for changed in (False, True, False):
            new_inp, new_weights = make_draw(params, changed, device)
            if inp is None:
                inp, weights = new_inp, new_weights
            else:
                inp.copy_(new_inp)
                if weights is not None:
                    weights.copy_(new_weights)
            inputs = tuple(value for value in (inp, weights) if value is not None)
            originals = tuple(value.clone() for value in inputs)
            expected = reference(inp, params["limit"], weights, params["activation"])
            out = torch.full_like(expected, float("nan")) if params["provided_out"] else None
            with torch.no_grad():
                actual = operator(inp, out=out, swiglu_limit=params["limit"],
                                  activation=params["activation"], weights=weights)
            unchanged(inputs, originals)
            if out is not None and actual is not out:
                raise AssertionError("Operator did not return the caller-owned output")
            compare(actual, expected, inp)
            wrong = expected + 100.
            try:
                compare(wrong, expected, inp)
            except AssertionError:
                pass
            else:
                raise AssertionError("Semantic comparator accepted deliberately incorrect output")
        records.append({"control": case["test_case_id"], "actual_operator_draws": 3,
                        "known_answer": "PASS", "changed_input": "PASS",
                        "read_only": "PASS", "negative_output": "rejected", "timed": False})
    return records
