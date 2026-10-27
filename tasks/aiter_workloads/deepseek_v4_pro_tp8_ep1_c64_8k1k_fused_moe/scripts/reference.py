"""Protected, explicitly bound MoE reference; no numerical policy defaults.

The stage computation follows ROCm/aiter op_tests/test_moe_2stage.py at
2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0. This draft supports its a8w4
recipe only. Selecting that recipe requires a separate workload/runtime join.
The runner verifies installed source identity before invoking this module.
"""
from __future__ import annotations

import math

_QUANTIZER = "aiter.ops.quant.per_1x32_f8_scale_f8_quant"
_STRICT_METHOD = "aiter_checked_whole_call_v1"
# The global threshold is upstream's warning/check level, selected before
# qualification. This is stricter than upstream's 0.01 rejection threshold.
_STRICT_THRESHOLDS = {"rtol": 0.01, "atol": 0.01, "max_logits_diff": 0.001}


def validate_contract(contract):
    """Fail before importing Torch/AITER when numerical acceptance is unbound."""
    if not isinstance(contract, dict) or contract.get("binding_status") != "frozen":
        raise ValueError("Numerical contract is unresolved; task execution is not qualified")
    expected = {
        "reference_kind": "aiter_torch_stages_a8w4",
        "a1_quantizer": _QUANTIZER,
        "a2_quantizer": _QUANTIZER,
        "activation": "silu",
        "gate_mode": "interleave",
        "weight_layout": "aiter_a16w4_gate_up_interleaved",
        "output_dtype": "bfloat16",
        "hidden_pad": 0,
        "intermediate_pad": 0,
    }
    for key, value in expected.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError(f"Unsupported or unresolved numerical contract field: {key}")
    if contract.get("stage1_dtype") not in ("bfloat16", "float32"):
        raise ValueError("stage1_dtype must be explicitly frozen to bfloat16 or float32")
    if type(contract.get("doweight_stage1")) is not bool:
        raise ValueError("doweight_stage1 must be explicitly bound")
    if "swiglu_limit" not in contract:
        raise ValueError("swiglu_limit must be explicit, including null for no clamp")
    limit = contract["swiglu_limit"]
    if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
        raise ValueError("swiglu_limit must be null or a positive finite number")
    if not isinstance(contract.get("evidence"), list) or not contract["evidence"] or not all(
        isinstance(item, str) and item.strip() for item in contract["evidence"]
    ):
        raise ValueError("Frozen numerical contract requires its evidence references")
    acceptance = contract.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("method") not in ("allclose", _STRICT_METHOD):
        raise ValueError("An explicitly frozen supported acceptance method is required")
    if acceptance["method"] == _STRICT_METHOD:
        if set(acceptance) != {"method", *_STRICT_THRESHOLDS} or any(
            type(acceptance[name]) not in (int, float) or acceptance[name] != value
            for name, value in _STRICT_THRESHOLDS.items()
        ):
            raise ValueError("aiter_checked_whole_call_v1 requires its fixed checked-call thresholds")
        if contract["stage1_dtype"] != "bfloat16":
            raise ValueError("aiter_checked_whole_call_v1 requires the upstream-default BF16 stage1 reference")
        if contract["doweight_stage1"]:
            raise ValueError("aiter_checked_whole_call_v1 requires weighting in stage 2")
    for name in ("rtol", "atol"):
        value = acceptance.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Acceptance requires an explicit finite nonnegative {name}")


def reference(payload, contract):
    """Evaluate the bound quantized semantics, independent of candidate source.

Canonical weights remain unshuffled; cases.load_case checks their exact mapping
to the public shuffled inputs. Quantizer identifiers and intermediate rounding
are frozen before search, never inferred from candidate dispatch or timing.
"""
    validate_contract(contract)
    import torch
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    hidden, _, _, topk_weight, topk_ids = payload["args"]
    canonical = payload["reference"]
    a1, a1_scale = per_1x32_f8_scale_f8_quant(
        hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    intermediate = torch_moe_stage1(
        a1, canonical["w1"], canonical["w2"], topk_weight, topk_ids,
        dtype=getattr(torch, contract["stage1_dtype"]),
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        a1_scale=a1_scale, w1_scale=canonical["w1_scale"],
        w1_bias=payload["kwargs"].get("bias1"),
        doweight=contract["doweight_stage1"], swiglu_limit=contract["swiglu_limit"],
    )
    a2, a2_scale = per_1x32_f8_scale_f8_quant(
        intermediate, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    return torch_moe_stage2(
        a2.reshape(topk_ids.shape[0], topk_ids.shape[1], -1),
        canonical["w1"], canonical["w2"], topk_weight, topk_ids,
        dtype=torch.bfloat16, quant_type=QuantType.per_1x32,
        w2_scale=canonical["w2_scale"], a2_scale=a2_scale,
        w2_bias=payload["kwargs"].get("bias2"),
        doweight=not contract["doweight_stage1"],
    )


def compare_output(actual, expected, contract):
    """Compare complete finite outputs under the explicitly declared method.

    The checked method uses upstream's combined predicate with its tighter
    warning threshold and explicit finite/zero handling. Its acceptance result
    is also used by the declared non-vetoing reference diagnostic.
    """
    validate_contract(contract)
    import torch

    if (not isinstance(expected, torch.Tensor) or expected.dtype != torch.bfloat16
            or expected.ndim != 2 or not expected.numel() or not torch.isfinite(expected).all()):
        raise ValueError("Reference produced an invalid output; this is not a candidate mismatch")
    if not isinstance(actual, torch.Tensor):
        raise AssertionError("Candidate must return a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError("Candidate output shape, dtype or device differs from the contract")
    if not torch.isfinite(actual).all():
        raise AssertionError("Candidate output contains nonfinite values")
    policy = contract["acceptance"]
    ref64, out64 = expected.double(), actual.double()
    metrics = {"compared_elements": expected.numel(),
               "max_absolute_error": (out64 - ref64).abs().max().item()}
    if policy["method"] == "allclose":
        try:
            torch.testing.assert_close(actual, expected, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
        except AssertionError:
            metrics["accepted"] = False
        else:
            metrics["accepted"] = True
        return metrics

    # Preserve upstream order and BF16 isclose arithmetic: rtol scales abs(out).
    close = torch.isclose(expected, actual, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
    mismatches = int(torch.count_nonzero(~close).item())
    denominator = torch.sum(ref64 * ref64 + out64 * out64)
    dot = torch.sum(ref64 * out64)
    den, dot_value = denominator.item(), dot.item()
    if not math.isfinite(den) or den < 0 or not math.isfinite(dot_value):
        raise ValueError("Strict comparison has invalid FP64 reductions")
    if den == 0:
        if torch.count_nonzero(expected).item() or torch.count_nonzero(actual).item():
            raise ValueError("Strict comparison has zero denominator for nonzero tensors")
        logits_diff = 0.0
    else:
        logits_diff = (1 - 2 * dot / denominator).item()
        if not math.isfinite(logits_diff):
            raise ValueError("Strict comparison has a nonfinite normalized difference")
    metrics.update(mismatch_count=mismatches, mismatch_fraction=mismatches / expected.numel(),
                   logits_diff=logits_diff, denominator=den,
                   accepted=mismatches == 0 or logits_diff <= policy["max_logits_diff"])
    return metrics


def check_output(actual, expected, contract):
    """Enforce the declared numerical rule for every candidate/baseline check."""
    metrics = compare_output(actual, expected, contract)
    if not metrics["accepted"]:
        raise AssertionError(f"Candidate output exceeds {contract['acceptance']['method']}: {metrics}")
    return metrics


def route_omission_diagnostic(payload, expected, contract):
    """Measure a structured approximation; its predicate outcome is not a veto.

    Set the smallest routing weight per token to zero, choosing the first slot
    on ties. Keep all other inputs, weights and scales unchanged and do not
    renormalize. This measures that intervention, not its optimization value.
    """
    if contract["doweight_stage1"]:
        raise ValueError("Route-omission diagnostic requires weighting in stage 2")
    weights = payload["args"][3]
    slots = weights.argmin(dim=1, keepdim=True)
    removed = weights.gather(1, slots)
    modified = weights.clone()
    modified.scatter_(1, slots, 0)
    diagnostic = {**payload, "args": payload["args"][:3] + (modified,) + payload["args"][4:]}
    observed = reference(diagnostic, contract)
    return {
        "selection": "smallest_routing_weight_first_slot_on_ties", "renormalized": False,
        "selected_routes": weights.shape[0],
        "zeroed_nonzero_routes": int((removed != 0).sum().item()),
        "removed_weight_min": removed.min().item(), "removed_weight_max": removed.max().item(),
        "comparison": compare_output(observed, expected, contract),
        "affects_qualification": False,
    }


def check_reference_controls(payload, contract):
    """Zero-input path and whole-output/localized comparator controls.

These check actual math/comparator behavior; they do not establish that this
reference recipe matches the served kernel's nonzero rounding semantics.
"""
    validate_contract(contract)
    import torch

    zero_payload = dict(payload)
    zero_payload["args"] = (torch.zeros_like(payload["args"][0]),) + payload["args"][1:]
    zero_payload["kwargs"] = {**payload["kwargs"], "bias1": None, "bias2": None}
    actual = reference(zero_payload, contract)
    expected = torch.zeros_like(payload["args"][0])
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or actual.device != expected.device
            or not torch.equal(actual, expected)):
        raise ValueError("Zero-input quantizer/reference path did not produce zero; calibration is unresolved")
    check_output(actual, expected, contract)
    amplitude = max(1.0, 2.0 * contract["acceptance"]["atol"])
    wrong = torch.full_like(expected, amplitude)
    try:
        check_output(wrong, expected, contract)
    except AssertionError:
        pass
    else:
        raise AssertionError("Comparator accepted deliberately incorrect output")
    expected = reference(payload, contract)
    hidden = payload["args"][0]
    if (
        not isinstance(expected, torch.Tensor) or expected.shape != hidden.shape
        or expected.dtype != getattr(torch, contract["output_dtype"])
        or expected.device != hidden.device or not torch.isfinite(expected).all()
    ):
        raise ValueError("Reference output violates the declared shape/dtype/device or finite-value contract")
    if expected.ndim != 2 or not expected.numel():
        raise ValueError("Localized comparator control has no nonempty output rows")
    policy = contract["acceptance"]
    if policy["method"] == _STRICT_METHOD:
        # A real row omission can satisfy the declared global allowance. This
        # synthetic single-element control has one mismatch and exactly D=1.
        control_ref = torch.zeros_like(expected)
        wrong = control_ref.clone()
        wrong[0, 0] = 1
        try:
            check_output(wrong, control_ref, contract)
        except AssertionError:
            pass
        else:
            raise AssertionError("Comparator accepted localized corruption outside the strict allowance")
        return {
            "zero_hidden_no_bias": "PASS", "wrong_output_rejected": "PASS",
            "localized_output_corruption": {
                "status": "PASS", "synthetic_zero_reference": True,
                "corrupted_elements": 1, "compared_elements": expected.numel(),
                "expected_logits_diff": 1.0, "max_absolute_error": 1.0,
            },
            "route_omission_diagnostic": route_omission_diagnostic(payload, expected, contract),
        }
    # Explicit allclose retains its real-row omission control. A dropped expert
    # can be numerically negligible and is not itself a reliable negative case.
    row = int(expected.float().abs().amax(dim=1).argmax().item())
    magnitude = expected[row].float().abs()
    if not (magnitude > policy["atol"] + policy["rtol"] * magnitude).any():
        raise ValueError("Localized comparator control has insufficient signal at the declared tolerance")
    wrong = expected.clone()
    wrong[row].zero_()
    try:
        check_output(wrong, expected, contract)
    except AssertionError:
        pass
    else:
        raise AssertionError("Comparator accepted a localized output omission exceeding its allowance")
    return {
        "zero_hidden_no_bias": "PASS", "wrong_output_rejected": "PASS",
        "localized_output_omission": {
            "status": "PASS", "row": row, "row_elements": expected.shape[1],
            "compared_elements": expected.numel(), "max_absolute_error": magnitude.max().item(),
        },
    }
