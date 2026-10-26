"""Protected, explicitly bound MoE reference; no numerical policy defaults.

The stage computation follows ROCm/aiter op_tests/test_moe_2stage.py at
2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0. This draft supports its a8w4
recipe only. Selecting that recipe requires a separate workload/runtime join.
The runner verifies installed source identity before invoking this module.
"""
from __future__ import annotations

import math

_QUANTIZER = "aiter.ops.quant.per_1x32_f8_scale_f8_quant"


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
    if not isinstance(acceptance, dict) or acceptance.get("method") != "allclose":
        raise ValueError("Only an explicitly frozen allclose acceptance is implemented")
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


def check_output(actual, expected, contract):
    """Require complete, finite outputs and the protected acceptance rule."""
    validate_contract(contract)
    import torch

    if not isinstance(expected, torch.Tensor) or not torch.isfinite(expected).all():
        raise ValueError("Reference produced an invalid output; this is not a candidate mismatch")
    if not isinstance(actual, torch.Tensor):
        raise AssertionError("Candidate must return a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError("Candidate output shape, dtype or device differs from the contract")
    if not torch.isfinite(actual).all():
        raise AssertionError("Candidate output contains nonfinite values")
    policy = contract["acceptance"]
    torch.testing.assert_close(actual, expected, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
    error = (actual.float() - expected.float()).abs()
    return {"compared_elements": expected.numel(), "max_absolute_error": error.max().item() if error.numel() else 0.0}


def check_reference_controls(payload, contract):
    """Known zero-input identity and a comparator rejection control.

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
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise AssertionError("Reference failed the zero-hidden/no-bias MoE identity")
    amplitude = max(1.0, 2.0 * contract["acceptance"]["atol"])
    wrong = torch.full_like(expected, amplitude)
    try:
        check_output(wrong, expected, contract)
    except AssertionError:
        pass
    else:
        raise AssertionError("Comparator accepted deliberately incorrect output")
    return {"zero_hidden_no_bias": "PASS", "wrong_output_rejected": "PASS"}
