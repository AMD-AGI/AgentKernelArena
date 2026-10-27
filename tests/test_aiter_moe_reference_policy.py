"""Prospective comparator checks; CPU tensors do not qualify a native MoE call."""
import builtins
import importlib.util
import json
from pathlib import Path

import pytest


TASK = (Path(__file__).resolve().parents[1] / "tasks" / "aiter_workloads"
        / "deepseek_v4_pro_tp8_ep1_c64_8k1k_fused_moe")
STRICT_METHOD = "aiter_checked_whole_call_v1"


@pytest.fixture(scope="module")
def reference_module():
    spec = importlib.util.spec_from_file_location("_moe_reference_policy", TASK / "scripts/reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract():
    # A fixture for validating prospective support, not an enrolled task policy.
    return {
        "binding_status": "frozen",
        "reference_kind": "aiter_torch_stages_a8w4",
        "a1_quantizer": "aiter.ops.quant.per_1x32_f8_scale_f8_quant",
        "a2_quantizer": "aiter.ops.quant.per_1x32_f8_scale_f8_quant",
        "activation": "silu",
        "gate_mode": "interleave",
        "weight_layout": "aiter_a16w4_gate_up_interleaved",
        "output_dtype": "bfloat16",
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "stage1_dtype": "bfloat16",
        "doweight_stage1": False,
        "swiglu_limit": None,
        "evidence": ["test-only-policy-fixture"],
        "acceptance": {"method": STRICT_METHOD, "rtol": 0.01,
                       "atol": 0.01, "max_logits_diff": 0.001},
    }


@pytest.fixture(scope="module")
def torch():
    # Keep pure contract tests runnable on the source-only host.
    return pytest.importorskip("torch", reason="CPU tensor comparator tests require PyTorch")


def test_task_preserves_prospective_numerical_policy():
    workload = json.loads((TASK / "workloads.json").read_text())
    assert workload["numerical_contract"]["acceptance"]["method"] == STRICT_METHOD
    assert workload["numerical_contract"]["acceptance"]["max_logits_diff"] == 0.001
    assert len(workload["cases"]) == 2


@pytest.mark.parametrize("function,args", [
    ("validate_contract", (None,)),
    ("reference", ({}, None)),
    ("check_output", (None, None, None)),
    ("check_reference_controls", ({}, None)),
])
def test_unbound_contract_fails_before_dependency_imports(reference_module, monkeypatch, function, args):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "aiter"}:
            raise AssertionError("Unbound policy must fail before Torch/AITER imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="unresolved"):
        getattr(reference_module, function)(*args)


def test_strict_contract_accepts_fixed_bf16_recipe(reference_module, contract):
    reference_module.validate_contract(contract)


@pytest.mark.parametrize("field", ["rtol", "atol", "max_logits_diff"])
@pytest.mark.parametrize("value", [0, 0.0101, True, "0.01", None, float("nan"), float("inf")])
def test_strict_thresholds_are_exact_and_not_configurable(reference_module, contract, field, value):
    contract["acceptance"][field] = value
    with pytest.raises(ValueError, match="fixed checked-call thresholds"):
        reference_module.validate_contract(contract)


@pytest.mark.parametrize("field", ["rtol", "atol", "max_logits_diff"])
def test_strict_thresholds_cannot_be_omitted(reference_module, contract, field):
    del contract["acceptance"][field]
    with pytest.raises(ValueError, match="fixed checked-call thresholds"):
        reference_module.validate_contract(contract)


@pytest.mark.parametrize("field,value", [("mask", [0]), ("max_absolute_error", 1.0)])
def test_strict_policy_has_no_extra_mask_or_veto(reference_module, contract, field, value):
    contract["acceptance"][field] = value
    with pytest.raises(ValueError, match="fixed checked-call thresholds"):
        reference_module.validate_contract(contract)


def test_strict_policy_requires_upstream_default_stage1_dtype(reference_module, contract):
    contract["stage1_dtype"] = "float32"
    with pytest.raises(ValueError, match="upstream-default BF16"):
        reference_module.validate_contract(contract)


def test_checked_policy_requires_stage2_weighting(reference_module, contract):
    contract["doweight_stage1"] = True
    with pytest.raises(ValueError, match="requires weighting in stage 2"):
        reference_module.validate_contract(contract)


def test_explicit_allclose_and_fp32_stage1_remain_supported(reference_module, contract):
    contract["stage1_dtype"] = "float32"
    contract["acceptance"] = {"method": "allclose", "rtol": 0.02, "atol": 0.03}
    reference_module.validate_contract(contract)


@pytest.mark.parametrize("actual_value", [1.03125, 0.96875])
def test_strict_global_allowance_accepts_distributed_mismatches(reference_module, contract, torch, actual_value):
    expected = torch.ones((4, 8), dtype=torch.bfloat16, device="cpu")
    actual = torch.full_like(expected, actual_value)
    metrics = reference_module.check_output(actual, expected, contract)
    assert metrics["compared_elements"] == 32
    assert metrics["mismatch_count"] == 32
    assert metrics["mismatch_fraction"] == 1.0
    assert metrics["denominator"] == pytest.approx(32 * (1 + actual_value ** 2))
    assert metrics["logits_diff"] == pytest.approx((1 - actual_value) ** 2 / (1 + actual_value ** 2))
    assert 0 < metrics["logits_diff"] < 0.001
    assert metrics["max_absolute_error"] == abs(1 - actual_value)


def test_checked_threshold_rejects_error_inside_upstream_rejection_envelope(reference_module, contract, torch):
    expected = torch.ones((4, 8), dtype=torch.bfloat16, device="cpu")
    actual = torch.full_like(expected, 1.0625)
    metrics = reference_module.compare_output(actual, expected, contract)
    assert 0.001 < metrics["logits_diff"] < 0.01
    assert metrics["accepted"] is False
    with pytest.raises(AssertionError, match=STRICT_METHOD):
        reference_module.check_output(actual, expected, contract)


def test_route_omission_is_reported_without_veto_or_renormalization(reference_module, contract, torch, monkeypatch):
    hidden = torch.ones((2, 3), dtype=torch.bfloat16, device="cpu")
    weights = torch.tensor([[0.2, 0.1, 0.1, 0.3, 0.15, 0.15],
                            [0.2, 0.2, 0.15, 0.1, 0.15, 0.2]], dtype=torch.float32)
    original = weights.clone()
    payload = {"args": (hidden, object(), object(), weights, object()), "kwargs": {"unchanged": object()}}

    def route_sum_reference(value, policy):
        assert value["args"][:3] == payload["args"][:3]
        assert value["args"][4] is payload["args"][4]
        assert value["kwargs"] is payload["kwargs"]
        changed = value["args"][3]
        assert torch.count_nonzero(changed != original).item() == 2
        wanted = original.clone()
        wanted[0, 1] = 0  # First of the two tied minima.
        wanted[1, 3] = 0  # A different slot for this token.
        assert torch.equal(changed, wanted)
        return changed.sum(dim=1, keepdim=True).expand_as(hidden).to(torch.bfloat16)

    monkeypatch.setattr(reference_module, "reference", route_sum_reference)
    result = reference_module.route_omission_diagnostic(payload, hidden, contract)
    assert result["comparison"]["accepted"] is False
    assert result["affects_qualification"] is False
    assert result["renormalized"] is False
    assert result["selected_routes"] == result["zeroed_nonzero_routes"] == 2
    assert torch.equal(weights, original)


def test_strict_preserves_upstream_isclose_order_and_zero_mismatch_arm(reference_module, contract, torch):
    expected = torch.zeros((2, 2), dtype=torch.bfloat16, device="cpu")
    actual = torch.full_like(expected, 165 / 16384)
    assert torch.isclose(expected, actual, rtol=0.01, atol=0.01).all()
    assert not torch.isclose(actual, expected, rtol=0.01, atol=0.01).any()
    metrics = reference_module.check_output(actual, expected, contract)
    assert metrics["mismatch_count"] == 0
    assert metrics["mismatch_fraction"] == 0.0
    assert metrics["logits_diff"] == 1.0
    contract["acceptance"] = {"method": "allclose", "rtol": 0.01, "atol": 0.01}
    with pytest.raises(AssertionError):
        reference_module.check_output(actual, expected, contract)


def test_strict_can_accept_a_real_row_omission_while_allclose_rejects(reference_module, contract, torch):
    expected = torch.ones((1024, 4), dtype=torch.bfloat16, device="cpu")
    actual = expected.clone()
    actual[0].zero_()
    metrics = reference_module.check_output(actual, expected, contract)
    assert metrics["mismatch_count"] == 4
    assert metrics["mismatch_fraction"] == 1 / 1024
    assert metrics["logits_diff"] == pytest.approx(1 / 2047)
    assert metrics["denominator"] == 8188
    contract["acceptance"] = {"method": "allclose", "rtol": 0.01, "atol": 0.01}
    with pytest.raises(AssertionError):
        reference_module.check_output(actual, expected, contract)


@pytest.mark.parametrize("localized", [False, True])
def test_strict_rejects_known_errors_outside_global_allowance(reference_module, contract, torch, localized):
    expected = torch.zeros((8, 4), dtype=torch.bfloat16, device="cpu")
    actual = expected.clone()
    if localized:
        actual[0, 0] = 1
    else:
        expected.fill_(1)
        actual.fill_(2)
    with pytest.raises(AssertionError, match=STRICT_METHOD) as error:
        reference_module.check_output(actual, expected, contract)
    for name in ("compared_elements", "max_absolute_error", "mismatch_count",
                 "mismatch_fraction", "logits_diff", "denominator"):
        assert name in str(error.value)


def test_strict_both_zero_has_defined_zero_denominator(reference_module, contract, torch):
    expected = torch.zeros((2, 3), dtype=torch.bfloat16, device="cpu")
    metrics = reference_module.check_output(expected.clone(), expected, contract)
    assert metrics == {"compared_elements": 6, "max_absolute_error": 0.0,
                       "mismatch_count": 0, "mismatch_fraction": 0.0,
                       "logits_diff": 0.0, "denominator": 0.0, "accepted": True}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_candidate_nonfinite_is_a_mismatch_but_reference_nonfinite_is_invalid(
    reference_module, contract, torch, value,
):
    good = torch.ones((2, 3), dtype=torch.bfloat16, device="cpu")
    bad = good.clone()
    bad[0, 0] = value
    with pytest.raises(AssertionError, match="nonfinite"):
        reference_module.check_output(bad, good, contract)
    with pytest.raises(ValueError, match="invalid output"):
        reference_module.check_output(good, bad, contract)


@pytest.mark.parametrize("kind", ["not_tensor", "empty", "dtype", "rank"])
def test_reference_must_be_nonempty_2d_bf16(reference_module, contract, torch, kind):
    good = torch.ones((2, 3), dtype=torch.bfloat16, device="cpu")
    bad = {"not_tensor": None, "empty": good[:0], "dtype": good.float(), "rank": good.flatten()}[kind]
    with pytest.raises(ValueError, match="invalid output"):
        reference_module.check_output(good, bad, contract)


@pytest.mark.parametrize("kind", ["not_tensor", "shape", "dtype", "device"])
def test_candidate_must_match_complete_reference_structure(reference_module, contract, torch, kind):
    expected = torch.ones((2, 3), dtype=torch.bfloat16, device="cpu")
    if kind == "not_tensor":
        actual = None
    elif kind == "shape":
        actual = expected[:1]
    elif kind == "dtype":
        actual = expected.float()
    else:
        # A meta tensor checks device disagreement without creating a GPU tensor.
        actual = torch.empty(expected.shape, dtype=expected.dtype, device="meta")
    with pytest.raises(AssertionError, match="tensor|shape, dtype or device"):
        reference_module.check_output(actual, expected, contract)


@pytest.mark.parametrize("denominator,dot,message", [
    (float("inf"), 1.0, "invalid FP64 reductions"),
    (float("nan"), 1.0, "invalid FP64 reductions"),
    (-1.0, 1.0, "invalid FP64 reductions"),
    (0.0, 1.0, "zero denominator for nonzero tensors"),
    (2.0, float("nan"), "invalid FP64 reductions"),
    (2.0, float("inf"), "invalid FP64 reductions"),
    (2.0, -float("inf"), "invalid FP64 reductions"),
    (1e-300, 1e308, "nonfinite normalized difference"),
])
def test_invalid_reductions_cannot_use_the_zero_mismatch_arm(
    reference_module, contract, torch, monkeypatch, denominator, dot, message,
):
    expected = torch.ones((2, 3), dtype=torch.bfloat16, device="cpu")
    # Inject arithmetic failures that small, finite BF16 fixtures cannot produce.
    sums = iter(torch.tensor(value, dtype=torch.float64, device="cpu") for value in (denominator, dot))
    monkeypatch.setattr(torch, "sum", lambda *args, **kwargs: next(sums))
    with pytest.raises(ValueError, match=message):
        reference_module.check_output(expected.clone(), expected, contract)


@pytest.mark.parametrize("method,control_name", [
    (STRICT_METHOD, "localized_output_corruption"),
    ("allclose", "localized_output_omission"),
])
def test_reference_controls_keep_real_zero_path_and_method_specific_corruption(
    reference_module, contract, torch, monkeypatch, method, control_name,
):
    if method == "allclose":
        contract["acceptance"] = {"method": method, "rtol": 0.01, "atol": 0.01}
    hidden = torch.ones((64, 4), dtype=torch.bfloat16, device="cpu")
    routing = torch.tensor([[0.2, 0.1, 0.1, 0.3, 0.15, 0.15]], dtype=torch.float32).repeat(64, 1)
    payload = {"args": (hidden, None, None, routing), "kwargs": {"bias1": "bound-bias", "bias2": "bound-bias"}}
    calls = []

    def identity_reference(value, policy):
        calls.append(value)
        return value["args"][0].clone()

    monkeypatch.setattr(reference_module, "reference", identity_reference)
    controls = reference_module.check_reference_controls(payload, contract)
    assert len(calls) == (3 if method == STRICT_METHOD else 2)
    assert torch.count_nonzero(calls[0]["args"][0]).item() == 0
    assert calls[0]["kwargs"] == {"bias1": None, "bias2": None}
    assert calls[1] is payload
    assert torch.equal(hidden, torch.ones_like(hidden))
    assert payload["kwargs"] == {"bias1": "bound-bias", "bias2": "bound-bias"}
    assert controls["zero_hidden_no_bias"] == "PASS"
    assert controls["wrong_output_rejected"] == "PASS"
    assert controls[control_name]["status"] == "PASS"
    assert controls[control_name]["compared_elements"] == 256
    if method == STRICT_METHOD:
        assert "localized_output_omission" not in controls
        assert controls[control_name]["synthetic_zero_reference"] is True
        assert controls[control_name]["corrupted_elements"] == 1
        assert controls[control_name]["expected_logits_diff"] == 1.0
    else:
        assert controls[control_name]["row_elements"] == 4
