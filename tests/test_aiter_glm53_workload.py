"""CPU checks of the new GLM5.3-Flash task's math and contract, not GPU validation."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.task_spec import load_task_spec

TASK = Path(__file__).resolve().parents[1] / "tasks/aiter_workloads/glm53_flash_tp8_c64_8k1k_fp8_mlp_down"


def module(name):
    spec = importlib.util.spec_from_file_location(f"glm53_{name}", TASK / f"scripts/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_task_loads_with_config_derived_weights_and_exact_model_projection():
    spec = load_task_spec(TASK / "config.yaml", task_id="aiter_workloads/" + TASK.name)
    contract = module("evaluate").load_contract()
    assert spec.baseline.correctness_policy == "required"
    assert spec.candidate.language == "gpu"
    assert [edit.path for edit in spec.candidate.editable] == ["source"]
    model = json.loads((TASK / "model_config.json").read_text())
    assert model["text_config"]["intermediate_size"] // contract["workload"]["tp"] == 1536
    assert "model.layers.0.mlp.down_proj" not in model["quantization_config"]["modules_to_not_convert"]
    assert not contract["operator"]["current_serving_dispatch_verified"]


def test_reference_uses_both_scale_axes_and_preserves_column_major_values():
    reference = module("reference")
    a = torch.ones((2, 256)).to(torch.float8_e4m3fn)
    b = torch.ones((256, 256)).to(torch.float8_e4m3fn)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t().contiguous().t()
    ws = torch.tensor([[1.0, 4.0], [2.0, 8.0]])
    actual = reference.reference(a, b, xs, ws)
    expected = torch.empty((2, 256), dtype=torch.bfloat16)
    expected[0, :128], expected[0, 128:] = 128 * 9, 128 * 18
    expected[1, :128], expected[1, 128:] = 128 * 19, 128 * 38
    assert torch.equal(actual, expected)


def test_small_outputs_cannot_pass_as_zero_or_nan():
    check = module("reference").check_output
    expected = torch.full((2, 4), 1e-4, dtype=torch.bfloat16)
    check(expected.clone(), expected)
    with pytest.raises(AssertionError, match="Frobenius"):
        check(torch.zeros_like(expected), expected)
    with pytest.raises(AssertionError, match="Nonfinite"):
        check(torch.full_like(expected, float("nan")), expected)


def test_snapshot_covers_padding_and_tensor_metadata():
    cases = module("cases")
    storage = torch.arange(32, dtype=torch.float32)
    view = storage[2:10].view(2, 4).t()
    before = cases.snapshot((view,))
    cases.check_unchanged((view,), before)
    storage[-1] = -1
    with pytest.raises(AssertionError, match="read-only"):
        cases.check_unchanged((view,), before)
    before = cases.snapshot((view,))
    view.transpose_(0, 1)
    with pytest.raises(AssertionError, match="read-only"):
        cases.check_unchanged((view,), before)
