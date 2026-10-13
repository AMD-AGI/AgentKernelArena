"""CPU controls for signed TanH scalar behavior."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/hip2hip/TanH"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controls():
    source = load(TASK / "pytorch_code_module/py_11178_TanH.py", "tanh_signed_module_controls")
    functional = load(TASK / "pytorch_code_functional/py_11178_TanH_func.py",
                      "tanh_signed_functional_controls")
    runner = load(TASK / "eval_tools/evaluate.py", "tanh_signed_runner_controls")
    return source.TanH().eval(), functional.TanH().eval(), runner


def compare(left, right, **bounds):
    return torch.allclose(left, right, **bounds)


def run(controls, kernel):
    module, functional, runner = controls
    return runner.check_signed_scalar_controls(module, functional, kernel,
                                               compare, 1e-4, 1e-5, device="cpu")


def test_honest_signed_controls_and_state_restore(controls):
    assert run(controls, None) == ["zero_a", "zero_max", "negative_a_only",
                                  "negative_max_only", "both_negative"]
    assert controls[0].a == controls[1].a == 1
    assert controls[0].max == controls[1].max == 10


@pytest.mark.parametrize("kernel", [
    lambda value, a, maximum: torch.tanh(abs(a) * value) * maximum,
    lambda value, a, maximum: torch.tanh(a * value) * abs(maximum),
    lambda value, a, maximum: torch.tanh(abs(a) * value) * abs(maximum),
])
def test_sign_erasing_candidate_rejected_and_state_restored(controls, kernel):
    with pytest.raises(ValueError, match="TanH disagrees with the mathematical reference"):
        run(controls, kernel)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_independent_oracle_rejects_wrong_module_reference(controls):
    module, _, _ = controls
    module.forward = lambda value: torch.tanh(abs(module.a) * value) * abs(module.max)
    with pytest.raises(ValueError, match="mathematical reference"):
        run(controls, None)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_correct_output_with_candidate_input_mutation_is_rejected(controls):
    def corrupt_input_after_output(value, a, maximum):
        output = torch.tanh(a * value) * maximum
        value.add_(1)
        return output

    with pytest.raises(ValueError, match="changed a caller-owned input"):
        run(controls, corrupt_input_after_output)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_signed_zero_input_mutation_is_rejected(controls):
    def change_zero_sign_after_output(value, a, maximum):
        output = torch.tanh(a * value) * maximum
        value[2] = -0.0
        return output

    with pytest.raises(ValueError, match="changed a caller-owned input"):
        run(controls, change_zero_sign_after_output)


def test_correct_module_output_with_reference_input_mutation_is_rejected(controls):
    module = controls[0]
    original_forward = module.forward

    def corrupt_input_after_output(value):
        output = original_forward(value)
        value.add_(1)
        return output

    module.forward = corrupt_input_after_output
    with pytest.raises(ValueError, match="changed a caller-owned input"):
        run(controls, None)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_correctness_action_wires_controls_without_changing_scored_rows():
    tree = ast.parse((TASK / "eval_tools/evaluate.py").read_text())
    correctness = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "correctness")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "check_signed_scalar_controls"
               for node in ast.walk(correctness))
    assert any(isinstance(node, ast.Constant) and node.value == "unscored_signed_scalar_controls"
               for node in ast.walk(correctness))
    import json
    rows = json.loads((TASK / "workload.json").read_text())["cases"]
    assert len(rows) == 11
    assert [row["test_case_id"] for row in rows] == [f"case_{index}" for index in range(11)]
