"""CPU controls for the unscored Feedforward input-domain checks."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/torch2hip/gpumode/10024_Feedforward"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models():
    adapter = load(TASK / "eval_tools/evaluate.py", "feedforward_geometry_adapter")
    source = load(TASK / "pytorch_code_module/py_10024_Feedforward.py", "feedforward_geometry_source")
    functional = load(TASK / "pytorch_code_functional/py_10024_Feedforward_func.py", "feedforward_geometry_functional")
    torch.manual_seed(0)
    module = source.Feedforward(4).eval()
    candidate = functional.Feedforward(4).eval()
    with torch.no_grad():
        for left, right in (("fc1.weight", "fc1_weight"), ("fc1.bias", "fc1_bias"),
                            ("fc2.weight", "fc2_weight"), ("fc2.bias", "fc2_bias")):
            candidate.state_dict()[right].copy_(module.state_dict()[left])
    compare = lambda expected, actual, **bounds: torch.allclose(expected, actual, **bounds)
    return adapter, source, functional, module, candidate, compare


def state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def unchanged(model, original):
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())


def check(models, hip_fn):
    adapter, _, _, module, candidate, compare = models
    before_module, before_candidate = state(module), state(candidate)
    try:
        return adapter.check_input_geometry_controls(
            module, candidate, hip_fn, compare, 1e-4, 1e-5)
    finally:
        unchanged(module, before_module)
        unchanged(candidate, before_candidate)


def test_original_five_scored_inputs_and_unscored_geometry(models):
    _, source, functional, _, _, _ = models
    cases = list(source.get_inputs())
    assert [(x.shape[0], y.shape[0], x.shape[1], y.shape[1]) for x, y in cases] == [
        (1, 1, 4, 4), (2, 2, 4, 4), (4, 4, 4, 4), (8, 8, 4, 4), (16, 16, 4, 4)]
    assert check(models, None) == ("x3_y1", "x1_y5", "strided_x3_y5")
    assert check(models, functional.module_fn) == ("x3_y1", "x1_y5", "strided_x3_y5")
    tree = ast.parse((TASK / "eval_tools/evaluate.py").read_text())
    correctness = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "correctness")
    performance = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "performance")
    invoked = lambda node: any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                               and call.func.id == "check_input_geometry_controls"
                               for call in ast.walk(node))
    assert invoked(correctness) and not invoked(performance)


def test_equal_half_specialization_passes_old_cases_but_fails_new_control(models):
    _, source, functional, module, candidate, _ = models

    def equal_half_only(x, y, *parameters):
        output = functional.module_fn(x, y, *parameters)
        return output if x.shape[0] == y.shape[0] else output.roll(1, 0)

    for x, y in source.get_inputs():
        torch.testing.assert_close(candidate(x, y, fn=equal_half_only), module(x, y))
    with pytest.raises(ValueError, match="unequal/strided geometry"):
        check(models, equal_half_only)


def test_nonlisted_shape_specialization_is_rejected(models):
    _, _, functional, _, _, _ = models

    def listed_rows_only(x, y, *parameters):
        output = functional.module_fn(x, y, *parameters)
        return output if x.shape[0] in (1, 2, 4, 8, 16) and y.shape[0] == x.shape[0] else output.flip(0)

    with pytest.raises(ValueError, match="unequal/strided geometry"):
        check(models, listed_rows_only)


def test_noncontiguous_view_specialization_is_rejected(models):
    _, _, functional, _, _, _ = models

    def contiguous_only(x, y, *parameters):
        output = functional.module_fn(x, y, *parameters)
        return output if x.is_contiguous() and y.is_contiguous() else output.flip(0)

    with pytest.raises(ValueError, match="strided_x3_y5"):
        check(models, contiguous_only)


def test_mutating_caller_input_cannot_pass_with_correct_output(models):
    _, _, functional, _, _, _ = models

    def mutates_x(x, y, *parameters):
        original = x.clone()
        with torch.no_grad():
            x.add_(1)
        return functional.module_fn(original, y, *parameters)

    with pytest.raises(ValueError, match="read-only candidate input 0"):
        check(models, mutates_x)


def test_mutating_live_parameter_is_rejected_and_restored(models):
    _, _, functional, _, _, _ = models

    def mutates_weight(x, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        original = fc1_weight.clone()
        with torch.no_grad():
            fc1_weight.add_(1)
        return functional.module_fn(x, y, original, fc1_bias, fc2_weight, fc2_bias)

    with pytest.raises(ValueError, match="modified functional state: fc1_weight"):
        check(models, mutates_weight)
