"""CPU controls for the TanH task's unscored tail geometries."""
import ast
import importlib.util
import json
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
    source = load(TASK / "pytorch_code_module/py_11178_TanH.py", "tail_reference")
    functional = load(TASK / "pytorch_code_functional/py_11178_TanH_func.py", "tail_functional")
    runner = load(TASK / "eval_tools/evaluate.py", "tail_runner")
    return source.TanH().eval(), functional.TanH().eval(), runner


def run(controls, kernel):
    module, functional, runner = controls
    return runner.check_tail_geometry_controls(module, functional, kernel,
                                               torch.allclose, 1e-4, 1e-5, device="cpu")


def test_honest_all_residues_and_restore(controls):
    assert run(controls, None) == [
        "float32_scalar", "float32_vec2", "float32_vec2_tail",
        "float32_vec4_vec2", "float32_vec4_vec2_tail",
        "float64_scalar", "float64_vec2_tail", "float64_two_vec2_tail",
    ]
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


@pytest.mark.parametrize("dtype,length,tail,name", [
    (torch.float32, 2, 2, "float32_vec2"),
    (torch.float32, 3, 1, "float32_vec2_tail"),
    (torch.float32, 6, 2, "float32_vec4_vec2"),
    (torch.float32, 7, 1, "float32_vec4_vec2_tail"),
    (torch.float64, 3, 1, "float64_vec2_tail"),
    (torch.float64, 5, 1, "float64_two_vec2_tail"),
])
def test_selected_remainder_only_wrong_implementation_rejected(controls, dtype, length, tail, name):
    def corrupt_selected_remainder(value, a, maximum):
        result = torch.tanh(a * value) * maximum
        if value.dtype == dtype and value.numel() == length:
            result.reshape(-1)[-tail:] = 0
        return result

    with pytest.raises(ValueError, match=f"{name} TanH disagrees"):
        run(controls, corrupt_selected_remainder)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_wrong_module_reference_rejected_by_math_answer(controls):
    module = controls[0]
    original = module.forward
    def wrong_tail(value):
        out = original(value)
        if value.dtype == torch.float32 and value.numel() == 6:
            out.reshape(-1)[-2:] = 0
        return out
    module.forward = wrong_tail
    with pytest.raises(ValueError, match="independent tail reference"):
        run(controls, None)


def test_correct_output_input_mutation_rejected(controls):
    def mutate(value, a, maximum):
        out = torch.tanh(a * value) * maximum
        if value.dtype == torch.float32 and value.numel() == 6:
            value.reshape(-1)[-1] += 1
        return out
    with pytest.raises(ValueError, match="changed a caller-owned input"):
        run(controls, mutate)


def test_scalar_mutation_rejected_and_restored(controls):
    functional = controls[1]
    def mutate_state(value, a, maximum):
        out = torch.tanh(a * value) * maximum
        if value.dtype == torch.float32 and value.numel() == 6:
            functional.max = 99
        return out
    with pytest.raises(ValueError, match="changed scalar state"):
        run(controls, mutate_state)
    assert (controls[0].a, controls[0].max, controls[1].a, controls[1].max) == (1, 10, 1, 10)


def test_correctness_wiring_and_scored_workload_unchanged():
    tree = ast.parse((TASK / "eval_tools/evaluate.py").read_text())
    correctness = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "correctness")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "check_tail_geometry_controls"
               for node in ast.walk(correctness))
    assert any(isinstance(node, ast.Constant) and node.value == "unscored_tail_geometry_controls"
               for node in ast.walk(correctness))
    rows = json.loads((TASK / "workload.json").read_text())["cases"]
    assert len(rows) == 11
    assert [row["test_case_id"] for row in rows] == [f"case_{i}" for i in range(11)]


def test_previous_controls_admit_size_six_cached_fault(controls):
    module, functional, runner = controls
    def six_only_bad(value, a, maximum):
        result = torch.tanh(a * value) * maximum
        if value.dtype == torch.float32 and value.numel() == 6:
            result.reshape(-1)[-2:] = 0
        return result
    assert len(runner.check_signed_scalar_controls(module, functional, six_only_bad,
                                                 torch.allclose, 1e-4, 1e-5, device="cpu")) == 5
    assert len(runner.check_layout_dtype_controls(module, functional, six_only_bad,
                                               torch.allclose, 1e-4, 1e-5, device="cpu")) == 5
    with pytest.raises(ValueError, match="float32_vec4_vec2 TanH disagrees"):
        run(controls, six_only_bad)


def test_reference_input_mutation_rejected(controls):
    module = controls[0]
    original = module.forward
    def mutate_reference(value):
        out = original(value)
        if value.dtype == torch.float64 and value.numel() == 3:
            value.reshape(-1)[-1] += 1
        return out
    module.forward = mutate_reference
    with pytest.raises(ValueError, match="changed a caller-owned input"):
        run(controls, None)
