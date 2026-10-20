"""CPU contracts for realistic HIP generation workloads; no GPU qualification."""
import ast
import copy
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "10024_Feedforward", "14044_PositionWiseFeedForward",
    "14069_TransformerFFNLayer", "5334_GateGRUSelectionLayer", "8325_MaskedLanguageModel",
)
COUNTS = (5, 10, 5, 5, 5)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _parts(name):
    task = ROOT / "tasks/torch2hip/gpumode" / name
    module = _load(next((task / "pytorch_code_module").glob("*.py")), name + "_module")
    functional = _load(next((task / "pytorch_code_functional").glob("*.py")), name + "_functional")
    runner = _load(task / "eval_tools/evaluate.py", name + "_runner")
    return task, module, functional, runner


def _models(name, module, functional):
    args, kwargs = module.get_init_inputs()
    torch.manual_seed(0)
    expected = getattr(module, name.split("_", 1)[1])(*args, **kwargs).eval()
    actual = getattr(functional, name.split("_", 1)[1])(*args, **kwargs).eval()
    checks = _load(ROOT / "tasks/torch2hip/gpumode" / name / "eval_tools/correctness_check.py",
                   name + "_checks")
    aligned, evidence = checks._align_state_dict(expected, actual)
    assert aligned, evidence
    return expected, actual, checks


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name,count", tuple(zip(TASKS, COUNTS)))
def test_manifest_exactly_matches_both_generators_and_constructor_parameters(name, count):
    task, module, functional, runner = _parts(name)
    rows = json.loads((task / "workload.json").read_text())["cases"]
    assert len(rows) == count
    runner.check_model_init_identity(rows, module, functional)
    kwargs = module.get_init_inputs()[1]
    widths = [value for key, value in kwargs.items()
              if key in {"input_size", "hidden_size", "fc_dim", "d_model", "hidden",
                         "filter_size", "dim_model", "vocab_size"}]
    assert min(widths) >= 128
    with torch.device("meta"):
        module_cases = list(module.get_inputs())
        functional_cases = list(functional.get_inputs())
    assert len(module_cases) == len(functional_cases) == count
    for row, left, right in zip(rows, module_cases, functional_cases):
        runner.check_case_identity(row, left)
        runner.check_case_identity(row, right)
        assert row["params"]["model_init"] == module.get_init_inputs()
    # The row dimension has both decode-size work and non-multiple tile tails.
    if name == "11709_InnerProd":
        assert any(case[1].shape[-1] % 16 != 0 for case in module_cases)
    else:
        rows_per_case = [sum(value.shape[0] for value in case) if name == "10024_Feedforward"
                         else math.prod(case[0].shape[:-1]) for case in module_cases]
        assert min(rows_per_case) <= 4 and max(rows_per_case) >= 256
        assert any(value > 16 and value % 16 != 0 for value in rows_per_case)
    config = yaml.safe_load((task / "config.yaml").read_text())
    assert config["candidate"]["initial_state"] == "unimplemented"
    assert not runner.has_implementation(task / config["candidate"]["editable"][0])


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("defect", ["manifest", "functional"])
def test_constructor_metadata_mismatch_is_rejected(name, defect):
    task, module, functional, runner = _parts(name)
    rows = json.loads((task / "workload.json").read_text())["cases"]
    if defect == "manifest":
        rows[0]["params"]["model_init"][1][next(iter(module.get_init_inputs()[1]))] = 4
    else:
        functional.get_init_inputs = lambda: [[], {"hidden": 4}]
    with pytest.raises(ValueError, match="initializer"):
        runner.check_model_init_identity(rows, module, functional)


@pytest.mark.parametrize("name", TASKS)
def test_full_cpu_module_and_independent_functional_baselines_agree(name):
    task, module, functional, runner = _parts(name)
    expected, actual, checks = _models(name, module, functional)
    controls = (_load(task / "eval_tools/case_controls.py", name + "_affine")
                if name == "11709_InnerProd" else None)
    rows = json.loads((task / "workload.json").read_text())["cases"]
    with torch.no_grad():
        for index, inputs in enumerate(module.get_inputs()):
            runner.check_case_identity(rows[index], inputs)
            if controls is not None:
                controls.configure_models((expected, actual), inputs)
            reference_inputs = [value.clone() for value in inputs]
            reference = expected(*inputs)
            result = actual(*inputs)
            runner.output_contract(reference, result)
            assert checks._compare_results(reference, result, rtol=1e-4, atol=1e-5)
            assert not checks._compare_results(reference, torch.zeros_like(reference),
                                               rtol=1e-4, atol=1e-5)
            for value, original in zip(inputs, reference_inputs):
                assert torch.equal(value, original)
            assert reference.numel() * reference.element_size() <= 4 * 1024**2
            assert sum(value.numel() * value.element_size() for value in inputs) <= 33 * 1024**2
            if name == "8325_MaskedLanguageModel":
                torch.testing.assert_close(reference.exp().sum(-1),
                                           torch.ones(reference.shape[:-1]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", TASKS)
def test_each_live_model_parameter_changes_a_nontrivial_reference_and_stays_aligned(name):
    task, module, functional, runner = _parts(name)
    expected, actual, checks = _models(name, module, functional)
    inputs = next(case for index, case in enumerate(module.get_inputs()) if index == 1)
    original = copy.deepcopy(expected.state_dict())
    with torch.no_grad():
        reference = expected(*inputs)
        for key, tensor in original.items():
            if not tensor.is_floating_point():
                continue
            for delta in (1., -1., 4.):
                state = copy.deepcopy(original)
                state[key].reshape(-1)[0] += delta
                expected.load_state_dict(state)
                changed = expected(*inputs)
                if not checks._compare_results(reference, changed, rtol=1e-4, atol=1e-5):
                    break
            else:
                pytest.fail(f"Parameter {key} is not exercised by {name}'s nontrivial inputs")
            aligned, detail = checks._align_state_dict(expected, actual)
            assert aligned, detail
            result = actual(*inputs)
            runner.output_contract(changed, result)
            assert checks._compare_results(changed, result, rtol=1e-4, atol=1e-5)
            expected.load_state_dict(original)


def test_feedforward_unequal_strided_and_parameter_controls_support_real_feature_width():
    name = "10024_Feedforward"
    _, module, functional, runner = _parts(name)
    expected, actual, checks = _models(name, module, functional)
    with torch.no_grad():
        result = runner.check_input_geometry_controls(
            expected, actual, None, checks._compare_results, 1e-4, 1e-5)
        assert result == ("x3_y1", "x1_y5", "strided_x3_y5")
        runner.check_parameter_variants(expected, actual, None, next(module.get_inputs()),
                                        checks._compare_results, 1e-4, 1e-5)


@pytest.mark.parametrize("name", TASKS)
def test_workload_upgrade_preserves_every_model_and_reference_math_node(name):
    task = ROOT / "tasks/torch2hip/gpumode" / name
    baseline = "099e44cf6deb74131d2cfc96a3a055949a2edbdc"
    for folder in ("pytorch_code_module", "pytorch_code_functional"):
        path = next((task / folder).glob("*.py"))
        old = subprocess.check_output(
            ["git", "show", baseline + ":" + path.relative_to(ROOT).as_posix()],
            cwd=ROOT, text=True)
        def arithmetic(source):
            tree = ast.parse(source)
            tree.body = [node for node in tree.body
                         if not (isinstance(node, ast.FunctionDef)
                                 and node.name in {"get_inputs", "get_init_inputs"})]
            return ast.dump(tree, include_attributes=False)
        assert arithmetic(path.read_text()) == arithmetic(old)
