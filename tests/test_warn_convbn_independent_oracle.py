"""CPU controls for the Conv2d/BatchNorm baseline's independent known answer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TASK = (Path(__file__).resolve().parents[1] / "tasks/torch2hip/kernelbench/level2"
        / "l2n73_Conv2d_BatchNorm_Scaling")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair():
    torch = pytest.importorskip("torch")
    runner = load(TASK / "eval_tools/evaluate.py", "convbn_oracle_eval")
    source = load(TASK / "pytorch_code_module/py_l2n73_Conv2d_BatchNorm_Scaling.py",
                  "convbn_oracle_module")
    functional = load(TASK / "pytorch_code_functional/py_l2n73_Conv2d_BatchNorm_Scaling_func.py",
                      "convbn_oracle_functional")
    torch.manual_seed(3)
    module = source.Conv2d_BatchNorm_Scaling(*source.get_init_inputs()).eval()
    selected = functional.Conv2d_BatchNorm_Scaling(*functional.get_init_inputs()).eval()
    selected.load_state_dict(module.state_dict())
    compare = lambda expected, actual, *, rtol, atol: torch.allclose(
        expected, actual, rtol=rtol, atol=atol)
    return runner, module, selected, compare


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def unchanged(model, before):
    torch = pytest.importorskip("torch")
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())


def test_known_answer_passes_and_restores_both_model_states():
    runner, module, selected, compare = pair()
    original_module, original_selected = snapshot(module), snapshot(selected)
    runner.check_independent_baseline_oracle(module, selected, compare, 1e-4, 1e-5)
    unchanged(module, original_module)
    unchanged(selected, original_selected)


@pytest.mark.parametrize("primitive", ["conv2d", "batch_norm"])
def test_shared_wrong_primitive_passes_pairwise_comparison_but_fails_known_answer(
        primitive, monkeypatch):
    torch = pytest.importorskip("torch")
    runner, module, selected, compare = pair()
    original_module, original_selected = snapshot(module), snapshot(selected)
    import torch.nn.functional as functional
    original = getattr(functional, primitive)

    def shared_wrong(*args, **kwargs):
        return original(*args, **kwargs) + 0.25

    monkeypatch.setattr(functional, primitive, shared_wrong)
    input_value = torch.arange(8 * 5 * 6, dtype=torch.float32).reshape(1, 8, 5, 6) / 16
    # The old pairwise oracle cannot detect this common-mode primitive defect.
    assert compare(module(input_value), selected(input_value), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="failed independent Conv/BN known-answer control"):
        runner.check_independent_baseline_oracle(module, selected, compare, 1e-4, 1e-5)
    unchanged(module, original_module)
    unchanged(selected, original_selected)
