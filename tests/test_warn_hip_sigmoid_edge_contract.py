import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / 'tasks/torch2hip/gpumode/11184_Sigmoid'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controls(monkeypatch):
    runner = load(TASK / 'eval_tools/evaluate.py', 'sigmoid_edge_runner')
    module = load(TASK / 'pytorch_code_module/py_11184_Sigmoid.py', 'sigmoid_edge_module').Sigmoid()
    functional = load(TASK / 'pytorch_code_functional/py_11184_Sigmoid_func.py', 'sigmoid_edge_functional').Sigmoid()
    monkeypatch.setitem(sys.modules, 'correctness_check', SimpleNamespace(
        _compare_results=lambda a, b, **bounds: torch.allclose(a, b, **bounds)))
    return runner, module, functional


def run(controls, kernel=None):
    runner, module, functional = controls
    try:
        return runner.check_sigmoid_controls(module, functional, kernel, 1e-4, 1e-5, device='cpu')
    finally:
        assert (module.a, module.max, functional.a, functional.max) == (1, 10, 1, 10)


def test_real_reference_and_functional_pass_all_unscored_controls(controls):
    rows = run(controls)
    assert len(rows) == 24
    assert set(rows) >= {'scalar', 'empty', 'extreme', 'negative_max_scalar', 'zero_a_extreme', 'zero_max_empty'}


@pytest.mark.parametrize('corruption', ['mutate', 'signed_zero', 'metadata', 'alias', 'empty_alias', 'negative_max'])
def test_candidate_edge_contract_violations_fail(controls, corruption):
    def kernel(x, a, maximum):
        result = torch.sigmoid(x * a) * maximum
        if corruption == 'mutate' and x.ndim == 0:
            x.add_(1)
        elif corruption == 'signed_zero' and x.numel() == 7:
            x[3] = -0.0
        elif corruption == 'metadata' and x.ndim == 0:
            x.unsqueeze_(0)
        elif corruption == 'alias':
            x.copy_(result)
            return x
        elif corruption == 'empty_alias' and x.numel() == 0:
            return x.view_as(x)
        elif corruption == 'negative_max':
            return torch.sigmoid(x * a) * abs(maximum)
        return result
    with pytest.raises(ValueError):
        run(controls, kernel)


@pytest.mark.parametrize('corruption', ['mutate', 'empty_alias'])
def test_reference_edge_contract_is_also_checked(controls, corruption):
    module = controls[1]
    original = module.forward
    def broken(x):
        result = original(x)
        if corruption == 'mutate' and x.ndim == 0:
            x.add_(1)
        elif corruption == 'empty_alias' and x.numel() == 0:
            return x
        return result
    module.forward = broken
    with pytest.raises(ValueError):
        run(controls)
