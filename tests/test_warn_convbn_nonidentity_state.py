"""CPU regressions for the ConvBN nonidentity state and input contract."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def control():
    source = load(TASK / 'pytorch_code_module/py_l2n73_Conv2d_BatchNorm_Scaling.py',
                  'convbn_module_source')
    selected = load(TASK / 'pytorch_code_functional/py_l2n73_Conv2d_BatchNorm_Scaling_func.py',
                    'convbn_functional_source')
    evaluator = load(TASK / 'eval_tools/evaluate.py', 'convbn_evaluator')
    torch.manual_seed(42)
    module = source.Conv2d_BatchNorm_Scaling(8, 64, 3, 2.0).eval()
    functional = selected.Conv2d_BatchNorm_Scaling(8, 64, 3, 2.0).eval()
    functional.load_state_dict(module.state_dict())
    x = torch.rand(1, 8, 5, 6)
    original = module(x.clone())
    before_module = evaluator._snapshot_control_tensors(module.state_dict())
    before_selected = evaluator._snapshot_control_tensors(functional.state_dict())
    return module, functional, selected.module_fn, evaluator, x, original, before_module, before_selected


def run(control, kernel=None):
    module, functional, _, evaluator, x, original, _, _ = control
    return evaluator.check_nonidentity_batchnorm(module, functional, kernel, [x],
                                                  original, torch.allclose, 1e-4, 1e-5)


def restored(control):
    module, functional, _, evaluator, _, _, before_module, before_selected = control
    evaluator._require_unchanged_control_tensors(module.state_dict(), before_module, 'module')
    evaluator._require_unchanged_control_tensors(functional.state_dict(), before_selected, 'selected')


def test_honest_control_and_final_state_restoration(control):
    before_x = control[4].clone()
    run(control)
    restored(control)
    assert torch.equal(control[4], before_x)


@pytest.mark.parametrize('state_name', ['weight', 'bias', 'running_mean', 'running_var'])
def test_correct_output_with_mutated_bn_state_rejected(control, state_name):
    honest = control[2]
    def mutate(x, weight, bias, gamma, beta, mean, variance, eps, scale):
        result = honest(x, weight, bias, gamma, beta, mean, variance, eps, scale)
        live = {'weight': gamma, 'bias': beta, 'running_mean': mean,
                'running_var': variance}[state_name]
        with torch.no_grad():
            live[0] += .25
        return result
    with pytest.raises(ValueError, match=f'selected state bn.{state_name} changed'):
        run(control, mutate)
    restored(control)


def test_correct_output_with_mutated_caller_input_rejected(control):
    honest = control[2]
    def mutate(x, *args):
        result = honest(x, *args)
        x.reshape(-1)[0] += 1
        return result
    with pytest.raises(ValueError, match='selected input 0 changed'):
        run(control, mutate)
    restored(control)


def test_reference_state_mutation_rejected_and_restored(control):
    module = control[0]
    original_forward = module.forward
    def mutate(x):
        result = original_forward(x)
        module.bn.running_mean[0] += .25
        return result
    module.forward = mutate
    with pytest.raises(ValueError, match='reference state bn.running_mean changed'):
        run(control)
    restored(control)


def test_reference_input_mutation_rejected_and_restored(control):
    module = control[0]
    original_forward = module.forward
    def mutate(x):
        result = original_forward(x)
        x.reshape(-1)[0] += 1
        return result
    module.forward = mutate
    with pytest.raises(ValueError, match='reference input 0 changed'):
        run(control)
    restored(control)


def test_candidate_exception_restores_original_state(control):
    def fail(x, weight, bias, gamma, beta, mean, variance, eps, scale):
        with torch.no_grad():
            mean[0] += 1
        raise RuntimeError('candidate failed')
    with pytest.raises(RuntimeError, match='candidate failed'):
        run(control, fail)
    restored(control)


def test_correctness_wiring_and_scored_files_preserved():
    tree = ast.parse((TASK / 'eval_tools/evaluate.py').read_text())
    correctness = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'correctness')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'check_nonidentity_batchnorm'
               for node in ast.walk(correctness))
    import json
    rows = json.loads((TASK / 'workload.json').read_text())['cases']
    assert len(rows) == 5
    assert [row['test_case_id'] for row in rows] == [f'case_{i}' for i in range(5)]
