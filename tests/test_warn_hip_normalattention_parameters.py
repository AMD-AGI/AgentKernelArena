"""CPU controls for public NormalAttention parameter variants."""
import importlib.util
from pathlib import Path

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/hip2hip/NormalAttention_dot'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = load(TASK / 'pytorch_code_module/py_1001_NormalAttention_dot.py', 'normalattention_reference')
FUNCTIONAL = load(TASK / 'pytorch_code_functional/py_1001_NormalAttention_dot_func.py', 'normalattention_functional')
EVALUATE = load(TASK / 'eval_tools/evaluate.py', 'normalattention_evaluate')


def models():
    torch.manual_seed(0)
    module = REFERENCE.NormalAttention_dot(4).eval()
    functional = FUNCTIONAL.NormalAttention_dot(4).eval()
    functional.load_state_dict(module.state_dict())
    torch.manual_seed(1337)
    inputs = [torch.randn(4, 4, 4, 4)]
    expected = module(*inputs)
    return module, functional, inputs, expected


def compare(a, b, *, rtol, atol):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def check(module, functional, inputs, expected, candidate):
    return EVALUATE.check_parameter_variants(
        module, functional, inputs, expected, candidate, compare, 1e-4, 1e-5)


def assert_restored(module, functional, inputs, states, original_input):
    for model, before in zip((module, functional), states):
        assert model.state_dict().keys() == before.keys()
        assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert torch.equal(inputs[0], original_input)


def snapshot(module, functional):
    return ({name: value.clone() for name, value in module.state_dict().items()},
            {name: value.clone() for name, value in functional.state_dict().items()})


def test_all_eight_live_parameters_use_independent_reference():
    module, functional, inputs, expected = models()
    states, original_input = snapshot(module, functional), inputs[0].clone()
    assert check(module, functional, inputs, expected, FUNCTIONAL.module_fn) == 8
    assert_restored(module, functional, inputs, states, original_input)


@pytest.mark.parametrize('frozen_name', EVALUATE.PARAMETER_NAMES)
def test_fixed_parameter_candidate_is_rejected_and_state_restored(frozen_name):
    module, functional, inputs, expected = models()
    states, original_input = snapshot(module, functional), inputs[0].clone()
    frozen = states[1][frozen_name]
    parameter_index = EVALUATE.PARAMETER_NAMES.index(frozen_name) + 1

    def fixed_parameter(*args):
        args = list(args)
        args[parameter_index] = frozen
        return FUNCTIONAL.module_fn(*args)

    with pytest.raises(ValueError, match=frozen_name):
        check(module, functional, inputs, expected, fixed_parameter)
    assert_restored(module, functional, inputs, states, original_input)


def test_candidate_exception_restores_all_parameters_and_input():
    module, functional, inputs, expected = models()
    states, original_input = snapshot(module, functional), inputs[0].clone()

    def raises_after_mutation(*args):
        with torch.no_grad():
            args[0].add_(1)
            args[1].fill_(99)
        raise RuntimeError('candidate failed')

    with pytest.raises(RuntimeError, match='candidate failed'):
        check(module, functional, inputs, expected, raises_after_mutation)
    assert_restored(module, functional, inputs, states, original_input)


def test_rectangular_geometry_controls_use_oracle_without_changing_state():
    assert EVALUATE.GEOMETRY_CONTROL_SHAPES == (
        (1, 4, 2, 3), (3, 4, 2, 5), (5, 4, 4, 2))
    module, functional, inputs, _ = models()
    states, original_input = snapshot(module, functional), inputs[0].clone()
    assert EVALUATE.check_geometry_variants(
        module, functional, FUNCTIONAL.module_fn, compare, 1e-4, 1e-5) == 3
    assert_restored(module, functional, inputs, states, original_input)


def test_rectangular_geometry_rejects_wrong_candidate_output():
    module, functional, _, _ = models()

    def wrong_geometry(*args):
        return FUNCTIONAL.module_fn(*args) + 0.25

    with pytest.raises(ValueError, match='geometry .* differs from PyTorch reference'):
        EVALUATE.check_geometry_variants(
            module, functional, wrong_geometry, compare, 1e-4, 1e-5)


def test_rectangular_geometry_rejects_stale_changed_input_output():
    module, functional, _, _ = models()
    cached = {}

    def stale_output(*args):
        shape = tuple(args[0].shape)
        cached.setdefault(shape, FUNCTIONAL.module_fn(*args).detach().clone())
        return cached[shape]

    with pytest.raises(ValueError, match='geometry .* differs from PyTorch reference'):
        EVALUATE.check_geometry_variants(
            module, functional, stale_output, compare, 1e-4, 1e-5)


@pytest.mark.parametrize('mutated', ('input', 'state'))
def test_rectangular_geometry_rejects_mutation_and_restores_state(mutated):
    module, functional, inputs, _ = models()
    states, original_input = snapshot(module, functional), inputs[0].clone()

    def mutating_candidate(*args):
        output = FUNCTIONAL.module_fn(*args)
        with torch.no_grad():
            args[0 if mutated == 'input' else 1].add_(1)
        return output

    with pytest.raises(ValueError, match='mutated an input|changed model state'):
        EVALUATE.check_geometry_variants(
            module, functional, mutating_candidate, compare, 1e-4, 1e-5)
    assert_restored(module, functional, inputs, states, original_input)
