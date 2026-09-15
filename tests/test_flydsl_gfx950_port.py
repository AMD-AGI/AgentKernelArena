"""CPU dependency compatibility and protected-contract checks; no GPU claims."""
import ast
import builtins
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks' / 'flydsl2flydsl'
CONTRACTS = {
    'rmsnorm_kernel': ('5bcb79ba0ea702e41a179b7d1dacfb3cfd4838df84a1934b10a742c0cef8a860',
                       '5b99ff047a8da242779792c6fd7ef22cae13938ba3be3f6471a89364b30bc93d'),
    'softmax_kernel': ('846f2c44f36fb263ddb20baf8580609050a1ddc68ac2ed02cfa9803b1169ce40',
                      '23698d96c2281044c172ca9152af0eec6f06b6dc08bb4ad1c17cbc415cd2e0a9'),
}


@pytest.mark.parametrize('task', CONTRACTS)
def test_port_preserves_declared_cases_and_original_numerical_timing_harness(task):
    for name, expected in zip(('cases.json', 'test_kernel_harness.py'), CONTRACTS[task]):
        assert hashlib.sha256((ROOT / task / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('task', CONTRACTS)
@pytest.mark.parametrize('runtime', ['legacy', 'current', 'broken_dependency'])
def test_vector_helpers_support_both_dependency_layouts_without_masking_errors(task, runtime):
    tree = ast.parse((ROOT / task / 'kernel.py').read_text())
    block = next(node for node in tree.body if isinstance(node, ast.Try)
                 and any(isinstance(child, ast.ImportFrom) and child.module == 'flydsl.expr.vector'
                         for child in node.body))
    legacy = SimpleNamespace(ReductionOp=object(), full=object())
    current = SimpleNamespace(ReductionOp=object(), full=object())
    attempted = []

    def import_provider(name, *args, **kwargs):
        attempted.append(name)
        if name == 'flydsl.expr.vector':
            if runtime == 'legacy':
                return legacy
            missing = 'missing_mlir_dependency' if runtime == 'broken_dependency' else name
            raise ModuleNotFoundError('Missing dependency', name=missing)
        assert name == 'flydsl.expr.typing'
        return current

    namespace = {'__builtins__': {**vars(builtins), '__import__': import_provider}}
    code = compile(ast.Module(body=[block], type_ignores=[]), 'vector-provider-compatibility', 'exec')
    if runtime == 'broken_dependency':
        with pytest.raises(ModuleNotFoundError) as error:
            exec(code, namespace)
        assert error.value.name == 'missing_mlir_dependency'
        assert attempted == ['flydsl.expr.vector']
    else:
        exec(code, namespace)
        expected = legacy if runtime == 'legacy' else current
        assert namespace['ReductionOp'] is expected.ReductionOp
        assert namespace['full'] is expected.full


@pytest.mark.parametrize('task', CONTRACTS)
def test_port_retains_original_architecture_and_adds_explicit_target(task):
    from src.task_spec import load_task_spec
    spec = load_task_spec(ROOT / task / 'config.yaml', task_id='flydsl2flydsl/' + task)
    assert spec.to_mapping()['platform_support']['required_arch'] == ['gfx942', 'gfx950']
    assert spec.to_mapping()['candidate']['editable'] == ['kernel.py']
    assert spec.to_mapping()['baseline']['kind'] == 'initial_candidate'
    assert spec.to_mapping()['baseline']['correctness_policy'] == 'required'
