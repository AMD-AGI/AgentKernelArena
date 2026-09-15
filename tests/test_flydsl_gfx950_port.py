"""CPU dependency compatibility and protected-contract checks; no GPU claims."""
import ast
import builtins
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks' / 'flydsl2flydsl'
CONTRACTS = {
    'rmsnorm_kernel': ('5bcb79ba0ea702e41a179b7d1dacfb3cfd4838df84a1934b10a742c0cef8a860',
                       '7998d9e3555b7a43299d83df69afa3f384abc926c7c1e9402ab0451b0593964e'),
    'softmax_kernel': ('846f2c44f36fb263ddb20baf8580609050a1ddc68ac2ed02cfa9803b1169ce40',
                      '2b9bc6b1f9a9f40f915d5e930e436a15aafc103d87f52d3c365e1ad73d22adc2'),
    'layernorm_kernel': ('5bcb79ba0ea702e41a179b7d1dacfb3cfd4838df84a1934b10a742c0cef8a860',
                        '991415f767641a852b2c3d883e679b117b0e9c04ec2395c45b1d135ddc7624a3'),
}


@pytest.mark.parametrize('task', CONTRACTS)
def test_port_preserves_declared_cases_and_original_numerical_timing_harness(task):
    cases, original_harness = CONTRACTS[task]
    assert hashlib.sha256((ROOT / task / 'cases.json').read_bytes()).hexdigest() == cases
    prefix = (ROOT / task / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import verify_timed_run',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    assert hashlib.sha256(prefix.encode()).hexdigest() == original_harness


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


@pytest.mark.parametrize('task', ['rmsnorm_kernel', 'softmax_kernel', 'layernorm_kernel', 'topk_gating_softmax_kernel'])
def test_port_retains_original_architecture_and_adds_explicit_target(task):
    from src.task_spec import load_task_spec
    spec = load_task_spec(ROOT / task / 'config.yaml', task_id='flydsl2flydsl/' + task)
    assert spec.to_mapping()['platform_support']['required_arch'] == ['gfx942', 'gfx950']
    assert spec.to_mapping()['candidate']['editable'] == ['kernel.py']
    assert spec.to_mapping()['baseline']['kind'] == 'initial_candidate'
    assert spec.to_mapping()['baseline']['correctness_policy'] == 'required'


@pytest.mark.parametrize('task', CONTRACTS)
@pytest.mark.parametrize('behavior', ['correct', 'wrong_measured', 'stale_replay', 'no_replay_write',
                                    'mutated_input', 'unbound'])
def test_measured_output_and_same_replay_are_required(task, behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun

    spec = importlib.util.spec_from_file_location('owned_port_replay', ROOT / task / 'scripts/replay_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    x = torch.tensor([[0.25, -1.0, 3.0]])
    original = x.clone()
    reference = (lambda: torch.softmax(x, dim=-1)) if task == 'softmax_kernel' else (
        lambda: x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5))
    if task == 'layernorm_kernel':
        reference = lambda: torch.nn.functional.layer_norm(x, (x.shape[-1],))
    expected = reference()
    output = expected.clone()
    calls = []

    def replay():
        calls.append('same captured unit')
        if behavior == 'stale_replay':
            output.copy_(expected)
        elif behavior != 'no_replay_write':
            output.copy_(reference())
        return output

    timed = TimedRun()
    if behavior != 'unbound':
        timed._bind(replay, output)
    if behavior == 'wrong_measured':
        output.add_(100)
    if behavior == 'mutated_input':
        x.add_(1)
    kwargs = dict(inputs=(x,), originals=(original,), expected=expected, perturb=lambda: x.neg_(),
                  reference=reference, compare=lambda actual, ref: torch.testing.assert_close(actual, ref))
    if behavior == 'correct':
        metadata = checks.verify_timed_run(timed, **kwargs)
        assert metadata['timed_output_correctness'] == metadata['replay_correctness'] == 'PASS'
        assert calls == ['same captured unit']
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            checks.verify_timed_run(timed, **kwargs)
    if behavior != 'mutated_input':
        torch.testing.assert_close(x, original)
