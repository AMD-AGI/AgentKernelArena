"""Protected-contract and actual replay failure controls for owned gfx950 ports."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl'
CONTRACTS = {'flash_attn_func_kernel': {'cases': 'a084aaf1280d9cb59865b925cc2b8e6e553a9ee92ecadbdc72b6cff37e7759a9', 'harness_prefix': '10a7007625b7a9f47bb55714de459fdeee9eea1d2249533552e65368b31f1792', 'operator_ast': {'kernel.py': 'e1d8b0a9a1cd1a7bab8b3d5b7de38a60c407efacd5e1945be2642414d977c493', 'kernels/__init__.py': 'ffba730c73ed3fb7eda50ac01930a9251bca46e6e069d7fb267bc4660f06382b', 'kernels/kernels_common.py': '607c63eb01c5567a31dedb4752f17b1274f111b84eabb1b26a441574bd1b9a6a'}}, 'hgemm_splitk_kernel': {'cases': 'f15156d5c9c6d6d0aa66e7aff29fea757903b7a3d650cb8c885460d2435ddfdd', 'harness_prefix': '2d16ea6f2efb04142f6e5dc81a2a25a67838ad5eae905900a36a36973c3e67f7', 'operator_ast': {'kernel.py': 'acdd1322d934bed1db4dcc0e42ef82df43f30439fe5d8fe37949b41aa25cd2ff', 'kernels/__init__.py': 'ffba730c73ed3fb7eda50ac01930a9251bca46e6e069d7fb267bc4660f06382b', 'kernels/tensor_shim.py': '31d8f9612a37ce75ee4cbe2df68854bd7b68eafa360a7085ae7f166ac934efb2'}}}


@pytest.mark.parametrize('task', CONTRACTS)
def test_port_preserves_operator_ast_cases_and_original_harness(task):
    path = ROOT / task
    expected = CONTRACTS[task]
    assert hashlib.sha256((path / 'cases.json').read_bytes()).hexdigest() == expected['cases']
    prefix = (path / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import verify_timed_run, compare_output',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    assert hashlib.sha256(prefix.encode()).hexdigest() == expected['harness_prefix']
    for name, digest in expected['operator_ast'].items():
        tree = ast.parse((path / name).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == digest


def load_checks(task):
    spec = importlib.util.spec_from_file_location('port_replay', ROOT / task / 'scripts/replay_checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('task', CONTRACTS)
@pytest.mark.parametrize('behavior', ['correct', 'wrong_measured', 'stale_replay', 'no_replay_write', 'mutated_input'])
def test_replay_requires_measured_and_perturbed_correct_output(task, behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun

    checks = load_checks(task)
    x = torch.tensor([[0.25, -1.0, 3.0]])
    original = x.clone()
    reference = lambda: x * 2
    expected = reference()
    output = expected.clone()
    def replay():
        if behavior == 'stale_replay':
            output.copy_(expected)
        elif behavior != 'no_replay_write':
            output.copy_(reference())
        return output
    timed = TimedRun()
    timed._bind(replay, output)
    if behavior == 'wrong_measured':
        output.add_(1)
    if behavior == 'mutated_input':
        x.add_(1)
    compare = lambda actual, ref: checks.compare_output(actual, ref, 1e-2)
    if task == 'hgemm_splitk_kernel':
        compare = lambda actual, ref: checks.compare_output(actual, ref, 1e-2, x.dtype)
    kwargs = dict(inputs=(x,), originals=(original,), expected=expected, perturb=lambda: x.neg_(),
                  reference=reference, compare=compare)
    if behavior == 'correct':
        assert checks.verify_timed_run(timed, **kwargs)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            checks.verify_timed_run(timed, **kwargs)
    if behavior != 'mutated_input':
        torch.testing.assert_close(x, original)


@pytest.mark.parametrize('task', CONTRACTS)
@pytest.mark.parametrize('bad', ['nan', 'inf', 'wrong', 'dtype', 'shape'])
def test_original_numerical_gate_rejects_invalid_measured_values(task, bad):
    import torch
    checks = load_checks(task)
    ref = torch.tensor([[1., -2.]])
    out = ref.clone()
    if bad in ('nan', 'inf'):
        out[0, 0] = float(bad)
    elif bad == 'wrong':
        out.add_(1)
    elif bad == 'dtype':
        out = out.double()
    else:
        out = out.flatten()
    with pytest.raises(AssertionError):
        if task == 'flash_attn_func_kernel':
            checks.compare_output(out, ref, 1e-2)
        else:
            checks.compare_output(out, ref, 1e-2, torch.float32)
