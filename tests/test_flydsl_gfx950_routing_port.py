"""CPU regression controls for the task-owned top-K measured-output checks."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/topk_gating_softmax_kernel'


def _checks():
    spec = importlib.util.spec_from_file_location('gfx950_routing_checks', TASK / 'scripts/replay_checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference():
    import torch
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_topk')
    namespace = {'torch': torch, 'DTYPE_FP32': torch.float32}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'original-routing-reference', 'exec'), namespace)
    return namespace['reference_topk']


def test_original_routing_cases_and_comparisons_are_retained():
    assert hashlib.sha256((TASK / 'cases.json').read_bytes()).hexdigest() == '09971067c372d872380d679b604cbbc547fe1409f0f524bb9b9a9c4c73f6cb59'
    prefix = (TASK / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import verify_topk_timed_run, require_topk_output',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    prefix = prefix.replace('    require_topk_output((topk_weights_dev, topk_indices_dev, token_expert_indices_dev),\n'
                            '                        gating_dev, topk, dtype_str, renormalize, reference_topk)\n\n', '')
    assert hashlib.sha256(prefix.encode()).hexdigest() == 'b716aeec4aec87f695a3bad645a0582ba714c190afc2437125e9176e677655ea'


@pytest.mark.parametrize('failure', ['nan', 'range', 'duplicate', 'weight_association', 'token_index', 'dtype'])
def test_routing_rejects_broken_output_contract(failure):
    import torch
    checks, reference = _checks(), _reference()
    gating = torch.tensor([[1.5, .4, -.2, -3.]])
    _, weights, indices, tei = reference(gating, 2)
    if failure == 'nan': weights[0, 0] = float('nan')
    if failure == 'range': indices[0, 0] = 4
    if failure == 'duplicate': indices[0, 0] = indices[0, 1]
    if failure == 'weight_association': weights = weights.flip(1)
    if failure == 'token_index': tei[0, 0] = -1
    if failure == 'dtype': indices = indices.long()
    with pytest.raises(AssertionError):
        checks.require_topk_output((weights, indices, tei), gating, 2, 'f32', True, reference)


def test_equally_probable_boundary_expert_remains_allowed():
    import torch
    checks, reference = _checks(), _reference()
    gating = torch.tensor([[2., 1., 1., 1.]])
    _, weights, indices, tei = reference(gating, 2)
    chosen = int(indices[0, 1])
    indices[0, 1] = next(i for i in (1, 2, 3) if i != chosen)
    checks.require_topk_output((weights, indices, tei), gating, 2, 'f32', True, reference)


@pytest.mark.parametrize('behavior', ['correct', 'stale', 'weights_only', 'no_write', 'unbound'])
def test_routing_requires_all_outputs_from_exact_replay(behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun
    checks, reference = _checks(), _reference()
    gating = torch.tensor([[1.5, .4, -.2, -3.]])
    original = gating.clone()
    original_outputs = tuple(v.clone() for v in reference(gating, 2)[1:])
    outputs = tuple(v.clone() for v in original_outputs)

    def replay():
        expected = original_outputs if behavior == 'stale' else reference(gating, 2)[1:]
        if behavior != 'no_write':
            for i, (out, value) in enumerate(zip(outputs, expected)):
                if behavior == 'weights_only' and i > 0: continue
                out.copy_(value)
        return outputs

    timed = TimedRun()
    if behavior != 'unbound': timed._bind(replay, outputs)
    if behavior == 'correct':
        assert checks.verify_topk_timed_run(timed, gating, original, 2, 'f32', True, reference)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            checks.verify_topk_timed_run(timed, gating, original, 2, 'f32', True, reference)
    torch.testing.assert_close(gating, original)
