"""Full RoPE/cache replay controls; CPU tests do not qualify GPU support."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/fused_rope_cache_kernel'
CONTRACT = {'cases': 'b15f1cce2d65a29677324f7efec84cca57a549fc88283dbbc0f40d0a39d248bb', 'harness_prefix': 'b6f99ebab957a107268286619b6c63bc1c72772033b954e4075636f7446eb013', 'operator_ast': {'kernel.py': 'db0885eade59dd325177d5ba05a14ef02d25ff1ffb1a2818b112eb0892b232b1', 'kernels/__init__.py': 'ffba730c73ed3fb7eda50ac01930a9251bca46e6e069d7fb267bc4660f06382b', 'kernels/kernels_common.py': '607c63eb01c5567a31dedb4752f17b1274f111b84eabb1b26a441574bd1b9a6a'}}


def test_original_operator_cases_and_numerical_timing_harness_preserved():
    assert hashlib.sha256((ROOT / 'cases.json').read_bytes()).hexdigest() == CONTRACT['cases']
    prefix = (ROOT / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import prepare_check, verify_timed_run',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    assert hashlib.sha256(prefix.encode()).hexdigest() == CONTRACT['harness_prefix']
    for name, digest in CONTRACT['operator_ast'].items():
        tree = ast.parse((ROOT / name).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == digest


@pytest.mark.parametrize('behavior', ['correct', 'wrong_q', 'wrong_k', 'wrong_key_cache', 'wrong_value_cache',
                                    'stale_replay', 'no_cache_write', 'damaged_unmapped', 'input_mutation'])
def test_all_rope_outputs_and_preserved_cache_state_are_checked(behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun
    spec = importlib.util.spec_from_file_location('rope_replay', ROOT / 'scripts/replay_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    q = torch.tensor([[[1., 2.]], [[3., 4.]]])
    inp = dict(Q=q, K=q.clone()*2, V=q.clone()*3, positions=torch.tensor([0, 1]),
               cos_cache=torch.ones(2, 1), sin_cache=torch.zeros(2, 1),
               slot_mapping=torch.tensor([0, 3]), k_scale=torch.ones(1), v_scale=torch.ones(1),
               BS=2, key_cache=torch.full((3, 2, 1, 2), 7.), value_cache=torch.full((3, 2, 1, 2), 8.),
               Q_out=torch.empty_like(q), K_out=torch.empty_like(q))
    # A CPU identity RoPE exercises the multi-output/cache controller independently of FlyDSL.
    reference = lambda x, cos, sin, positions: x.clone()
    check = checks.prepare_check(inp, reference, 1e-2, 1e-2)
    outputs = tuple(inp[name] for name in ('Q_out', 'K_out', 'key_cache', 'value_cache'))
    for output, expected in zip(outputs, check['expected']):
        output.copy_(expected)
    def replay():
        expected = check['expected'] if behavior == 'stale_replay' else check['reference']()
        for index, (output, ref) in enumerate(zip(outputs, expected)):
            if behavior == 'no_cache_write' and index >= 2:
                continue
            output.copy_(ref)
        if behavior == 'damaged_unmapped':
            outputs[2][-1].zero_()
        return outputs
    timed = TimedRun()
    timed._bind(replay, outputs)
    names = ['wrong_q', 'wrong_k', 'wrong_key_cache', 'wrong_value_cache']
    if behavior in names:
        outputs[names.index(behavior)].add_(1)
    if behavior == 'input_mutation':
        inp['Q'].add_(1)
    if behavior == 'correct':
        assert checks.verify_timed_run(timed, **check)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises(AssertionError):
            checks.verify_timed_run(timed, **check)
