"""MoE replay rejects incomplete routes, wrong associations, stale output, and dirty state."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/moe_sorting_kernel'
CONTRACT = {'cases': 'd61592893aa2847e7f72fe0c60dbd8c280e36dc15908582cf1e605d75169d40f', 'harness_prefix': 'dc9a95c7b58dfed18c33d304a345df081435fa14ae0ac6685adee258ffd3e7c8', 'operator_ast': {'kernel.py': '957d7a3ed1f7a92d4d84f8a2f31b4c03a93ce5fd39df62dd5eda309b4b4c9a1d', 'kernels/__init__.py': 'ffba730c73ed3fb7eda50ac01930a9251bca46e6e069d7fb267bc4660f06382b', 'kernels/kernels_common.py': '607c63eb01c5567a31dedb4752f17b1274f111b84eabb1b26a441574bd1b9a6a', 'kernels/moe_sorting_kernel.py': '957d7a3ed1f7a92d4d84f8a2f31b4c03a93ce5fd39df62dd5eda309b4b4c9a1d'}}


def test_original_operator_cases_and_harness_preserved():
    assert hashlib.sha256((ROOT / 'cases.json').read_bytes()).hexdigest() == CONTRACT['cases']
    prefix = (ROOT / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import prepare_check, verify_timed_run, compare_outputs',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    prefix = prefix.replace('    compare_outputs((gpu_ids, gpu_w, gpu_eids, gpu_nvalid, gpu_moe_buf),\n'
                            '                    (ref_ids, ref_w, ref_eids, ref_nvalid),\n'
                            '                    token_count=T, topk=topk, unit_size=unit_size)\n\n', '')
    assert hashlib.sha256(prefix.encode()).hexdigest() == CONTRACT['harness_prefix']
    for name, digest in CONTRACT['operator_ast'].items():
        tree = ast.parse((ROOT / name).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == digest


def reference(ids, weights, experts, unit):
    import torch
    tree = ast.parse((ROOT / 'test_kernel_harness.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'moe_sorting_reference')
    scope = {'torch': torch, 'UNIT_SIZE': unit}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'protected_moe_reference', 'exec'), scope)
    return scope['moe_sorting_reference'](ids, weights, experts, unit)


@pytest.mark.parametrize('behavior', ['correct', 'nan_weights', 'wrong_weights', 'wrong_ids', 'wrong_experts',
                                    'wrong_counts', 'dirty_buffer', 'stale_replay', 'no_replay_write'])
def test_all_defined_moe_outputs_required(behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun
    spec = importlib.util.spec_from_file_location('moe_checks', ROOT / 'scripts/replay_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    ids = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32)
    weights = torch.tensor([[.1, .2], [.3, .4], [.5, .6]])
    unit, experts = 4, 3
    expected = reference(ids, weights, experts, unit)
    outputs = tuple(value.clone() for value in expected) + (torch.zeros(3, 4096, dtype=torch.bfloat16),)
    check = checks.prepare_check(ids, weights, experts, unit, reference)
    def replay():
        assert (weights >= 0).all() and (weights <= 1).all()
        new = expected if behavior == 'stale_replay' else reference(ids, weights, experts, unit)
        if behavior != 'no_replay_write':
            for actual, ref in zip(outputs[:4], new):
                actual.copy_(ref)
            outputs[4].zero_()
        return outputs
    timed = TimedRun()
    timed._bind(replay, outputs)
    if behavior == 'nan_weights':
        outputs[1][0] = float('nan')
    elif behavior == 'wrong_weights':
        outputs[1][0] += .1
    elif behavior == 'wrong_ids':
        outputs[0][0] = outputs[0][1]
    elif behavior == 'wrong_experts':
        outputs[2][0] = 1
    elif behavior == 'wrong_counts':
        outputs[3][0] -= 1
    elif behavior == 'dirty_buffer':
        outputs[4][0, 0] = 1
    if behavior == 'correct':
        assert checks.verify_timed_run(timed, **check)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises(AssertionError):
            checks.verify_timed_run(timed, **check)
