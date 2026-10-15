"""Finite one-step/cache contract, independent of Triton or GPU qualification."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_linear_attn_decode'
GUARD = 'validate_inputs(q, k, v, kv_caches, slope_rate, slot_idx)'


@pytest.fixture
def task(monkeypatch):
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(TASK))
    monkeypatch.chdir(TASK)
    spec = importlib.util.spec_from_file_location('_linear_contract_test', TASK / 'scripts/task_runner.py')
    harness = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, harness)
    spec.loader.exec_module(harness)
    return harness, sys.modules['scripts.input_contract']


def inputs(slots=(2, -1, 0), *, slot_dtype=torch.int32):
    batch = len(slots)
    return dict(q=torch.ones(batch, 2, 1, 32, dtype=torch.float16),
                k=torch.ones(batch, 2, 1, 32, dtype=torch.float16),
                v=torch.ones(batch, 2, 1, 32, dtype=torch.float16),
                kv_caches=torch.ones(5, 2, 32, 32),
                slope_rate=torch.tensor([.1, .2]),
                slot_idx=torch.tensor(slots, dtype=slot_dtype))


@pytest.mark.parametrize('slots', [(2, 0), (2, -1, 0), (2, -1, -1, 0), (-1, -1), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_distinct_active_slots_and_repeated_padding_are_valid(task, slots, dtype):
    harness, contract = task
    kw = inputs(slots, slot_dtype=dtype)
    original = {key: value.clone() for key, value in kw.items()}
    contract.validate_inputs(**kw)
    assert all(torch.equal(value, original[key]) for key, value in kw.items())
    harness.reference_linear_attn_decode(**kw)
    inactive = set(range(5)) - {slot for slot in slots if slot >= 0}
    for slot in inactive:
        assert torch.equal(kw['kv_caches'][slot], original['kv_caches'][slot])


@pytest.mark.parametrize('fault', [
    'duplicate', 'duplicate_with_padding', 'negative', 'out_of_range', 'slot_dtype',
    'slot_rank', 'slot_batch', 'multiple_tokens', 'key_shape', 'value_shape',
    'cache_shape', 'slope_shape', 'non_power_two_dim', 'value_tail', 'block_zero', 'block_non_power_two',
])
def test_invalid_domain_rejected_before_reference_mutates_cache(task, fault):
    harness, contract = task
    kw = inputs()
    if fault == 'duplicate': kw['slot_idx'] = torch.tensor([0, 0, 1], dtype=torch.int32)
    if fault == 'duplicate_with_padding': kw['slot_idx'] = torch.tensor([0, -1, 0], dtype=torch.int32)
    if fault == 'negative': kw['slot_idx'][0] = -2
    if fault == 'out_of_range': kw['slot_idx'][0] = 5
    if fault == 'slot_dtype': kw['slot_idx'] = kw['slot_idx'].float()
    if fault == 'slot_rank': kw['slot_idx'] = kw['slot_idx'].reshape(3, 1)
    if fault == 'slot_batch': kw['slot_idx'] = kw['slot_idx'][:2]
    if fault == 'multiple_tokens':
        for key in ('q', 'k', 'v'): kw[key] = kw[key].expand(3, 2, 2, 32)
    if fault == 'key_shape': kw['k'] = kw['k'][:2]
    if fault == 'value_shape': kw['v'] = kw['v'][:, :1]
    if fault == 'cache_shape': kw['kv_caches'] = kw['kv_caches'][:, :, :16]
    if fault == 'slope_shape': kw['slope_rate'] = kw['slope_rate'].reshape(2, 1)
    if fault == 'non_power_two_dim':
        for key in ('q', 'k'): kw[key] = kw[key][..., :31]
        kw['kv_caches'] = kw['kv_caches'][:, :, :31]
    if fault == 'value_tail':
        kw['v'] = torch.ones(3, 2, 1, 33)
        kw['kv_caches'] = torch.ones(5, 2, 32, 33)
    if fault.startswith('block_'): kw['BLOCK_SIZE'] = 0 if fault == 'block_zero' else 3
    before = {key: value.clone() for key, value in kw.items() if isinstance(value, torch.Tensor)}
    with pytest.raises(ValueError):
        contract.validate_inputs(**kw)
    if not fault.startswith('block_'):
        with pytest.raises(ValueError):
            harness.reference_linear_attn_decode(**kw)
    assert all(torch.equal(kw[key], value) for key, value in before.items())


def test_duplicate_slots_have_two_different_update_semantics_and_are_rejected(task):
    harness, _ = task
    # Scalar first-feature example of the unchanged reference recurrence:
    # old=4, decay=.5, outer products=1 then 3. A sequential reference yields
    # 3 then 4.5, whereas two independent batch programs read old=4 and yield
    # 3 and 5. Neither is a defined ordering for duplicate-slot GPU updates.
    old = torch.tensor(4.)
    outer = torch.tensor([1., 3.])
    sequential = []
    state = old.clone()
    for value in outer:
        state = value + .5 * state
        sequential.append(state.clone())
    independent = outer + .5 * old
    assert torch.equal(torch.stack(sequential), torch.tensor([3., 4.5]))
    assert torch.equal(independent, torch.tensor([3., 5.]))
    assert not torch.equal(torch.stack(sequential), independent)
    with pytest.raises(ValueError, match='unique'):
        harness.reference_linear_attn_decode(**inputs((0, 0)))


def test_all_five_original_draws_remain_valid_and_reference_math_is_unchanged(task):
    harness, contract = task
    assert harness.TEST_SHAPES == [(1, 4, 64, 64), (4, 8, 64, 64), (2, 16, 128, 128),
                                   (8, 4, 64, 32), (1, 32, 64, 64)]
    for index, (batch, heads, dim, value_dim) in enumerate(harness.TEST_SHAPES):
        torch.manual_seed(42 + index)
        slope = torch.rand(heads) * .1 + .01
        slots = torch.arange(batch, dtype=torch.int32)
        q = torch.randn(batch, heads, 1, dim, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn(batch, heads, 1, value_dim, dtype=torch.float16)
        cache = (torch.randn(batch, heads, dim, value_dim, dtype=torch.float16) * .1).float()
        args = (q, k, v, cache, slope, slots)
        contract.validate_inputs(*args)
        state = (k[:, :, 0].float()[..., None] * v[:, :, 0].float()[..., None, :]
                 + torch.exp(-slope)[None, :, None, None] * cache)
        expected = (q[:, :, 0].float()[..., None] * state).sum(dim=2)[:, :, None]
        actual = harness.reference_linear_attn_decode(*args)
        assert torch.equal(cache, state)
        assert torch.equal(actual, expected)


def test_repeated_padding_control_keeps_original_controls_and_full_cache_checks(task):
    harness, _ = task
    cases = list(harness.semantic_controls.control_cases('cpu'))
    assert [case['kwargs']['slot_idx'].tolist() for case in cases] == [[2, 0], [2, -1, 0], [2, -1, -1, 0]]
    def operation(**kw):
        state = kw['kv_caches'].float().clone()
        output = harness.reference_linear_attn_decode(**{**kw, 'kv_caches': state})
        kw['kv_caches'].copy_(state)
        return output.to(kw['q'].dtype)
    rows = harness.semantic_controls.run_controls(SimpleNamespace(linear_attn_decode_forward=operation), 'cpu')
    assert len(rows) == 3 and all(row['full_cache_checked'] and row['active_output_checked'] for row in rows)


def test_reference_guard_is_additive_and_kernel_wrapper_generated_timer_unchanged(task):
    _, _ = task
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    assert digest((TASK / 'source/triton_linear_attn_decode.py').read_text()) == (
        'e8429e9326718bff4af57ea980b1c2b415af0bb4e24b6aa65eb9754dba3b5af5')
    text = (TASK / 'scripts/task_runner.py').read_text()
    tree = ast.parse(text)
    reference = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'reference_linear_attn_decode')
    guards = [node for node in reference.body if isinstance(node, ast.Expr)
              and ast.unparse(node.value) == GUARD]
    assert len(guards) == 1
    reference.body.remove(guards[0])
    assert digest(ast.dump(reference, include_attributes=False)) == (
        '65b9a17df6b52f94770248615a953957b8f6a5b1cc1379791ab950ab2f4a511d')
    assert digest(text.split('# >>> AKA-GENERATED:')[1].split('# <<< AKA-GENERATED <<<')[0]) == (
        'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a')
    timed = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_bench_fn')
    assert not any(isinstance(node, ast.Name) and node.id == 'validate_inputs' for node in ast.walk(timed))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == '_benchmark_cuda_graph_or_events']
    assert len(calls) == 1
    assert [(key.arg, ast.unparse(key.value)) for key in calls[0].keywords] == [
        ('warmup', 'WARMUP_ITERATIONS'), ('repetition', 'BENCHMARK_ITERATIONS'),
        ('prepare_fn', 'lambda: kv_work.copy_(kv_caches)'), ('timed_run', 'timed')]


def test_actual_performance_runner_rejects_duplicate_setup_before_capture(task, monkeypatch):
    harness, _ = task
    for name in ('randn', 'rand', 'arange'):
        original = getattr(torch, name)
        def factory(*args, _original=original, _name=name, **kwargs):
            kwargs['device'] = 'cpu'
            result = _original(*args, **kwargs)
            if _name == 'arange':
                result.zero_()
            return result
        monkeypatch.setattr(torch, name, factory)
    monkeypatch.setattr(harness, 'TEST_SHAPES', [(2, 2, 4, 32)])
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid slot setup reached candidate/capture')
    monkeypatch.setattr(harness, 'load_module', lambda: SimpleNamespace(linear_attn_decode_forward=forbidden))
    monkeypatch.setattr(harness, '_benchmark_cuda_graph_or_events', forbidden)
    rows = harness.run_performance()
    assert len(rows) == 1 and rows[0]['execution_time_ms'] == -1
    assert 'unique' in rows[0]['error']


def test_new_input_guard_is_a_protected_task_file(task, tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    from src.task_spec import load_task_spec
    workspace = tmp_path / 'workspace'
    shutil.copytree(TASK, workspace)
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_linear_attn_decode')
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    guard = workspace / 'scripts/input_contract.py'
    guard.write_text(guard.read_text().replace("raise ValueError('Active slot indices must be unique for one decode step')", 'return'))
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(snapshot, discard_added=False)


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_both_roles_require_same_protected_manifest_preconditions(task, monkeypatch, role):
    harness, contract = task
    spec = importlib.util.spec_from_file_location('_linear_adapter_test', TASK / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original = json.loads((TASK / 'workloads.json').read_text())
    assert original['input_contract'] == contract.INPUT_CONTRACT
    fake = SimpleNamespace(TEST_SHAPES=harness.TEST_SHAPES, INPUT_CONTRACT=contract.INPUT_CONTRACT,
                           run_semantic_controls=lambda: [], run_correctness=lambda **kw: (True, None))
    monkeypatch.setattr(adapter, 'load_harness', lambda: fake)
    assert adapter.evaluate(role, 'correctness')['status'] == 'PASS'
    changed = copy.deepcopy(original)
    changed['input_contract']['active_slots'] = 'duplicates_allowed'
    monkeypatch.setattr(adapter, 'load_manifest', lambda: changed)
    result = adapter.evaluate(role, 'correctness')
    assert result['status'] == 'FAIL' and 'preconditions' in result['reason']
