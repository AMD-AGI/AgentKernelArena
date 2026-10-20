"""CPU checks for the paged-prefix supported-domain correctness controls.

These tests exercise the protected manifest binding, the oracle and the public
wrapper guard with CPU doubles. They do not run Triton or qualify the task on
GPU; that requires a fresh task_validator run on compatible hardware.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'tasks/triton2triton/vllm/triton_paged_prefix_prefill'
TASK = ROOT / RELATIVE
BASE = '099e44cf6deb74131d2cfc96a3a055949a2edbdc'
ORIGINAL_CONTROLS = ('ragged_permuted_window_scale', 'ragged_nonpower2_window_scale')
NAMES = ('q', 'k', 'v', 'o', 'k_cache', 'v_cache', 'b_loc', 'b_start_loc', 'b_seq_len')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def task(monkeypatch):
    # The runner changes directory on import; keep each test rooted.
    monkeypatch.chdir(ROOT)
    checks = load(TASK / '_arena_checks.py', '_paged_domain_checks')
    harness = load(TASK / 'scripts/task_runner.py', '_paged_domain_harness')
    monkeypatch.chdir(ROOT)
    return checks, harness


DOMAIN = load(TASK / '_domain_controls.py', '_paged_domain_table').DOMAIN_CASES
CASES = tuple(DOMAIN)


def base_bytes(relative):
    return subprocess.check_output(['git', 'show', f'{BASE}:{RELATIVE}/{relative}'], cwd=ROOT)


def naive_attention(args, kwargs, *, ignore_context=False, future_tokens=0,
                    unshifted=False, default_scale=False, no_window=False):
    """Token-by-token FP64 oracle with interval masks, independent of the shared one."""
    q, k, v, _, k_cache, v_cache, b_loc, starts, seq_lens = (value.cpu() for value in args)
    dim, page, width = q.shape[-1], k_cache.shape[3], k_cache.shape[4]
    scale = dim ** -0.5 if default_scale else kwargs.get('sm_scale', dim ** -0.5)
    window = 0 if no_window else max(kwargs.get('sliding_window') or 0, 0)
    blocks, kv_heads = k_cache.shape[:2]
    # Unpack [block, kv, dim/x, slot, x] to [block, kv, slot, dim] by the layout definition.
    dense_k = k_cache.double().permute(0, 1, 3, 2, 4).reshape(blocks, kv_heads, page, dim)
    dense_v = v_cache.double().permute(0, 1, 3, 2)
    group = q.shape[1] // k.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float64)
    for batch in range(seq_lens.numel()):
        lo, hi = int(starts[batch]), int(starts[batch + 1])
        context = int(seq_lens[batch]) - (hi - lo)
        keys, values = [], []
        for token in range(context):
            physical = int(b_loc[batch, token // page])
            keys.append(dense_k[physical, :, token % page])
            values.append(dense_v[physical, :, token % page])
        keys += [row.double() for row in k[lo:hi]]
        values += [row.double() for row in v[lo:hi]]
        if not keys:
            continue
        keys, values = torch.stack(keys), torch.stack(values)
        for row in range(hi - lo):
            last = (row if unshifted else context + row) + 1 + future_tokens
            last = min(last, keys.shape[0])
            first = max(0, last - window) if window else 0
            if ignore_context:
                first = max(first, context)
            for head in range(q.shape[1]):
                logits = keys[first:last, head // group] @ q[lo + row, head].double() * scale
                out[lo + row, head] = torch.softmax(logits, 0) @ values[first:last, head // group]
    return out


def test_manifest_rows_are_bound_to_protected_domain_table(task):
    checks, harness = task
    data = json.loads((TASK / 'workloads.json').read_text())
    rows = data['cases']
    assert [row['test_case_id'] for row in rows[7:]] == list(CASES)
    for row in rows[7:]:
        assert row['params'] == DOMAIN[row['test_case_id']]
        assert row['params']['contract_case'] == row['test_case_id']
        assert row['checks'] == ['correctness'] and row['status'] == 'PASS'
        assert row['dtype'] == 'float16,float32,int32'
    declared = {row['test_case_id']: row['params'] for row in rows if 'contract_case' in row['params']}
    assert list(checks.CONTRACT_CASES) == list(ORIGINAL_CONTROLS) + list(CASES)
    assert declared == checks.CONTRACT_CASES
    assert checks.DOMAIN_CONTROLS.DOMAIN_CASES == DOMAIN


def test_declared_domain_spans_lengths_heads_pages_scales_and_windows():
    padded = {c['head_dim'] for c in DOMAIN.values() if c['head_dim'] & (c['head_dim'] - 1)}
    assert padded == {24, 80, 96, 112}
    assert {c['head_dim'] for c in DOMAIN.values()} >= {16, 32, 128}
    pages = {c['block_size'] for c in DOMAIN.values()}
    assert {p for p in pages if p & (p - 1)} == {12, 40} and {8, 64, 128} <= pages
    contexts = [n for c in DOMAIN.values() for n in c['context_lengths']]
    queries = [n for c in DOMAIN.values() for n in c['query_lengths']]
    assert 0 in contexts and 0 in queries and 1 in queries and max(queries) > 128
    assert any(n % 32 and n % c['block_size'] for c in DOMAIN.values() for n in c['context_lengths'])
    assert {c['query_heads'] // c['kv_heads'] for c in DOMAIN.values()} >= {1, 2, 3, 4}
    windows = [c.get('sliding_window') for c in DOMAIN.values()]
    assert None in windows and {3, 40, 4096} <= set(windows)
    assert all(abs(c['sm_scale'] - c['head_dim'] ** -0.5) > 0.02 for c in DOMAIN.values())


@pytest.mark.parametrize('case', CASES)
def test_generated_inputs_match_declaration_and_use_nonidentity_pages(task, case):
    checks, harness = task
    c = DOMAIN[case]
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    for name, value in zip(NAMES, args):
        wanted = c['output_shapes']['o'] if name == 'o' else c['input_shapes'][name]
        assert list(value.shape) == wanted
        assert value.dtype == (torch.float16 if value.is_floating_point() else torch.int32)
        assert value.is_contiguous()
    assert args[7].tolist() == [0] + torch.tensor(c['query_lengths']).cumsum(0).tolist()
    assert args[8].tolist() == c['sequence_lengths']
    assert kwargs == {'max_input_len': max(c['query_lengths']), 'sm_scale': c['sm_scale'],
                      **({'sliding_window': c['sliding_window']} if 'sliding_window' in c else {})}
    mapped = [int(args[6][b, j]) for b, n in enumerate(c['context_lengths'])
              for j in range(-(-n // c['block_size']))]
    assert len(set(mapped)) == len(mapped) and all(page >= 2 and page % 2 == 0 for page in mapped)
    assert mapped != sorted(mapped)  # Not the identity order.
    assert all(torch.isfinite(value).all() for value in args[:6] if value.is_floating_point())
    again, _ = checks.control_inputs(harness, case, 'cpu')
    assert all(torch.equal(a, b) for a, b in zip(args, again))


@pytest.mark.parametrize('case', CASES)
def test_oracle_matches_independent_naive_cache_walk(task, case):
    checks, harness = task
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    # Pin the layout definition element by element at a few cache positions.
    k_cache, page, width = args[4], args[4].shape[3], args[4].shape[4]
    for token, dim in [(0, 0), (page - 1, width), (page + 3, args[0].shape[-1] - 1)]:
        dense = k_cache.permute(0, 1, 3, 2, 4).reshape(*k_cache.shape[:2], page, -1)
        assert dense[1, 0, token % page, dim] == k_cache[1, 0, dim // width, token % page, dim % width]
    plan = checks.CallPlan(harness, args, kwargs)
    expected = plan.expected[0]
    assert torch.isfinite(expected).all()
    naive = naive_attention(args, kwargs)
    torch.testing.assert_close(expected, naive.to(torch.float16), atol=2e-3, rtol=1e-3)
    plan.check((naive.to(torch.float16),))


WRONG = {
    'zeros': {},
    'ignore_context': {'ignore_context': True},
    'one_future_token': {'future_tokens': 1},
    'causal_without_context_offset': {'unshifted': True},
    'default_scale': {'default_scale': True},
    'window_ignored': {'no_window': True},
}


@pytest.mark.parametrize('variant', WRONG)
@pytest.mark.parametrize('case', CASES)
def test_wrong_attention_outputs_are_rejected(task, case, variant):
    checks, harness = task
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    plan = checks.CallPlan(harness, args, kwargs)
    if variant == 'zeros':
        wrong = torch.zeros_like(args[0])
    else:
        wrong = naive_attention(args, kwargs, **WRONG[variant]).to(torch.float16)
    window = DOMAIN[case].get('sliding_window')
    if variant == 'window_ignored' and (window is None or window >= max(DOMAIN[case]['sequence_lengths'])):
        # No window, or a declared width longer than every sequence: a no-op.
        plan.check((wrong,))
        return
    with pytest.raises(AssertionError):
        plan.check((wrong,))


def candidate_double(fault=None):
    def candidate(*args, **kwargs):
        output = args[3]
        output.copy_(naive_attention(args, kwargs).to(output.dtype))
        if fault == 'unwritten_last_row':
            output[-1] = float('nan')
        elif fault == 'nonfinite':
            output[0, 0, 0] = float('inf')
        elif fault is not None:
            target = args[NAMES.index(fault)]
            target.view(-1)[-1] += 1
    return candidate


@pytest.mark.parametrize('fault', [None, 'unwritten_last_row', 'nonfinite',
                                   *[name for name in NAMES if name != 'o']])
def test_public_wrapper_guard_rejects_bad_outputs_and_input_mutation(task, monkeypatch, fault):
    checks, harness = task
    # CPU numerical double only; the GPU validator checks the real Triton launch.
    monkeypatch.setattr(checks, 'checked_candidate_call', lambda module, fn, *a, **k: fn(*a, **k))
    candidate = candidate_double(fault)
    module = SimpleNamespace(context_attention_fwd=candidate)
    harness.load_module = lambda: module
    for case in CASES:
        args, kwargs = checks.control_inputs(harness, case, 'cpu')
        pristine = [value.clone() for value in args]
        with checks.checked_modules(harness):
            checked = harness.load_module().context_attention_fwd
            if fault is None:
                checked(*args, **kwargs)
                assert all(torch.equal(a, b) for a, b in list(zip(args, pristine))[4:])
                assert all(torch.equal(a, b) for a, b in list(zip(args, pristine))[:3])
                assert torch.isfinite(args[3]).all()
            else:
                with pytest.raises(AssertionError, match='read-only' if fault in NAMES else None):
                    checked(*args, **kwargs)
        assert module.context_attention_fwd is candidate


class TimedRun(SimpleNamespace):
    pass


def cpu_replay_benchmark(fn, *, timed_run, warmup, repetition):
    assert (warmup, repetition) == (10, 100)
    outputs = fn()

    def rerun():
        for target, value in zip(outputs, fn()):
            target.copy_(value)
        return outputs

    timed_run.outputs, timed_run.rerun = outputs, rerun
    return .125, {'benchmark_method': 'cuda_graph', 'benchmark_warmup': 10,
                  'benchmark_samples': 100}


@pytest.mark.parametrize('stale_replay', [False, True])
def test_installed_control_runs_wrapper_and_perturbed_replay(task, monkeypatch, stale_replay):
    checks, harness = task
    monkeypatch.setattr(checks, 'checked_candidate_call', lambda module, fn, *a, **k: fn(*a, **k))
    inputs = checks.control_inputs
    monkeypatch.setattr(checks, 'control_inputs', lambda h, case: inputs(h, case, 'cpu'))
    calls = []
    good = candidate_double()

    def candidate(*args, **kwargs):
        calls.append(1)
        if stale_replay and len(calls) > 2:
            return None  # Leave the poisoned replay output unwritten.
        return good(*args, **kwargs)

    harness.load_module = lambda: SimpleNamespace(context_attention_fwd=candidate)
    harness._TimedRun, harness._benchmark_cuda_graph_or_events = TimedRun, cpu_replay_benchmark
    checks.install_controls(harness)
    for case in CASES:
        calls.clear()
        if stale_replay:
            with pytest.raises(AssertionError):
                harness.run_contract_correctness(case)
            continue
        ok, evidence = harness.run_contract_correctness(case)
        assert ok and evidence['unscored_control'] and evidence['perturbed_input_replay_checked']
        assert evidence['timed_output_checked'] and len(calls) == 3


def test_adapter_dispatches_every_domain_row_and_fails_closed(task, monkeypatch):
    checks, _ = task
    adapter = load(TASK / '_arena_eval.py', '_paged_domain_adapter')
    real = adapter.load_harness()
    monkeypatch.chdir(ROOT)
    assert real.CONTRACT_CASES == checks.CONTRACT_CASES
    calls = []

    def contract(case):
        calls.append(case)
        return case != CASES[-1], 'CPU control sentinel'

    real.run_correctness = lambda **kwargs: (True, None)
    real.run_contract_correctness = contract
    monkeypatch.setattr(adapter, 'load_harness', lambda: real)
    result = adapter.evaluate('candidate', 'correctness')
    assert calls == list(ORIGINAL_CONTROLS) + list(CASES)
    status = {row['test_case_id']: row['status'] for row in result['cases']}
    assert result['status'] == 'FAIL' and status.pop(CASES[-1]) == 'FAIL'
    assert set(status.values()) == {'PASS'}
    data = json.loads((TASK / 'workloads.json').read_text())
    for mutate in ('manifest', 'table'):
        tampered = copy.deepcopy(data)
        if mutate == 'manifest':
            tampered['cases'][-1]['params']['head_dim'] += 8
            monkeypatch.setattr(adapter, 'load_manifest', lambda: tampered)
        else:
            monkeypatch.setattr(adapter, 'load_manifest', lambda: data)
            monkeypatch.setitem(real.CONTRACT_CASES, CASES[0], {**DOMAIN[CASES[0]], 'block_size': 9})
        result = adapter.evaluate('candidate', 'correctness')
        assert result['status'] == 'FAIL' and 'manifest' in result['reason'].lower()
    monkeypatch.setattr(adapter, 'load_manifest', lambda: data)
    monkeypatch.setitem(real.CONTRACT_CASES, CASES[0], DOMAIN[CASES[0]])

    def forbidden(*args, **kwargs):
        pytest.fail('Performance must not execute or score domain controls')

    real.run_contract_correctness = forbidden
    real.run_performance = lambda: [{'test_case_id': f'perf{i}', 'execution_time_ms': .125,
                                     'benchmark_method': 'cuda_graph'} for i in range(1, 6)]
    result = adapter.evaluate('candidate', 'performance')
    assert result['status'] == 'PASS'
    assert [row['test_case_id'] for row in result['cases']] == [f'perf{i}' for i in range(1, 6)]


def strip_domain_hooks(tree):
    """Remove exactly the domain loader, merge and dispatch statements."""
    def hook(node):
        if isinstance(node, ast.FunctionDef):
            return node.name == '_load_domain_controls'
        return 'DOMAIN_CONTROLS' in ast.unparse(node)
    removed = [node for node in tree.body if hook(node)]
    assert [type(node).__name__ for node in removed] == ['FunctionDef', 'Assign', 'If', 'Expr']
    tree.body = [node for node in tree.body if not hook(node)]
    inputs = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'control_inputs')
    dispatch = [node for node in inputs.body if hook(node)]
    assert len(dispatch) == 1 and isinstance(dispatch[0], ast.If)
    inputs.body.remove(dispatch[0])
    assert 'DOMAIN_CONTROLS' not in ast.unparse(tree)
    return tree


def test_scored_rows_kernel_runner_and_checks_are_unchanged_from_base():
    before = json.loads(base_bytes('workloads.json'))
    after = json.loads((TASK / 'workloads.json').read_text())
    assert {k: v for k, v in after.items() if k != 'cases'} == {k: v for k, v in before.items() if k != 'cases'}
    assert after['cases'][:len(before['cases'])] == before['cases']
    assert [r['test_case_id'] for r in after['cases'] if 'performance' in r['checks']] == [
        f'perf{i}' for i in range(1, 6)]
    for relative in ('scripts/task_runner.py', 'source/triton_paged_prefix_prefill.py',
                     '_arena_eval.py'):
        assert (TASK / relative).read_bytes() == base_bytes(relative), relative
    # Removing only the domain loader/merge and its dispatch restores the base AST,
    # so the scored oracle, guards, replay and original controls are unchanged.
    current = strip_domain_hooks(ast.parse((TASK / '_arena_checks.py').read_text()))
    original = ast.parse(base_bytes('_arena_checks.py').decode())
    assert ast.dump(current, include_attributes=False) == ast.dump(original, include_attributes=False)
    runner = ast.parse(base_bytes('scripts/task_runner.py').decode())
    values = {t.id: n.value for n in runner.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name)}
    assert ast.literal_eval(values['WARMUP_ITERATIONS']) == 10
    assert ast.literal_eval(values['BENCHMARK_ITERATIONS']) == 100


def test_domain_module_is_protected_from_optimization_edits(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    from src.task_spec import load_task_spec
    workspace = tmp_path / 'triton_paged_prefix_prefill'
    shutil.copytree(TASK, workspace, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_paged_prefix_prefill')
    assert all(edit.path != '_domain_controls.py' for edit in spec.candidate.editable)
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    assert '_domain_controls.py' in snapshot.digests
    target = workspace / '_domain_controls.py'
    target.write_text(target.read_text().replace("'sm_scale': 0.07", "'sm_scale': 0.0707"))
    with pytest.raises(RuntimeError):
        verify_workspace_harness(snapshot, discard_added=False)
