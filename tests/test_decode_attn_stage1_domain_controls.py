"""CPU checks for the stage-1 decode attention input-domain controls.

The real protected wrapper runs with a CPU model of the unchanged kernel body in
place of the Triton launch. Native Triton and GPU behavior need a separate
task_validator run on compatible hardware.
"""
import ast
import copy
import functools
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_decode_attn_stage1'
RELATIVE = TASK.relative_to(ROOT).as_posix()
BASE = '099e44cf6deb74131d2cfc96a3a055949a2edbdc'
DISPATCH = ('    if case_index is not None and case_index >= 20000:\n'
            '        from _domain_controls import run_control\n'
            '        return run_control(case_index - 20000, load_module)\n')
FAULTS_ALL = ('zeros', 'identity_pages', 'tail_decoy', 'skip_lse', 'mutate_k',
              'mutate_pages', 'rebind_output', 'stale')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controls():
    return load(TASK / '_domain_controls.py', '_test_stage1_domain_controls')


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.chdir(ROOT)  # The runner changes directory at import.
    return load(TASK / 'scripts/task_runner.py', '_test_stage1_runner')


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.chdir(ROOT)
    return load(TASK / '_arena_eval.py', '_test_stage1_adapter')


def git_show(relative):
    return subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT)


class Launch:
    def __init__(self, call):
        self.call = call

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.call(grid, *args, **kwargs)


def masked_load(flat, index, mask, dtype=torch.float32):
    values = torch.zeros(index.shape, dtype=dtype)
    values[mask] = flat[index[mask]].to(dtype)
    return values


def kernel_model(fault=None):
    """Block-by-block CPU model of the unchanged _fwd_kernel_stage1 body.

    It consumes exactly the grid, strides and constexprs that the real protected
    wrapper passes, including BLOCK_N tiles, page lookups and online softmax.
    """
    state = {}

    def launch(grid, Q, K, V, sm_scale, Req, Lens, Out, s_req, s_qb, s_qh, s_kb, s_kh,
               s_vb, s_vh, s_ob, s_oh, s_os, *, kv_group_num, BLOCK_DMODEL, BLOCK_DV,
               BLOCK_N, NUM_KV_SPLITS, PAGE_SIZE, logit_cap, Lk, Lv, num_warps, num_stages):
        assert grid == (Q.shape[0], Q.shape[1], NUM_KV_SPLITS)
        assert (BLOCK_N, num_stages) == (64, 2) and num_warps == (4 if kv_group_num == 1 else 2)
        assert BLOCK_DMODEL >= Lk == Q.shape[-1] and BLOCK_DV >= Lv
        if fault == 'stale' and 'cached' in state:
            Out.copy_(state['cached'])
            return
        if fault == 'mutate_k':
            K.view(-1)[0] += 1
        if fault == 'mutate_pages':
            Req.view(-1)[0] = Req.view(-1)[1]
        if fault == 'rebind_output':
            Out.set_(Out.clone())
        q_all, k_all, v_all, table, out = (t.reshape(-1) for t in (Q, K, V, Req, Out))
        offs_d, offs_dv = torch.arange(BLOCK_DMODEL), torch.arange(BLOCK_DV)
        mask_d, mask_dv = offs_d < Lk, offs_dv < Lv
        for b, h, split in itertools.product(*(range(n) for n in grid)):
            kv_head = h // kv_group_num
            length = int(Lens[b])
            q = masked_load(q_all, b * s_qb + h * s_qh + offs_d, mask_d)
            per = -(-length // NUM_KV_SPLITS)
            start = per * split
            end = min(start + per, length)
            if fault == 'tail_decoy' and start < length == end:
                end += 1
            base = b * s_ob + h * s_oh + split * s_os
            if end <= start:
                if fault == 'inactive_write':
                    out[base + offs_dv[mask_dv]] = 0.0
                continue
            e_max, e_sum, acc = torch.tensor(-math.inf), torch.tensor(0.0), torch.zeros(BLOCK_DV)
            for block in range(start, end, BLOCK_N):
                offs_n = block + torch.arange(BLOCK_N)
                valid = offs_n < end
                if fault == 'identity_pages':
                    page = offs_n // PAGE_SIZE
                else:
                    page = masked_load(table, s_req * b + offs_n // PAGE_SIZE, valid, torch.long)
                location = page * PAGE_SIZE + offs_n % PAGE_SIZE
                k = masked_load(k_all, location[:, None] * s_kb + kv_head * s_kh + offs_d[None, :],
                                valid[:, None] & mask_d[None, :])
                qk = (q[None, :] * k).sum(1) * sm_scale
                if logit_cap > 0 and fault != 'no_cap':
                    qk = logit_cap * (2 * torch.sigmoid(2 * (qk / logit_cap)) - 1)
                qk = torch.where(valid, qk, torch.tensor(-math.inf))
                v = masked_load(v_all, location[:, None] * s_vb + kv_head * s_vh + offs_dv[None, :],
                                valid[:, None] & mask_dv[None, :])
                n_e_max = torch.maximum(qk.max(), e_max)
                re_scale = torch.exp(e_max - n_e_max)
                p = torch.exp(qk - n_e_max)
                acc = acc * re_scale + (p[:, None] * v).sum(0)
                e_sum = e_sum * re_scale + p.sum()
                e_max = n_e_max
            value, lse = acc / e_sum, e_max + torch.log(e_sum)
            if fault == 'zeros':
                value, lse = torch.zeros_like(value), torch.tensor(0.0)
            out[base + offs_dv[mask_dv]] = value[mask_dv]
            if fault != 'skip_lse':
                out[base + Lv] = lse
        if fault == 'stale':
            state['cached'] = Out.clone()

    tree = ast.parse((TASK / 'source/_fwd_kernel_stage1.py').read_text())
    wrapper = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'decode_att_m_fwd')
    namespace = {'torch': torch, '_fwd_kernel_stage1': Launch(launch),
                 'triton': SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), 'actual decode_att_m_fwd', 'exec'),
         namespace)
    return SimpleNamespace(decode_att_m_fwd=namespace['decode_att_m_fwd'])


def naive_reference(args, kwargs):
    """Scalar-indexed FP64 two-pass softmax, independent of the module oracle."""
    q, k, v, out, pages, lengths, splits, scale, page_size = args
    cap = kwargs['logit_cap']
    result = torch.full(out.shape, float('nan'), dtype=torch.float64)
    group = q.shape[1] // k.shape[1]
    for b in range(q.shape[0]):
        length = int(lengths[b])
        chunk = (length + splits - 1) // splits
        for split in range(splits):
            begin, end = split * chunk, min(length, (split + 1) * chunk)
            if begin >= end:
                continue
            locations = [int(pages[b, t // page_size]) * page_size + t % page_size
                         for t in range(begin, end)]
            for h in range(q.shape[1]):
                scores = []
                for location in locations:
                    score = float((q[b, h].double() * k[location, h // group].double()).sum()) * scale
                    scores.append(cap * math.tanh(score / cap) if cap > 0 else score)
                peak = max(scores)
                weights = [math.exp(score - peak) for score in scores]
                total = sum(weights)
                values = torch.stack([v[location, h // group].double() for location in locations])
                result[b, h, split, :-1] = (torch.tensor(weights, dtype=torch.float64)[:, None]
                                            * values).sum(0) / total
                result[b, h, split, -1] = peak + math.log(total)
    return result


def cpu_installed_harness(harness, controls, monkeypatch, model):
    """Actual runner dispatch and installed _arena_checks guard, CPU tensors only."""
    checks = load(TASK / '_arena_checks.py', '_test_stage1_checks')
    # Only the native Triton launch audit is bypassed; CallPlan checks remain.
    monkeypatch.setattr(checks, 'checked_candidate_call', lambda module, fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(controls, 'run_control', functools.partial(controls.run_control, device='cpu'))
    monkeypatch.setitem(sys.modules, '_domain_controls', controls)
    harness.load_module = lambda: model
    checks.install(harness)
    return checks


def test_manifest_rows_and_domain_table_are_bound_to_protected_module(adapter, controls):
    data = adapter.load_manifest()
    assert data['domain_controls'] == json.loads(json.dumps(controls.CASES))
    rows = [r for r in data['cases'] if r['test_case_id'].startswith('control-domain-')]
    assert rows == [{'test_case_id': 'control-domain-' + case['name'],
                     'dtype': case['dtype'] + ',float32,int32',
                     'params': {'case_index': 20000 + index, 'configuration': case},
                     'checks': ['correctness'], 'status': 'PASS'}
                    for index, case in enumerate(json.loads(json.dumps(controls.CASES)))]
    assert data['cases'][-len(rows):] == rows and len(rows) == 7


@pytest.mark.parametrize('tamper', ['row_seed', 'table_length', 'drop_row', 'extra_row', 'module'])
def test_manifest_or_module_drift_fails_closed(adapter, monkeypatch, tmp_path, tamper):
    data = json.loads((TASK / 'workloads.json').read_text())
    root = tmp_path / 'task'
    shutil.copytree(TASK, root)
    if tamper == 'row_seed':
        data['cases'][-1]['params']['configuration']['seeds'][0] += 1
    elif tamper == 'table_length':
        data['domain_controls'][0]['sequence_lengths'][0] += 1
    elif tamper == 'drop_row':
        del data['cases'][-1]
    elif tamper == 'extra_row':
        extra = copy.deepcopy(data['cases'][-1])
        extra['test_case_id'] = 'control-domain-extra'
        extra['params']['case_index'] = 20007
        data['cases'].append(extra)
    else:
        source = (root / '_domain_controls.py').read_text()
        assert source.count("'seeds': [2101, 2102]") == 1
        (root / '_domain_controls.py').write_text(source.replace("'seeds': [2101, 2102]",
                                                                 "'seeds': [2101, 2103]"))
    (root / 'workloads.json').write_text(json.dumps(data, indent=2) + '\n')
    monkeypatch.setattr(adapter, 'ROOT', root)
    monkeypatch.setattr(adapter, 'MANIFEST', root / 'workloads.json')
    with pytest.raises(ValueError, match='Domain'):
        adapter.load_manifest()


@pytest.mark.parametrize('index', range(7))
@pytest.mark.parametrize('variant', (0, 1))
def test_generator_matches_declared_shapes_routing_poison_and_decoys(controls, index, variant):
    case = controls.CASES[index]
    args, kwargs = controls.control_inputs(index, 'cpu', variant)
    q, k, v, out, pages, lengths, splits, scale, page_size = args
    shapes = dict(zip(('q', 'k_buffer', 'v_buffer', 'req_to_tokens', 'b_seqlen'),
                      (list(t.shape) for t in (q, k, v, pages, lengths))))
    assert shapes == case['input_shapes'] and list(out.shape) == case['output_shapes']['att_out']
    assert {q.dtype, k.dtype, v.dtype} == {getattr(torch, case['dtype'])}
    assert (out.dtype, pages.dtype, lengths.dtype) == (torch.float32, torch.int32, torch.int32)
    assert all(t.is_contiguous() for t in (q, k, v, out, pages, lengths))
    assert (splits, page_size, kwargs) == (case['num_splits'], case['page_size'],
                                           {'logit_cap': case['logit_cap']})
    assert scale == pytest.approx(1 / math.sqrt(case['k_head_dim']))
    declared = case['sequence_lengths']
    assert lengths.tolist() == (declared[-1:] + declared[:-1] if variant else declared)
    active = controls.active_splits(lengths, splits)
    if variant == 0:
        assert active.sum(1).tolist() == case['active_splits']
    mask = active[:, None, :, None].expand_as(out)
    assert torch.isnan(out[mask]).all() and out[~mask].eq(case['initial_partial_value']).all()
    # Valid nonidentity routing: distinct physical pages per logical page, with
    # padding entries on unused decoy pages and decoy V in every unused slot.
    assert pages.min() >= 0 and pages.max() < case['physical_pages']
    used, referenced, identity = [], torch.zeros(k.shape[0], dtype=torch.bool), True
    for b, length in enumerate(lengths.tolist()):
        needed = -(-length // page_size)
        row = pages[b, :needed].tolist()
        used += row
        identity &= row == list(range(len(used) - needed, len(used)))
        assert set(pages[b, needed:].tolist()).isdisjoint(used)
        positions = torch.arange(length)
        referenced[pages[b, positions // page_size].long() * page_size + positions % page_size] = True
    assert len(set(used)) == len(used) and not identity
    assert referenced.sum() == sum(declared)
    assert v[~referenced].eq(controls.DECOY_VALUE).all()
    assert not v[referenced].eq(controls.DECOY_VALUE).any()


def test_controls_span_the_declared_domain_and_readme_states_it(controls):
    cases = controls.CASES
    page_sizes = {c['page_size'] for c in cases}
    k_dims = {c['k_head_dim'] for c in cases}
    v_dims = {c['v_head_dim'] for c in cases}
    lengths = [n for c in cases for n in c['sequence_lengths']]
    assert 1 in page_sizes and page_sizes - {1, 2, 4, 8, 16, 32, 64, 128}
    assert any(c['page_size'] > 64 for c in cases)
    assert min(k_dims | v_dims) == 16 and max(k_dims | v_dims) == 256
    assert any(c['k_head_dim'] > c['v_head_dim'] for c in cases)
    assert any(c['k_head_dim'] < c['v_head_dim'] for c in cases)
    assert {d for d in k_dims | v_dims if d & (d - 1)}
    assert 1 in lengths and max(lengths) > 4096
    assert all(any(n % c['page_size'] and n % 64 for n in c['sequence_lengths'])
               for c in cases if c['page_size'] > 1)
    assert {1, 16} <= {c['num_splits'] for c in cases}
    assert any(sum(c['active_splits']) < c['num_splits'] * c['batch'] for c in cases)
    assert {c['dtype'] for c in cases} == {'float16', 'bfloat16'}
    assert {c['logit_cap'] > 0 for c in cases} == {False, True}
    assert {c['query_heads'] // c['kv_heads'] for c in cases} >= {1, 2, 3, 4, 8}
    readme = (TASK / 'README.md').read_text()
    assert 'Must handle arbitrary' not in readme
    assert '### Supported input domain' in readme and 'from 16 through 256' in readme
    for case in cases:
        assert f"`{case['name']}`" in readme


@pytest.mark.parametrize('index', range(7))
def test_fp64_oracle_matches_naive_and_cpu_kernel_model(controls, index):
    args, kwargs = controls.control_inputs(index, 'cpu', 0)
    expected, active = controls.reference(*args[:3], args[4], args[5], *args[6:], **kwargs)
    mask = active[:, None, :, None].expand(expected.shape)
    if index != 5:  # The scalar loop is too slow for 12k tokens; the kernel model covers it.
        naive = naive_reference(args, kwargs)
        assert torch.isnan(naive[~mask]).all()
        torch.testing.assert_close(expected[mask], naive[mask], atol=1e-9, rtol=1e-9)
    output = args[3].clone()
    kernel_model().decode_att_m_fwd(*args[:3], output, *args[4:], **kwargs)
    torch.testing.assert_close(output[mask].double(), expected[mask], atol=1e-4, rtol=1e-4)


def test_validate_task_reference_self_check_runs_and_discriminates(controls, harness, monkeypatch):
    controls.validate_reference(harness)
    original = controls.reference
    monkeypatch.setattr(controls, 'reference',
                        lambda *a, logit_cap=0.0: original(*a, logit_cap=0.0))
    with pytest.raises(AssertionError, match='uncapped'):
        controls.validate_reference(harness)



@pytest.mark.parametrize('broken', (False, True))
def test_validate_task_action_runs_domain_self_check(adapter, monkeypatch, broken):
    found = importlib.util.find_spec
    # CPU-only environment: report the Triton dependency present for this action only.
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *a: object() if name == 'triton' else found(name, *a))
    load_harness = adapter.load_harness

    def harness():
        h = load_harness()
        if broken:  # A small oracle disagreement must fail the task-level action.
            original = h.reference_stage1
            h.reference_stage1 = lambda *args: original(*args) + 0.05
        return h

    monkeypatch.setattr(adapter, 'load_harness', harness)
    result = adapter.evaluate('task', 'validate-task')
    if broken:
        assert result['status'] == 'FAIL' and result['failure_kind'] == 'execution_failure'
        assert 'not close' in result['reason']
    else:
        assert result['status'] == 'PASS', result.get('reason')
        assert result['metadata']['domain_reference_validated'] is True

@pytest.mark.parametrize('index', range(7))
def test_real_wrapper_passes_through_installed_runner_dispatch(controls, harness, monkeypatch, index):
    cpu_installed_harness(harness, controls, monkeypatch, kernel_model())
    ok, evidence = harness.run_correctness(case_index=20000 + index)
    assert ok, evidence
    assert evidence['independent_fp64_oracle_checked'] and evidence['changed_input_second_call_checked']
    case = controls.CASES[index]
    elements = case['query_heads'] * (case['v_head_dim'] + 1)
    assert evidence['active_elements_checked'] == 2 * sum(case['active_splits']) * elements
    assert evidence['active_elements_checked'] + evidence['inactive_elements_preserved'] == (
        2 * case['batch'] * case['num_splits'] * elements)


# Cap and inactive-slot faults apply only to controls that have those branches.
FAULT_CASES = ([(index, fault) for index in range(7) for fault in FAULTS_ALL]
               + [(2, 'no_cap'), (6, 'no_cap'), (0, 'inactive_write'), (6, 'inactive_write')])


@pytest.mark.parametrize('index,fault', FAULT_CASES)
def test_wrong_routing_outputs_state_and_inputs_are_rejected(controls, index, fault):
    case = controls.CASES[index]
    if fault == 'no_cap':
        assert case['logit_cap'] > 0
    if fault == 'inactive_write':
        assert sum(case['active_splits']) < case['batch'] * case['num_splits']
    ok, reason = controls.run_control(index, lambda: kernel_model(fault), device='cpu')
    assert not ok and reason.startswith(f"Domain control {case['name']}:")
    expected = {'mutate_k': 'read-only', 'mutate_pages': 'read-only',
                'rebind_output': 'metadata/storage', 'skip_lse': 'nonfinite',
                'inactive_write': 'inactive caller-owned'}.get(fault)
    if expected:
        assert expected in reason


@pytest.mark.parametrize('fault', ('zeros', 'identity_pages', 'tail_decoy', 'stale'))
def test_installed_guard_and_domain_oracle_reject_wrong_results_on_long_case(
        controls, harness, monkeypatch, fault):
    cpu_installed_harness(harness, controls, monkeypatch, kernel_model(fault))
    ok, reason = harness.run_correctness(case_index=20005)
    assert not ok and 'long_page16_splits16' in reason


def test_unknown_domain_index_is_rejected(controls):
    for invalid in (-1, 7, True, 1.0):
        with pytest.raises(ValueError, match='Unknown domain'):
            controls.run_control(invalid, lambda: None, device='cpu')


def test_adapter_dispatches_domain_rows_only_to_correctness(adapter, monkeypatch):
    data = json.loads((TASK / 'workloads.json').read_text())
    calls = []

    def correctness(*, case_index):
        calls.append(case_index)
        return (case_index != 20003, {'sentinel': case_index} if case_index != 20003 else 'rejected')

    h = SimpleNamespace(TEST_SHAPES=data['input_table'], CONTRACT_CASES=load(
        TASK / '_arena_checks.py', '_test_stage1_dispatch_checks').CONTRACT_CASES,
        run_correctness=correctness, run_contract_correctness=lambda case: (True, {}),
        run_performance=lambda: [{'test_case_id': f'perf{i}', 'execution_time_ms': .125,
                                  'benchmark_method': 'cuda_graph'} for i in range(1, 6)])
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    result = adapter.evaluate('candidate', 'correctness')
    assert calls == [0, 1, 2, 3, 4, 10000, 10001] + list(range(20000, 20007))
    rows = {row['test_case_id']: row for row in result['cases']}
    assert result['status'] == 'FAIL'
    assert [r for r in rows if rows[r]['status'] == 'FAIL'] == ['control-domain-page128_k192_v128']
    domain = rows['control-domain-page1_hd16_short_inactive']['metrics']
    assert domain == {'unscored_domain_control_passed': True, 'evidence': {'sentinel': 20000}}
    assert rows['perf1']['metrics'] == {'original_case_checks_passed': True}
    performance = adapter.evaluate('candidate', 'performance')
    assert performance['status'] == 'PASS'
    assert [r['test_case_id'] for r in performance['cases']] == [f'perf{i}' for i in range(1, 6)]


def test_scored_rows_kernel_runner_and_timing_unchanged_from_base(harness):
    base = json.loads(git_show(f'{RELATIVE}/workloads.json'))
    current = json.loads((TASK / 'workloads.json').read_text())
    added = [r for r in current['cases'] if r['test_case_id'].startswith('control-domain-')]
    assert added and all(r['checks'] == ['correctness'] for r in added)
    current['cases'] = [r for r in current['cases'] if r not in added]
    del current['domain_controls']
    assert current == base
    assert [r['test_case_id'] for r in current['cases'] if 'performance' in r['checks']] == [
        f'perf{i}' for i in range(1, 6)]
    runner = (TASK / 'scripts/task_runner.py').read_text()
    assert runner.count(DISPATCH) == 1
    assert runner.replace(DISPATCH, '', 1).encode() == git_show(f'{RELATIVE}/scripts/task_runner.py')
    for relative in ('source/_fwd_kernel_stage1.py', '_arena_checks.py', '_upstream_controls.py'):
        assert (TASK / relative).read_bytes() == git_show(f'{RELATIVE}/{relative}'), relative
    assert (harness.WARMUP_ITERATIONS, harness.BENCHMARK_ITERATIONS) == (10, 100)
    assert json.loads(json.dumps(harness.TEST_SHAPES)) == base['input_table']


def test_domain_module_is_protected_from_candidate_edits(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    from src.task_spec import load_task_spec
    workspace = tmp_path / 'triton_decode_attn_stage1'
    shutil.copytree(TASK, workspace)
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_decode_attn_stage1')
    assert all(entry.path != '_domain_controls.py' for entry in spec.candidate.editable)
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    module = workspace / '_domain_controls.py'
    original = module.read_bytes()
    module.write_bytes(original.replace(b'ATOL = 0.01', b'ATOL = 1.0'))
    with pytest.raises(RuntimeError):
        verify_workspace_harness(snapshot, discard_added=False)
    module.write_bytes(original)
    verify_workspace_harness(snapshot, discard_added=False)
