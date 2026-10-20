"""CPU checks for the grouped decode stage-1 correctness-only domain controls.

The protected wrapper and kernel bodies are executed on CPU through a small
``tl`` emulation. This checks the control plumbing, oracle and domain claims
against the actual source. It is not GPU or Triton compiler validation.
"""
import ast
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_decode_attn_grouped_stage1'
BASE = '099e44cf6deb74131d2cfc96a3a055949a2edbdc'
ORIGINAL_CONTROLS = ['ragged_permuted_pages', 'ragged_permuted_capped', 'inactive_ragged_pages',
                     'kv_group_17_two_heads', 'kv_group_24_two_heads']


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)  # The task runner changes into its task directory.
    return module


@pytest.fixture
def task():
    checks = load(TASK / '_arena_checks.py', '_grouped_domain_checks')
    harness = load(TASK / 'scripts/task_runner.py', '_grouped_domain_runner')
    return checks, harness


DOMAIN = list(load(TASK / '_arena_domain_controls.py', '_grouped_domain_table').DOMAIN_CASES)


# --- CPU emulation of the actual protected wrapper and kernel source ---------

class Pointer:
    def __init__(self, flat, offset=0):
        self.flat, self.offset = flat, offset

    def __add__(self, offset):
        return Pointer(self.flat, self.offset + offset)


def _load(pointer, mask=None, other=0.0):
    offset = torch.as_tensor(pointer.offset).long()
    if mask is None:
        return pointer.flat[offset]
    offset, mask = torch.broadcast_tensors(offset, torch.as_tensor(mask))
    # Masked lanes are never dereferenced; unmasked out-of-range lanes raise.
    values = pointer.flat[torch.where(mask, offset, torch.zeros_like(offset))]
    return torch.where(mask, values, torch.tensor(other, dtype=values.dtype))


def _store(pointer, value, mask):
    offset, value, mask = torch.broadcast_tensors(
        torch.as_tensor(pointer.offset).long(), torch.as_tensor(value), torch.as_tensor(mask))
    pointer.flat[offset[mask]] = value[mask].to(pointer.flat.dtype)


def emulated_module(mutate_kernel=None):
    """Return the actual wrapper whose Triton launch runs the actual kernel body."""
    source = TASK / 'source/_fwd_grouped_kernel_stage1.py'
    nodes = {node.name: node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef)}
    for node in nodes.values():
        node.decorator_list = []
    if mutate_kernel is not None:
        mutate_kernel(nodes['_fwd_grouped_kernel_stage1'])
    pid = [None]
    tl = SimpleNamespace(
        constexpr=int, float32=torch.float32, program_id=lambda axis: pid[0][axis],
        cdiv=lambda a, b: (a + b - 1) // b, arange=lambda start, end: torch.arange(start, end),
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype), load=_load, store=_store,
        dot=lambda a, b: a.float() @ b.float(), exp=torch.exp, log=torch.log,
        sigmoid=torch.sigmoid, maximum=torch.maximum, minimum=torch.minimum,
        where=torch.where, max=lambda x, axis: x.amax(axis), sum=lambda x, axis: x.sum(axis))
    kernel_scope = {'tl': tl}
    body = [nodes['tanh'], nodes['_fwd_grouped_kernel_stage1']]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(source), 'exec'), kernel_scope)
    launches = []

    class Launch:
        def __getitem__(self, grid):
            def run(*args, num_warps, num_stages, **constexprs):
                launches.append({'grid': grid, 'num_warps': num_warps,
                                 'num_stages': num_stages, **constexprs})
                for value in args:
                    assert not isinstance(value, torch.Tensor) or value.is_contiguous()
                wrapped = [Pointer(v.view(-1)) if isinstance(v, torch.Tensor) else v for v in args]
                for program in itertools.product(*(range(size) for size in grid)):
                    pid[0] = program
                    kernel_scope['_fwd_grouped_kernel_stage1'](*wrapped, **constexprs)
            return run

    wrapper_scope = {'triton': SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                                               cdiv=lambda a, b: (a + b - 1) // b),
                     '_fwd_grouped_kernel_stage1': Launch()}
    exec(compile(ast.Module(body=[nodes['decode_grouped_att_m_fwd']], type_ignores=[]),
                 str(source), 'exec'), wrapper_scope)
    return SimpleNamespace(decode_grouped_att_m_fwd=wrapper_scope['decode_grouped_att_m_fwd'],
                           launches=launches)


def naive_reference(args, kwargs, mapping=lambda head, group, kv_heads: head // group):
    """Scalar-loop float64 attention, independent of the protected oracle."""
    q, k, v, out, pages, lengths, splits, scale, page_size = args
    cap = kwargs['logit_cap']
    heads, kv_heads, lv = q.shape[1], k.shape[1], v.shape[-1]
    group = heads // kv_heads
    result = out.clone()
    for b in range(q.shape[0]):
        length = int(lengths[b])
        width = (length + splits - 1) // splits
        for head in range(heads):
            kv_head = mapping(head, group, kv_heads)
            query = q[b, head].double()
            for split in range(splits):
                begin, end = split * width, min((split + 1) * width, length)
                if begin >= end:
                    continue
                scores, rows = [], []
                for position in range(begin, end):
                    token = int(pages[b, position // page_size]) * page_size + position % page_size
                    score = float(k[token, kv_head].double() @ query) * scale
                    if cap > 0:
                        score = cap * float(torch.tanh(torch.tensor(score / cap, dtype=torch.float64)))
                    scores.append(score)
                    rows.append(v[token, kv_head].double())
                top = max(scores)
                weights = [float(torch.exp(torch.tensor(s - top, dtype=torch.float64))) for s in scores]
                total = sum(weights)
                result[b, head, split, :lv] = (sum(w * r for w, r in zip(weights, rows)) / total).float()
                result[b, head, split, lv] = top + float(torch.log(torch.tensor(total, dtype=torch.float64)))
    return result


def cpu_graph_double(harness):
    """Unscored replay double: rerun reinvokes the wrapper into the same buffers."""
    class TimedRun(SimpleNamespace):
        pass

    def benchmark(measured, *, timed_run, warmup, repetition):
        assert (warmup, repetition) == (10, 100)
        timed_run.outputs = measured()
        timed_run.rerun = measured
        return 0.125, {'benchmark_method': 'cuda_graph', 'benchmark_warmup': warmup,
                       'benchmark_samples': repetition}

    harness._TimedRun = TimedRun
    harness._benchmark_cuda_graph_or_events = benchmark


def install_cpu(checks, harness, monkeypatch, module):
    inputs = checks.control_inputs
    monkeypatch.setattr(checks, 'control_inputs', lambda h, case: inputs(h, case, 'cpu'))
    # Only the native Triton launch audit is bypassed; everything else is real.
    monkeypatch.setattr(checks, 'checked_candidate_call', lambda m, fn, *a, **kw: fn(*a, **kw))
    harness.load_module = lambda: module
    cpu_graph_double(harness)
    checks.install_controls(harness)


def writing_candidate(compute):
    def candidate(q, k, v, out, pages, lengths, splits, scale, page_size, logit_cap=0.0):
        args = (q, k, v, out, pages, lengths, splits, scale, page_size)
        out.copy_(compute(args, {'logit_cap': logit_cap}))
    return SimpleNamespace(decode_grouped_att_m_fwd=candidate)


def table(checks):
    return checks.DOMAIN.DOMAIN_CASES


# --- Manifest and protected table ---------------------------------------------

def test_manifest_rows_are_the_protected_domain_table_appended_after_original_controls(task):
    checks, _ = task
    data = json.loads((TASK / 'workloads.json').read_text())
    ids = [row['test_case_id'] for row in data['cases']]
    assert ids == [f'perf{i}' for i in range(1, 6)] + ORIGINAL_CONTROLS + DOMAIN
    assert list(checks.CONTRACT_CASES) == ORIGINAL_CONTROLS + DOMAIN
    for row in data['cases'][10:]:
        assert row['checks'] == ['correctness'] and row['status'] == 'PASS'
        assert row['dtype'] == 'float16,float32,int32'
        assert row['params'] == json.loads(json.dumps(table(checks)[row['test_case_id']]))
        assert checks.CONTRACT_CASES[row['test_case_id']] is table(checks)[row['test_case_id']]


def test_adapter_dispatches_every_domain_row_and_rejects_tampered_parameters(task, monkeypatch):
    checks, _ = task
    adapter = load(TASK / '_arena_eval.py', '_grouped_domain_adapter')
    data = json.loads((TASK / 'workloads.json').read_text())
    calls = []

    def control(case):
        calls.append(case)
        return case not in DOMAIN, 'domain sentinel failure'

    h = SimpleNamespace(TEST_SHAPES=data['input_table'], CONTRACT_CASES=checks.CONTRACT_CASES,
                        run_correctness=lambda **kw: (True, None), run_contract_correctness=control)
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    result = adapter.evaluate('candidate', 'correctness')
    assert calls == ORIGINAL_CONTROLS + DOMAIN
    assert result['status'] == 'FAIL'
    assert [r['status'] for r in result['cases']] == ['PASS'] * 10 + ['FAIL'] * len(DOMAIN)
    for key, change in (('page_size', 1), ('kv_heads', 1), ('sequence_lengths', [9, 9])):
        tampered = copy.deepcopy(data)
        tampered['cases'][-1]['params'][key] = change
        monkeypatch.setattr(adapter, 'load_manifest', lambda tampered=tampered: tampered)
        result = adapter.evaluate('candidate', 'correctness')
        assert result['status'] == 'FAIL' and 'manifest' in result['reason'].lower()
    monkeypatch.setattr(adapter, 'load_manifest', lambda: data)
    h.run_contract_correctness = lambda case: pytest.fail('performance must not run controls')
    h.run_performance = lambda: [{'test_case_id': f'perf{i}', 'execution_time_ms': .5,
                                  'benchmark_method': 'cuda_graph'} for i in range(1, 6)]
    result = adapter.evaluate('candidate', 'performance')
    assert result['status'] == 'PASS'
    assert [r['test_case_id'] for r in result['cases']] == [f'perf{i}' for i in range(1, 6)]


def test_scored_rows_runner_and_manifest_header_are_unchanged_from_base():
    def at_base(relative):
        return subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT)
    prefix = TASK.relative_to(ROOT).as_posix()
    base = json.loads(at_base(f'{prefix}/workloads.json'))
    current = json.loads((TASK / 'workloads.json').read_text())
    scored = [row for row in current['cases'] if 'performance' in row['checks']]
    assert scored == [row for row in base['cases'] if 'performance' in row['checks']]
    assert scored == current['cases'][:5] and len(scored) == 5
    # Every base row (scored cases and earlier controls) is retained unchanged.
    assert current['cases'][:len(base['cases'])] == base['cases']
    assert {k: v for k, v in current.items() if k != 'cases'} == {
        k: v for k, v in base.items() if k != 'cases'}
    # Original inputs, comparisons, tolerances, 10 warmups and 100 samples.
    assert (TASK / 'scripts/task_runner.py').read_bytes() == at_base(f'{prefix}/scripts/task_runner.py')


# --- Domain coverage and generated inputs -------------------------------------

def test_domain_table_spans_the_wrapper_domain_beyond_existing_cases(task):
    checks, _ = task
    cases = table(checks).values()
    groups = {c['kv_group_num'] for c in cases}
    assert {1, 3, 5, 7, 12, 16, 33} <= groups
    assert {1, 3} <= {c['kv_heads'] for c in cases}
    pages = {c['page_size'] for c in cases}
    assert {1, 2, 3, 5, 12, 64} <= pages and any(p > 32 for p in pages)
    assert {(16, 16), (40, 72), (80, 80), (96, 64), (128, 128), (256, 256)} <= {
        (c['head_dim'], c['value_head_dim']) for c in cases}
    assert {1, 2, 4, 5, 6, 7} <= {c['num_splits'] for c in cases}
    assert {0.0, 1.5, 2.5} == {c['logit_cap'] for c in cases}
    lengths = [n for c in cases for n in c['sequence_lengths']]
    assert 1 in lengths and any(n > 64 for n in lengths)
    # Every case has a partial final page (when pages exceed one token) and a
    # partial final BLOCK_N=32 tile.
    assert all(any(n % 32 and (c['page_size'] == 1 or n % c['page_size'])
                   for n in c['sequence_lengths']) for c in cases)
    inactive = 0
    for c in cases:
        width = [(n + c['num_splits'] - 1) // c['num_splits'] for n in c['sequence_lengths']]
        inactive += sum(s * w >= n for n, w in zip(c['sequence_lengths'], width)
                        for s in range(c['num_splits']))
    assert inactive


@pytest.mark.parametrize('case', DOMAIN)
def test_inputs_match_declared_shapes_and_routing_is_valid_nonidentity(task, case):
    checks, harness = task
    c = table(checks)[case]
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    again, _ = checks.control_inputs(harness, case, 'cpu')
    assert all(not isinstance(a, torch.Tensor) or torch.equal(a, b) for a, b in zip(args, again))
    q, k, v, out, pages, lengths, splits, scale, page_size = args
    names = dict(zip(('q', 'k_buffer', 'v_buffer', 'req_to_tokens', 'b_seqlen'), (q, k, v, pages, lengths)))
    assert {name: list(value.shape) for name, value in names.items()} == c['input_shapes']
    assert list(out.shape) == c['output_shapes']['att_out']
    assert (q.dtype, k.dtype, v.dtype, out.dtype, pages.dtype, lengths.dtype) == (
        torch.float16, torch.float16, torch.float16, torch.float32, torch.int32, torch.int32)
    assert (splits, page_size, kwargs) == (c['num_splits'], c['page_size'], {'logit_cap': c['logit_cap']})
    assert scale == pytest.approx(c['head_dim'] ** -0.5)
    assert lengths.tolist() == c['sequence_lengths'] and out.eq(23.5).all()
    flat = pages.reshape(-1)
    assert len(set(flat.tolist())) == flat.numel() == c['physical_pages'] - 1
    assert 0 <= int(flat.min()) and int(flat.max()) < c['physical_pages']
    identity = torch.arange(flat.numel(), dtype=torch.int32).reshape_as(pages)
    assert not torch.equal(pages, identity)
    # The replay rolls each table row by one column; all entries stay in range.
    assert max(c['sequence_lengths']) <= (pages.shape[1] - 1) * page_size


@pytest.mark.parametrize('case', DOMAIN)
def test_actual_wrapper_launch_parameters_and_head_mapping(task, case):
    """Code evidence for the README domain: fixed tiles, no DPE split, per-group blocks."""
    checks, harness = task
    c = table(checks)[case]
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    module = emulated_module()
    module.decode_grouped_att_m_fwd(*args, **kwargs)
    launch, = module.launches
    pow2 = lambda n: 1 << (n - 1).bit_length()
    assert launch['grid'] == (c['batch'], c['kv_heads'] * -(-c['kv_group_num'] // 16), c['num_splits'])
    assert {key: launch[key] for key in ('BLOCK_DMODEL', 'BLOCK_DPE', 'BLOCK_DV', 'BLOCK_N', 'BLOCK_H',
            'NUM_KV_SPLITS', 'PAGE_SIZE', 'kv_group_num', 'q_head_num', 'Lk', 'Lv',
            'num_warps', 'num_stages')} == {
        'BLOCK_DMODEL': pow2(c['head_dim']), 'BLOCK_DPE': 0, 'BLOCK_DV': pow2(c['value_head_dim']),
        'BLOCK_N': 32, 'BLOCK_H': 16, 'NUM_KV_SPLITS': c['num_splits'], 'PAGE_SIZE': c['page_size'],
        'kv_group_num': c['kv_group_num'], 'q_head_num': c['query_heads'], 'Lk': c['head_dim'],
        'Lv': c['value_head_dim'], 'num_warps': 4, 'num_stages': 2}


# --- Oracle independence and the actual installed control path -----------------

@pytest.mark.parametrize('case', DOMAIN)
def test_domain_oracle_matches_independent_naive_and_original_references(task, case):
    checks, harness = task
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    values = dict(enumerate(args)) | kwargs
    expected, = checks.DOMAIN.expected_outputs(harness, values)
    torch.testing.assert_close(expected, naive_reference(args, kwargs), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(expected, checks.expected_outputs(harness, values)[0], atol=1e-4, rtol=1e-4)
    active = checks.output_write_mask(values)
    assert expected[~active].eq(23.5).all() and torch.isfinite(expected).all()


@pytest.mark.parametrize('case', DOMAIN)
def test_emulated_actual_kernel_passes_installed_control_with_replay(task, case, monkeypatch):
    checks, harness = task
    module = emulated_module()
    install_cpu(checks, harness, monkeypatch, module)
    ok, evidence = harness.run_contract_correctness(case)
    assert ok and evidence['unscored_control'] and evidence['independent_float64_domain_oracle']
    assert evidence['timed_output_checked'] and evidence['perturbed_input_replay_checked']
    # Checked call, measured call and perturbed replay each ran the kernel body.
    assert len(module.launches) == 3


def _wrong_mapping_matters(c):
    return c['kv_heads'] > 1 and c['kv_group_num'] > 1


WRONG = {
    'zeros': lambda checks: lambda args, kw: torch.where(
        checks.output_write_mask(dict(enumerate(args))), torch.zeros_like(args[3]), args[3]),
    'head_modulo_kv_heads': lambda checks: lambda args, kw: naive_reference(
        args, kw, mapping=lambda head, group, kv_heads: head % kv_heads),
    'uncapped': lambda checks: lambda args, kw: naive_reference(args, {'logit_cap': 0.0}),
    'inactive_slot_write': lambda checks: lambda args, kw: torch.where(
        checks.output_write_mask(dict(enumerate(args))), naive_reference(args, kw), torch.zeros_like(args[3])),
}


@pytest.mark.parametrize('wrong', sorted(WRONG))
@pytest.mark.parametrize('case', DOMAIN)
def test_installed_control_rejects_wrong_outputs(task, case, wrong, monkeypatch):
    checks, harness = task
    c = table(checks)[case]
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    candidate = WRONG[wrong](checks)
    if wrong == 'head_modulo_kv_heads' and not _wrong_mapping_matters(c):
        # h % kv_heads == h // group when the group or the KV head count is one.
        torch.testing.assert_close(candidate(args, kwargs), naive_reference(args, kwargs))
        return
    if wrong == 'uncapped' and c['logit_cap'] == 0:
        return
    if wrong == 'inactive_slot_write' and checks.output_write_mask(dict(enumerate(args))).all():
        return
    install_cpu(checks, harness, monkeypatch, writing_candidate(candidate))
    with pytest.raises(AssertionError):
        harness.run_contract_correctness(case)


@pytest.mark.parametrize('case', DOMAIN)
@pytest.mark.parametrize('index', (0, 1, 2, 4, 5))
def test_installed_control_rejects_readonly_input_mutation(task, case, index, monkeypatch):
    checks, harness = task

    def mutate(args, kwargs):
        target = args[index]
        if target.is_floating_point():
            target.add_(1)
        else:
            target.copy_(target.flip(0))
        return naive_reference(args, kwargs)

    install_cpu(checks, harness, monkeypatch, writing_candidate(mutate))
    with pytest.raises(AssertionError, match='read-only'):
        harness.run_contract_correctness(case)


def test_correct_candidate_leaves_inputs_byte_identical_and_kernel_mutation_is_caught(task, monkeypatch):
    checks, harness = task
    case = 'domain_group33_page12_k40_v72'
    args, kwargs = checks.control_inputs(harness, case, 'cpu')
    plan = checks.CallPlan(harness, args, kwargs, checks.DOMAIN.expected_outputs)
    plan.poison((args[3],))
    emulated_module().decode_grouped_att_m_fwd(*args, **kwargs)
    plan.unchanged()
    plan.check((args[3],))

    class NeighbourKVHead(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != 'cur_kv_head':
                return node
            return ast.parse('((cur_kv_head + 1) % (q_head_num // kv_group_num))').body[0].value

    def wrong_kv_head(kernel):
        # K/V reads use the next KV head instead of h // kv_group_num.
        changed = 0
        for node in ast.walk(kernel):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in ('offs_buf_k', 'offs_buf_v'):
                node.value = NeighbourKVHead().visit(node.value)
                changed += 1
        assert changed == 2

    install_cpu(checks, harness, monkeypatch, emulated_module(wrong_kv_head))
    with pytest.raises(AssertionError, match='Mismatched elements'):
        harness.run_contract_correctness(case)


def test_indivisible_query_heads_leave_heads_unwritten_as_readme_states():
    module = emulated_module()
    q = torch.randn(1, 10, 16, dtype=torch.float16)
    k = torch.randn(8, 4, 16, dtype=torch.float16)
    out = torch.full((1, 10, 1, 17), float('nan'))
    module.decode_grouped_att_m_fwd(q, k, k.clone(), out, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
                                    torch.tensor([8], dtype=torch.int32), 1, 0.25, 2)
    assert torch.isfinite(out[0, :8]).all() and torch.isnan(out[0, 8:]).all()
