"""CPU checks for the triton_bmm correctness-only domain controls.

The actual protected ``bmm_triton`` wrapper runs against a CPU model of the
unchanged device kernel's pointer arithmetic. These checks do not execute
Triton and do not establish GPU correctness or timing.
"""
import ast
import difflib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_bmm'
TASK_ID = 'triton2triton/vllm/triton_bmm'
BASE = '099e44cf6deb74131d2cfc96a3a055949a2edbdc'
DISPATCH = ('    if case_index is not None and case_index >= 20000:\n'
            '        from _domain_controls import run_control\n'
            '        return run_control(case_index - 20000, load_module)\n')
TILES = {'float16': (128, 128, 32), 'bfloat16': (128, 128, 32), 'float32': (64, 64, 32)}
NONCONTIGUOUS = {'transposed', 'batch_inner', 'padded', 'inner_step2', 'batch_step2', 'broadcast_batch'}
CONTROL_COUNT = 11
FAULTS = ('zeros', 'drop_k_tail', 'drop_mn_tail', 'scored_shapes_only', 'write_storage', 'nan')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_text(relative):
    return subprocess.check_output(['git', 'show', f'{BASE}:{TASK.relative_to(ROOT).as_posix()}/{relative}'],
                                   cwd=ROOT, text=True)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.syspath_prepend(str(TASK))
    monkeypatch.chdir(TASK)
    controls = load(TASK / '_domain_controls.py', '_test_bmm_domain_controls')
    # The runner dispatch imports this exact module object.
    monkeypatch.setitem(sys.modules, '_domain_controls', controls)
    adapter = load(TASK / '_arena_eval.py', '_test_bmm_domain_adapter')
    harness = adapter.load_harness()
    monkeypatch.setattr(adapter, 'load_harness', lambda: harness)
    return SimpleNamespace(controls=controls, harness=harness, adapter=adapter)


def cpu_cuda(monkeypatch):
    for name in ('empty', 'full'):
        original = getattr(torch, name)

        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)


def scored_shapes():
    return {tuple(row) for row in json.loads(base_text('workloads.json'))['input_table']}


def bmm_module(*, fault=None, seen=None):
    """Execute the actual protected wrapper around a CPU model of bmm_kernel."""
    tree = ast.parse((TASK / 'source/triton_bmm.py').read_text())
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bmm_triton')
    scored = scored_shapes()

    class Kernel:
        def __getitem__(self, grid):
            def launch(a, b, c, batch, m, n, k, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, **meta):
                if seen is not None:
                    seen.append({'a': a, 'b': b, 'grid': grid, 'meta': meta,
                                 'strides': ((sab, sam, sak), (sbb, sbk, sbn), (scb, scm, scn))})
                if fault == 'assume_contiguous':
                    sab, sam, sak, sbb, sbk, sbn = m * k, k, 1, k * n, n, 1
                # Same base-pointer and stride arithmetic as the device kernel.
                x = torch.as_strided(a, (batch, m, k), (sab, sam, sak))
                y = torch.as_strided(b, (batch, k, n), (sbb, sbk, sbn))
                if fault == 'drop_k_tail':
                    kept = k // meta['BLOCK_SIZE_K'] * meta['BLOCK_SIZE_K']
                    x, y = x[..., :kept], y[:, :kept]
                acc = torch.bmm(x.float(), y.float())
                if fault == 'drop_mn_tail':
                    acc[:, m // meta['BLOCK_SIZE_M'] * meta['BLOCK_SIZE_M']:] = 0
                    acc[:, :, n // meta['BLOCK_SIZE_N'] * meta['BLOCK_SIZE_N']:] = 0
                if fault == 'zeros' or (fault == 'scored_shapes_only' and (batch, m, n, k) not in scored):
                    acc.zero_()
                if fault == 'nan':
                    acc[:, -1, -1] = float('nan')
                torch.as_strided(c, (batch, m, n), (scb, scm, scn)).copy_(acc.to(c.dtype))
                if fault == 'write_storage':
                    torch.as_strided(a, (1,), (1,), 0).add_(1)
            return launch

    namespace = {'torch': torch, 'triton': SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
                 'bmm_kernel': Kernel()}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual bmm wrapper>', 'exec'), namespace)
    return SimpleNamespace(bmm_triton=namespace['bmm_triton'])


def test_manifest_rows_match_control_table_and_are_unscored(env):
    data = env.adapter.load_manifest()
    cases = env.controls.CASES
    assert len(cases) == CONTROL_COUNT and data['domain_controls'] == cases
    assert data['cases'][5:] == [
        {'test_case_id': 'control-domain-' + case['name'], 'dtype': case['dtype'],
         'params': {'case_index': 20000 + index, 'configuration': case},
         'checks': ['correctness'], 'status': 'PASS'} for index, case in enumerate(cases)]
    assert [row['test_case_id'] for row in data['cases'] if 'performance' in row['checks']] == [
        f'perf{index}' for index in range(1, 6)]
    assert {case['dtype'] for case in cases} == set(TILES)
    assert {case['a_layout'] for case in cases} | {case['b_layout'] for case in cases} == (
        NONCONTIGUOUS | {'contiguous'}) == set(env.controls.LAYOUTS)
    for dtype in TILES:
        assert any(case['dtype'] == dtype and {case['a_layout'], case['b_layout']} & NONCONTIGUOUS
                   for case in cases)
    for case in cases:
        batch, m, n, k = case['shape_bmnk']
        tile_m, tile_n, tile_k = TILES[case['dtype']]
        assert m % tile_m and n % tile_n and k % tile_k, case['name']
        assert tuple(case['shape_bmnk']) not in scored_shapes()
    shapes = [case['shape_bmnk'] for case in cases]
    assert any(shape[0] == 1 for shape in shapes) and any(shape[1] == 1 and shape[3] == 1 for shape in shapes)
    assert all(any(shape[dim] % 2 for shape in shapes) for dim in range(4))


@pytest.mark.parametrize('index', range(CONTROL_COUNT))
def test_declared_layouts_strides_and_sentinel_storage(env, index):
    controls = env.controls
    case = controls.CASES[index]
    a, b, owners = controls.make_case(index, 'cpu')
    batch, m, n, k = case['shape_bmnk']
    assert a.shape == (batch, m, k) and b.shape == (batch, k, n)
    assert a.dtype == b.dtype == getattr(torch, case['dtype'])
    assert (a.stride(), b.stride()) == controls.expected_strides(case)
    for view, owner, layout in zip((a, b), owners, (case['a_layout'], case['b_layout'])):
        assert view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
        assert view.is_contiguous() is (layout == 'contiguous')
        if layout == 'padded':
            assert view.storage_offset() > 0 and view.stride(1) % 16
        if layout in ('padded', 'inner_step2', 'batch_step2'):
            assert int((owner == controls.SENTINEL).sum()) == owner.numel() - view.numel() > 0
        if layout == 'broadcast_batch':
            assert view.stride(0) == 0 and owner.shape[0] == 1
    # Seeded CPU generation is deterministic across calls and devices.
    again = controls.make_case(index, 'cpu')
    assert torch.equal(a, again[0]) and torch.equal(b, again[1])


def test_validation_oracle_accepts_original_formula_and_rejects_wrong_outputs(env, monkeypatch):
    env.controls.validate_reference(env.harness)
    # A reference that ignored the last K term must fail the validator itself.
    reference = env.controls.reference
    monkeypatch.setattr(env.controls, 'reference', lambda a, b: reference(a[..., :-1], b[:, :-1]))
    with pytest.raises(AssertionError):
        env.controls.validate_reference(env.harness)


@pytest.mark.parametrize('index', range(CONTROL_COUNT))
def test_fp64_reference_matches_independent_computation(env, index):
    controls = env.controls
    a, b, _ = controls.make_case(index, 'cpu')
    expected = controls.reference(a, b)
    assert expected.dtype == torch.float64 and expected.device.type == 'cpu'
    assert expected.shape == (a.shape[0], a.shape[1], b.shape[2])
    independent = torch.einsum('bmk,bkn->bmn', a.contiguous().double(), b.contiguous().double())
    torch.testing.assert_close(expected, independent, atol=1e-10, rtol=1e-10)
    # Exact-summation spot checks at the first and last tile-tail elements.
    for bi, mi, ni in {(0, 0, 0), (a.shape[0] - 1, a.shape[1] - 1, b.shape[2] - 1)}:
        exact = math.fsum(float(a[bi, mi, ki]) * float(b[bi, ki, ni]) for ki in range(a.shape[2]))
        assert abs(float(expected[bi, mi, ni]) - exact) <= 1e-9 * max(1.0, abs(exact))


@pytest.mark.parametrize('index', range(CONTROL_COUNT))
def test_actual_wrapper_model_passes_every_control_through_runner(env, monkeypatch, index):
    cpu_cuda(monkeypatch)
    case = env.controls.CASES[index]
    seen = []
    module = bmm_module(seen=seen)
    assert env.controls.run_control(index, lambda: module) == (True, None)
    launch = seen[-1]
    a, b, _ = env.controls.make_case(index, 'cpu')
    batch, m, n, k = case['shape_bmnk']
    tile_m, tile_n, tile_k = TILES[case['dtype']]
    # The protected wrapper forwards caller views and their strides unchanged.
    assert launch['strides'][:2] == (a.stride(), b.stride())
    assert launch['strides'][2] == (m * n, n, 1)
    assert launch['grid'] == (batch, math.ceil(m / tile_m) * math.ceil(n / tile_n))
    assert (launch['meta']['BLOCK_SIZE_M'], launch['meta']['BLOCK_SIZE_N'],
            launch['meta']['BLOCK_SIZE_K']) == (tile_m, tile_n, tile_k)
    # The actual runner dispatch also applies the protected _arena_checks gate.
    monkeypatch.setattr(env.harness, 'load_module', lambda: module)
    assert env.harness.run_correctness(case_index=20000 + index) == (True, None)


@pytest.mark.parametrize('fault', FAULTS)
@pytest.mark.parametrize('index', range(CONTROL_COUNT))
def test_controls_reject_wrong_device_outputs(env, monkeypatch, index, fault):
    cpu_cuda(monkeypatch)
    ok, reason = env.controls.run_control(index, lambda: bmm_module(fault=fault))
    assert not ok and 'AssertionError' in reason
    if fault == 'write_storage':
        assert 'read-only' in reason


@pytest.mark.parametrize('index', range(CONTROL_COUNT))
def test_controls_reject_kernel_that_ignores_caller_strides(env, monkeypatch, index):
    cpu_cuda(monkeypatch)
    case = env.controls.CASES[index]
    ok, reason = env.controls.run_control(index, lambda: bmm_module(fault='assume_contiguous'))
    strided = {case['a_layout'], case['b_layout']} & NONCONTIGUOUS
    assert ok is not bool(strided), reason


def test_scored_cases_alone_cannot_detect_shape_special_casing(env, monkeypatch):
    checks = load(TASK / '_arena_checks.py', '_test_bmm_original_checks')
    module = bmm_module(fault='scored_shapes_only')
    for index, (batch, m, n, k) in enumerate(env.harness.TEST_SHAPES):
        torch.manual_seed(42 + index)
        a = torch.randn(batch, m, k, dtype=torch.float16)
        b = torch.randn(batch, k, n, dtype=torch.float16)
        checks.check_output(module.bmm_triton(a, b), checks.reference(a, b))
    cpu_cuda(monkeypatch)
    assert not any(env.controls.run_control(index, lambda: module)[0] for index in range(CONTROL_COUNT))


@pytest.mark.parametrize('fault', ['dtype', 'strides', 'device', 'shape', 'not_tensor', 'nonfinite'])
@pytest.mark.parametrize('index', [3, 8, 10])
def test_controls_reject_output_metadata_defects(env, monkeypatch, index, fault):
    cpu_cuda(monkeypatch)
    real = bmm_module()

    def candidate(a, b):
        out = real.bmm_triton(a, b)
        if fault == 'dtype':
            return out.double()
        if fault == 'strides':
            # Same values in a padded-pitch view instead of the wrapper allocation.
            padded = torch.zeros(out.shape[0], out.shape[1], out.shape[2] + 1, dtype=out.dtype)
            padded[..., :-1] = out
            return padded[..., :-1]
        if fault == 'device':
            return torch.empty_like(out, device='meta')
        if fault == 'shape':
            return out[:, :, :-1]
        if fault == 'not_tensor':
            return out.tolist()
        out.view(-1)[0] = float('inf')
        return out
    ok, reason = env.controls.run_control(index, lambda: SimpleNamespace(bmm_triton=candidate))
    assert not ok and 'AssertionError' in reason


@pytest.mark.parametrize('index', [4, 8, 10])
def test_reference_precedes_candidate_and_inputs_are_read_only(env, monkeypatch, index):
    cpu_cuda(monkeypatch)
    controls = env.controls
    events = []
    real = bmm_module()
    reference = controls.reference

    def recorded_reference(a, b):
        events.append(('reference', a.data_ptr(), b.data_ptr()))
        return reference(a, b)

    def observed(a, b):
        events.append(('candidate', a.data_ptr(), b.data_ptr(), a.stride(), b.stride()))
        return real.bmm_triton(a, b)
    monkeypatch.setattr(controls, 'reference', recorded_reference)
    assert controls.run_control(index, lambda: SimpleNamespace(bmm_triton=observed)) == (True, None)
    (first, *ref_ptrs), (second, *call) = events
    assert (first, second) == ('reference', 'candidate')
    # The oracle used pristine clones, never the buffers handed to the candidate.
    assert set(ref_ptrs).isdisjoint(call[:2])
    expected = controls.make_case(index, 'cpu')
    assert tuple(call[2:]) == (expected[0].stride(), expected[1].stride())

    def mutate_and_return_pristine_result(a, b):
        expected_output = real.bmm_triton(a.clone(), b.clone())
        a.mul_(-1)
        return expected_output

    def mutate_storage_only(a, b):
        output = real.bmm_triton(a, b)
        torch.as_strided(b, (1,), (1,), 0).add_(1)
        return output
    for candidate in (mutate_and_return_pristine_result, mutate_storage_only):
        ok, reason = controls.run_control(index, lambda: SimpleNamespace(bmm_triton=candidate))
        assert not ok and 'read-only' in reason


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_real_adapter_dispatches_all_controls_for_both_roles(env, monkeypatch, role):
    cpu_cuda(monkeypatch)
    harness = env.harness
    original = harness.run_correctness
    seen = []

    def run(*, case_index):
        seen.append(case_index)
        return original(case_index=case_index) if case_index >= 20000 else (True, None)
    monkeypatch.setattr(harness, 'run_correctness', run)
    monkeypatch.setattr(harness, 'load_module', lambda: bmm_module())
    result = env.adapter.evaluate(role, 'correctness')
    assert result['status'] == 'PASS', result
    assert seen == [row['params']['case_index'] for row in result['cases']]
    assert seen == [0, 1, 2, 3, 4] + list(range(20000, 20000 + CONTROL_COUNT))
    monkeypatch.setattr(harness, 'load_module', lambda: bmm_module(fault='zeros'))
    result = env.adapter.evaluate(role, 'correctness')
    assert result['status'] == 'FAIL'
    assert [row['test_case_id'] for row in result['cases'] if row['status'] == 'FAIL'] == [
        'control-domain-' + case['name'] for case in env.controls.CASES]


def test_adapter_performance_and_validation_cases_keep_controls_unscored(env, monkeypatch):
    find_spec = importlib.util.find_spec
    # CPU fixture only: pretend Triton is installed so validate-task reaches the oracle check.
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *args: object() if name == 'triton' else find_spec(name, *args))
    monkeypatch.setattr(env.harness, 'run_performance', lambda: [
        {'test_case_id': f'perf{index}', 'execution_time_ms': 0.5, 'benchmark_method': 'cuda_graph'}
        for index in range(1, 6)])
    result = env.adapter.evaluate('candidate', 'performance')
    assert result['status'] == 'PASS' and [row['test_case_id'] for row in result['cases']] == [
        f'perf{index}' for index in range(1, 6)]
    result = env.adapter.evaluate('task', 'validate-task')
    assert result['status'] == 'PASS', result
    assert [row['checks'] for row in result['cases']] == (
        [['correctness', 'performance']] * 5 + [['correctness']] * CONTROL_COUNT)


@pytest.mark.parametrize('drift', ['table', 'row', 'scored_check', 'missing', 'extra'])
def test_adapter_rejects_control_descriptor_drift(env, monkeypatch, drift):
    data = deepcopy(env.adapter.load_manifest())
    if drift == 'table':
        data['domain_controls'][0]['shape_bmnk'][1] = 128
    if drift == 'row':
        data['cases'][-1]['params']['configuration']['b_layout'] = 'contiguous'
    if drift == 'scored_check':
        data['cases'][-1]['checks'] = ['correctness', 'performance']
    if drift == 'missing':
        data['cases'].pop()
    if drift == 'extra':
        data['cases'].append(dict(deepcopy(data['cases'][-1]), test_case_id='control-domain-extra'))
    monkeypatch.setattr(env.adapter, 'load_manifest', lambda: data)
    monkeypatch.setattr(env.harness, 'load_module', lambda: pytest.fail('Drift reached candidate'))
    result = env.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL' and 'protected domain' in result['reason'].lower()


def test_scored_rows_kernel_checks_and_timing_unchanged_from_base(env):
    data = deepcopy(env.adapter.load_manifest())
    base = base_text('workloads.json')
    assert len(data.pop('domain_controls')) == CONTROL_COUNT
    data['cases'] = data['cases'][:-CONTROL_COUNT]
    assert json.dumps(data, indent=2) + '\n' == base
    runner = (TASK / 'scripts/task_runner.py').read_text()
    assert runner.count(DISPATCH) == 1
    assert runner.replace(DISPATCH, '', 1) == base_text('scripts/task_runner.py')
    for relative in ('source/triton_bmm.py', '_arena_checks.py'):
        assert (TASK / relative).read_text() == base_text(relative)
    old_adapter = base_text('_arena_eval.py').splitlines(True)
    new_adapter = (TASK / '_arena_eval.py').read_text().splitlines(True)
    opcodes = difflib.SequenceMatcher(a=old_adapter, b=new_adapter, autojunk=False).get_opcodes()
    assert {tag for tag, *_ in opcodes} == {'equal', 'insert'}
    harness = env.harness
    assert [list(shape) for shape in harness.TEST_SHAPES] == json.loads(base)['input_table']
    assert (harness.WARMUP_ITERATIONS, harness.BENCHMARK_ITERATIONS) == (10, 100)


@pytest.mark.parametrize('change', ['kernel', 'helper', 'controls', 'manifest', 'wrapper', 'runner'])
def test_guard_protects_control_module_and_keeps_kernel_editable(tmp_path, change):
    work = tmp_path / 'task'
    shutil.copytree(TASK, work, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    spec = load_task_spec(work / 'config.yaml', task_id=TASK_ID)
    edit, = spec.candidate.editable
    assert edit.path == 'source/triton_bmm.py' and edit.scope == 'symbols'
    assert edit.symbols == ('bmm_kernel',) and edit.allow_new_helpers
    assert not edit.contains('_domain_controls.py')
    guard = snapshot_workspace_harness(work, task_spec=spec)
    assert '_domain_controls.py' in guard.digests
    source = work / edit.path
    tree = ast.parse(source.read_text())
    if change in ('kernel', 'wrapper'):
        name = 'bmm_kernel' if change == 'kernel' else 'bmm_triton'
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.body.append(ast.Pass())
    elif change == 'helper':
        tree.body.extend(ast.parse('@triton.jit\ndef tile_helper(x):\n    return x\n').body)
    elif change == 'controls':
        (work / '_domain_controls.py').write_text('CASES = []\n')
    elif change == 'manifest':
        manifest = json.loads((work / 'workloads.json').read_text())
        manifest['cases'] = manifest['cases'][:5]
        (work / 'workloads.json').write_text(json.dumps(manifest, indent=2) + '\n')
    elif change == 'runner':
        runner = work / 'scripts/task_runner.py'
        runner.write_text(runner.read_text().replace(DISPATCH, '', 1))
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    if change in ('kernel', 'helper'):
        verify_workspace_harness(guard, discard_added=False)
    else:
        with pytest.raises(RuntimeError):
            verify_workspace_harness(guard, discard_added=False)
