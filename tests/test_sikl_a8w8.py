"""CPU evidence for the blockwise-scaled FP8 GEMM SIKL tasks' contracts, guards and timing protocol, not GPU validation."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import importlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from src.task_protocol import CaseManifest, baseline_correctness_accepted, parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tasks/Aiter-task'
PREFIX = 'gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_'
TASKS = sorted(SUITE.glob(PREFIX + '*'))
RAW = SUITE / (PREFIX + 'asraw_n4096_k256')
LOGICAL = SUITE / (PREFIX + 'aslogical_n4096_k256')
DIAGNOSTIC = {PREFIX + 'asraw_n1024_k4096', PREFIX + 'aslogical_n1024_k4096'}
MODULE_NAMES = ['task_contract', 'task_inputs', 'task_compare', 'task_initialize',
                'task_reference', 'task_baseline', 'task_measure', 'task_validation', 'evaluate', 'export_solution']


@contextmanager
def modules(task, monkeypatch):
    # Bare imports are deliberately task-local in production, with a separate
    # subprocess per action. Isolate them equally in these in-process tests.
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(task / 'scripts'))
        for name in MODULE_NAMES:
            patch.delitem(sys.modules, name, raising=False)
        contract = importlib.import_module('task_contract')
        yield contract
        for name in MODULE_NAMES:
            sys.modules.pop(name, None)


def parse_report(report):
    return parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(report, allow_nan=False),
                                role=report['role'], action=report['action'],
                                returncode=0 if report['status'] == 'PASS' else 1)


def test_family_has_eleven_raw_and_four_logical_tasks():
    storages = [json.loads((t / 'workload.json').read_text())['a_scale_storage'] for t in TASKS]
    assert len(TASKS) == 15 and storages.count('raw') == 11 and storages.count('logical') == 4
    for task in TASKS:
        assert ('_asraw_' in task.name) == (json.loads((task / 'workload.json').read_text())['a_scale_storage'] == 'raw')


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name[len(PREFIX):])
def test_manifest_covers_every_workload_row_with_its_layout(task, monkeypatch):
    with modules(task, monkeypatch) as contract:
        workload = contract.load_workload()
        rows = contract.case_manifest(workload)
        axes = workload['axes']
        assert [row['params']['m'] for row in rows] == [2**i for i in range(13)]
        assert axes['sn'] == -(-axes['n'] // 128) and axes['sk'] == -(-axes['k'] // 128)
        assert list(workload['inputs']) == ['a', 'b', 'a_scale', 'b_scale']
        for row in rows:
            assert row['shape'] == [row['params']['m'], axes['n'], axes['k']]
            assert row['params']['a_scale_storage'] == workload['a_scale_storage']
            assert row['params']['b_layout'] == 'aiter_shuffle_16x16' and row['dtype'] == 'float8_e4m3fn'
        CaseManifest.from_result(parse_report({
            'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task', 'status': 'PASS',
            'cases': rows, 'metadata': {'candidate_state': 'unimplemented'}}))


@pytest.mark.parametrize('mutation', ['scale_extent', 'storage', 'block', 'unshuffled_n', 'case_field', 'input_order'])
def test_manifest_rejects_malformed_task_data(mutation, monkeypatch):
    with modules(RAW, monkeypatch) as contract:
        workload = deepcopy(contract.load_workload())
        if mutation == 'scale_extent':
            workload['axes']['sn'] += 1
        elif mutation == 'storage':
            workload['a_scale_storage'] = 'plain'
        elif mutation == 'block':
            workload['block_size'] = [64, 128]
        elif mutation == 'unshuffled_n':
            workload['axes']['n'], workload['axes']['sn'] = 4104, 33
        elif mutation == 'case_field':
            workload['cases'][0]['atol'] = 1
        else:
            workload['inputs'] = dict(reversed(list(workload['inputs'].items())))
        with pytest.raises(ValueError):
            contract.case_manifest(workload)


@pytest.mark.parametrize('source', [
    'import torch\ndef f(a, b, s, t):\n return torch._scaled_mm(a, b, s, t)',
    'def f(a, b):\n return a @ b',
    'import torch\ndef f(a, b):\n return torch.matmul(a, b)',
    'from torch import _scaled_mm',
    'from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle',
    'import task_reference',
])
def test_import_guard_rejects_library_matrix_products(source, monkeypatch):
    with modules(RAW, monkeypatch) as contract:
        with pytest.raises(RuntimeError):
            contract.assert_source_independent(source)


def test_import_guard_allows_flydsl_and_plumbing(monkeypatch):
    with modules(RAW, monkeypatch) as contract:
        contract.assert_source_independent(
            'import torch\nimport flydsl.expr as fx\n'
            'def build(m, n, k):\n    return lambda a, b, s, t: torch.empty((m, n), dtype=torch.bfloat16)\n')


@pytest.mark.parametrize('task', [RAW, LOGICAL], ids=['raw', 'logical'])
def test_independent_controls_execute_real_callbacks_on_cpu(task, monkeypatch):
    pytest.importorskip('torch')
    with modules(task, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        records = validation.run_controls(measure, device='cpu')
        assert [r['name'] for r in records] == ['blockwise_scaled_known_answer', 'comparator_positive_and_negative']
        assert all(r['status'] == 'PASS' and r['device'] == 'cpu' for r in records)
        assert records[0]['evidence']['a_scale_storage'] == ('raw' if task == RAW else 'logical')
        json.dumps(records, allow_nan=False)


@pytest.mark.parametrize('task', [RAW, LOGICAL], ids=['raw', 'logical'])
@pytest.mark.parametrize('fault', ['other_storage', 'no_unshuffle', 'unit_a_scale', 'unit_b_scale'])
def test_known_answer_detects_specific_reference_faults(task, fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(task, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        reference = measure.task_reference
        storage = measure.task_inputs.A_SCALE_STORAGE
        bound = dict(reference._callable.keywords)
        if fault == 'other_storage':
            bound['a_scale_storage'] = 'logical' if storage == 'raw' else 'raw'
            monkeypatch.setattr(reference, '_callable', partial(reference._blockwise_scaled_reference, **bound))
        elif fault == 'no_unshuffle':
            # Plain storage skips both the weight unshuffle and the raw scale decoding.
            bound['a_scale_storage'] = 'plain'
            monkeypatch.setattr(reference, '_callable', partial(reference._blockwise_scaled_reference, **bound))
        else:
            name = 'a_scale' if fault == 'unit_a_scale' else 'b_scale'
            original = reference._callable
            monkeypatch.setattr(reference, '_callable',
                                lambda **kw: original(**{**kw, name: torch.ones_like(kw[name])}))
        monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'blockwise_scaled_known_answer' and records[-1]['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['always_accept', 'always_reject', 'exact_only'])
def test_controls_detect_broken_comparator(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(RAW, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')

        def broken(actual, expected):
            if fault == 'always_reject':
                raise AssertionError('reject everything')
            if fault == 'exact_only':
                assert torch.equal(actual, expected)
        monkeypatch.setattr(measure.task_compare, 'run', broken)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'comparator_positive_and_negative' and records[-1]['status'] == 'FAIL'


def _simulated_graph_helper(kernel):
    """The canonical helper's contract on CPU: one replay per sample, prepared by
    ``prepare_fn`` before its start event, ``after_sample`` after its end event,
    ``rerun_ms`` through the same path."""
    def benchmark(fn, *, warmup, repetition, target_ms, prepare_fn, timed_run):
        del target_ms
        for _ in range(warmup + 3):
            prepare_fn()
            fn()

        def sample():
            prepare_fn()
            return fn(), kernel.cost
        values = []
        for _ in range(repetition):
            outputs, cost = sample()
            values.append(cost)
            timed_run.after_sample(outputs)

        def rerun_ms():
            timed_run.outputs, cost = sample()
            return cost
        timed_run.bound, timed_run.outputs, timed_run.rerun_ms = True, outputs, rerun_ms
        return sum(values) / len(values), {'benchmark_method': 'cuda_graph',
                                           'benchmark_effective_repeats': 1}
    return types.SimpleNamespace(
        TimedRun=lambda: types.SimpleNamespace(bound=False, outputs=None),
        benchmark_cuda_graph_or_events=benchmark)


class _Honest:
    def __init__(self, reference):
        self.reference, self.cost, self.calls, self.out = reference, None, 0, None

    def compute(self, args):
        self.cost = 1.0
        self.out = self.reference.run(**dict(zip(('a', 'b', 'a_scale', 'b_scale'), args)))
        return self.out

    def __call__(self, *args):
        self.calls += 1
        return self.compute(args)


class _ValueMemo(_Honest):
    """Returns a stored, correct result for any activation and scales seen before."""
    def __init__(self, reference):
        super().__init__(reference)
        self.store = {}

    def __call__(self, *args):
        import torch
        key = (args[0].view(torch.uint8).numpy().tobytes(), args[2].numpy().tobytes())
        if key in self.store:
            self.cost = 0.1
            return self.store[key].clone()
        self.store[key] = self.compute(args).clone()
        return self.store[key]


class _SkipsEveryThirdCall(_Honest):
    def __call__(self, *args):
        self.calls += 1
        if self.out is not None and self.calls % 3 == 0:
            self.cost = 0.01
            return self.out
        return self.compute(args)


def _cpu(measure, monkeypatch, torch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    build = measure.task_inputs.build_case_inputs
    monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda case: build(case, device='cpu'))


@pytest.mark.parametrize('kernel_type, status, failure_kind', [
    (_Honest, 'PASS', None),
    (_ValueMemo, 'FAIL', 'timing_input_memoized'),
    (_SkipsEveryThirdCall, 'FAIL', 'numerical_mismatch'),
])
def test_protocol_rejects_known_timed_path_exploits_and_accepts_honest_timing(
        kernel_type, status, failure_kind, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(RAW, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        monkeypatch.setattr(measure, 'choose_checked_samples', lambda repetition, count: list(range(count)))
        kernel = kernel_type(measure.task_reference)
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        result = measure.time_case(measure.task_inputs.CASES[1], role='candidate', launch=kernel)
        assert result['status'] == status, result.get('reason')
        assert result.get('failure_kind') == failure_kind
        assert result['metadata']['timed_output_correctness']['metadata']['checked_invocations'] == (
            measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS)


@pytest.mark.parametrize('mutated', ['a', 'b', 'a_scale', 'b_scale'])
def test_timed_guard_rejects_mutation_of_each_input(mutated, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(RAW, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        names = ['a', 'b', 'a_scale', 'b_scale']

        class Mutating(_Honest):
            def compute(self, args):
                out = super().compute(args)
                args[names.index(mutated)].view(torch.uint8).flatten()[0] ^= 1
                return out

        kernel = Mutating(measure.task_reference)
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        with pytest.raises(RuntimeError, match=f'protected input tensor: {mutated}$'):
            measure.time_case(measure.task_inputs.CASES[1], role='candidate', launch=kernel)


def test_draws_vary_activations_and_hold_weights(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(RAW, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        inputs = inputs_module.build_case_inputs(inputs_module.CASES[3], device='cpu')
        raw = lambda tensor: tensor.view(torch.uint8)
        before = {name: raw(value).clone() for name, value in inputs.items()}
        first, second = inputs_module.call_varying_draws(inputs, [5, 6])
        assert set(first) == {'a', 'a_scale'}
        assert all(torch.equal(raw(inputs[n]), v) for n, v in before.items())
        assert not torch.equal(raw(first['a']), raw(second['a']))
        assert not torch.equal(first['a_scale'], second['a_scale'])


def test_batched_capture_is_rejected(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(RAW, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        helper = types.SimpleNamespace(
            TimedRun=lambda: types.SimpleNamespace(bound=True),
            benchmark_cuda_graph_or_events=lambda *a, **kw: (
                0.01, {'benchmark_method': 'cuda_graph', 'benchmark_effective_repeats': 8}))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
        result = measure.time_case(measure.task_inputs.CASES[0], role='candidate', launch=lambda *a: None)
        assert result['status'] == 'FAIL' and result['failure_kind'] == 'timing_protocol'


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name[len(PREFIX):])
def test_baseline_policy_and_evidence_are_task_specific(task):
    spec = load_task_spec(task / 'config.yaml', task_id=f'Aiter-task/{task.name}')
    readme = (task / 'README.md').read_text()
    if task.name in DIAGNOSTIC:
        assert spec.baseline.correctness_policy == 'diagnostic'
        assert 'BpreShuffle_32x128E' in spec.baseline.diagnostic_reason and 'splitK=6' in spec.baseline.diagnostic_reason
        assert '## Production baseline numerical evidence' in readme
    else:
        assert spec.baseline.correctness_policy == 'required' and spec.baseline.diagnostic_reason is None
        assert '## Production baseline numerical evidence' not in readme


def test_diagnostic_exception_covers_only_completed_numerical_baseline_mismatch(monkeypatch):
    task = SUITE / (PREFIX + 'asraw_n1024_k4096')
    spec = load_task_spec(task / 'config.yaml', task_id=f'Aiter-task/{task.name}')
    with modules(task, monkeypatch) as contract:
        rows = contract.case_manifest(contract.load_workload())
        manifest = CaseManifest.from_result(parse_report({
            'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task', 'status': 'PASS',
            'cases': rows, 'metadata': {'candidate_state': 'unimplemented'}}))
        cases = [{k: v for k, v in row.items() if k != 'checks'} for row in rows]
        for row in cases:
            row['status'] = 'PASS'
        cases[7].update(status='FAIL', failure_kind='numerical_mismatch', reason='finite mismatch')
        report = {'protocol': 'arena-eval-v1', 'role': 'baseline', 'action': 'correctness', 'status': 'FAIL',
                  'cases': cases, 'reason': '1/13 cases failed', 'failure_kind': 'numerical_mismatch'}
        assert baseline_correctness_accepted(parse_report(report), baseline=spec.baseline,
                                             phase='task_validation', manifest=manifest)
        cases[7].update(failure_kind='execution_error')
        report['failure_kind'] = 'execution_error'
        assert not baseline_correctness_accepted(parse_report(report), baseline=spec.baseline,
                                                 phase='task_validation', manifest=manifest)


@pytest.mark.parametrize('broken', [False, True])
def test_validate_task_runs_mandatory_controls_and_preserves_manifest(tmp_path, monkeypatch, broken):
    pytest.importorskip('torch')
    task = tmp_path / 'task'
    shutil.copytree(LOGICAL, task)
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'task_validation')
        source = task / contract.load_config()['baseline']['source_files'][0]
        source.parent.mkdir(parents=True)
        source.write_text('# fake materialization, for orchestration unit test only')
        monkeypatch.setattr(runner, 'require_runtime', lambda w: None)
        monkeypatch.setattr(runner, 'runtime_evidence', lambda w: {'backend': 'injected CPU unit test'})
        original = validation.run_controls
        monkeypatch.setattr(validation, 'run_controls', lambda m, **kw: original(m, device='cpu', **kw))
        calls = []
        monkeypatch.setattr(runner, 'validate_case', lambda case, m: calls.append(case['case_id']) or {'status': 'PASS'})
        if broken:
            monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        result = parse_report(runner.run('task', 'validate-task'))
        assert [r['test_case_id'] for r in result.cases] == list(measure.task_inputs.CASE_IDS)
        assert result.metadata['candidate_state'] == 'unimplemented'
        if broken:
            assert not result.passed and result.metadata['validation_controls'][-1]['status'] == 'FAIL'
        else:
            assert result.passed and calls == list(measure.task_inputs.CASE_IDS)


def test_exported_binding_reads_shapes_and_passes_inputs_in_order(tmp_path, monkeypatch):
    """Exercise the binding with host sentinels, not a claim of FlyDSL execution."""
    copy = tmp_path / 'task'
    shutil.copytree(RAW, copy)
    artifact = tmp_path / 'unpacked'
    artifact.mkdir()
    (artifact / 'candidate.py').write_text('def builder(**axes):\n    return lambda *inputs: (axes, inputs)\n')
    with modules(copy, monkeypatch) as contract:
        exporter = importlib.import_module('export_solution')
        path = artifact / 'sikl_entry.py'
        path.write_text(exporter.tensor_entry({'file': 'candidate.py', 'symbol': 'builder'}, contract.load_workload()))
        spec = importlib.util.spec_from_file_location('a8w8_binding_unit_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = types.SimpleNamespace(shape=(8, 256))
        b = types.SimpleNamespace(shape=(4096, 256))
        axes, inputs = module.run(a, b, 'a_scale', 'b_scale')
        assert axes == {'m': 8, 'n': 4096, 'k': 256} and inputs == (a, b, 'a_scale', 'b_scale')
