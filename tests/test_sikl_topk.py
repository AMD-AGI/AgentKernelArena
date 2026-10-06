"""CPU evidence for the paged top-k SIKL task's contract, guards and timing protocol, not GPU validation."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from src.task_protocol import CaseManifest, parse_command_result

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/Aiter-task/topk_transform_paged_paged_k512_page_size64'
MODULE_NAMES = ['task_contract', 'task_inputs', 'task_compare', 'task_initialize',
                'task_reference', 'task_baseline', 'task_measure', 'task_validation', 'evaluate', 'export_solution']
BOUNDARY = [0, 1, 63, 64, 65, 511, 512, 513, 1024, 2048, 4096, 8192, 65536, 131072, 262207, 262208]
INPUT_NAMES = ['scores', 'seq_lens', 'metadata', 'page_size', 'page_tables']


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


def _cases(measure):
    return {case['case_id']: case for case in measure.task_inputs.CASES}


def test_manifest_covers_every_batch_with_bundle_boundary_and_mixed_lengths(monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        workload = contract.load_workload()
        rows = contract.case_manifest(workload)
        assert len(rows) == 233 and workload['boundary_lengths'] == BOUNDARY
        batches = [2**i for i in range(13)]
        expected_ids = []
        for batch in batches:
            expected_ids += [f'batch_{batch}_bundle', *(f'batch_{batch}_len_{n}' for n in BOUNDARY)]
            if batch > 1:
                expected_ids.append(f'batch_{batch}_mixed')
        assert [row['test_case_id'] for row in rows] == expected_ids
        for row, case in zip(rows, workload['cases']):
            assert row['checks'] == ['correctness', 'performance']
            assert row['shape'] == [case['batch'], 262208]
            assert row['params']['plan_rows'] == case['batch'] + 1
            assert row['params']['pages'] == 4097 and row['params']['page_size'] == 64 and row['params']['k'] == 512
            assert row['params']['lengths'] == case['lengths']
        assert len({row['params']['uuid'] for row in rows}) == 233
        mixed = next(case for case in workload['cases'] if case['case_id'] == 'batch_4_mixed')
        assert contract.row_lengths(mixed, workload) == [513, 65536, 1, 512]
        full = next(case for case in workload['cases'] if case['case_id'] == 'batch_16_mixed')
        assert sorted(contract.row_lengths(full, workload)) == sorted(BOUNDARY)
        manifest = CaseManifest.from_result(parse_report({
            'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task', 'status': 'PASS',
            'cases': rows, 'metadata': {'candidate_state': 'unimplemented'}}))
        assert len(manifest.cases) == 233


@pytest.mark.parametrize('mutation', [
    'length_above_capacity', 'negative_length', 'even_cycle_stride', 'cycle_offset',
    'boundary_above_capacity', 'plan_rows', 'extra_field', 'page_size_scalar', 'output_param'])
def test_manifest_rejects_malformed_task_data(mutation, monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        workload = deepcopy(contract.load_workload())
        uniform = next(c for c in workload['cases'] if isinstance(c['lengths'], dict) and 'uniform' in c['lengths'])
        mixed = next(c for c in workload['cases'] if isinstance(c['lengths'], dict) and 'cycle' in c['lengths'])
        if mutation == 'length_above_capacity':
            uniform['lengths'] = {'uniform': 262209}
        elif mutation == 'negative_length':
            uniform['lengths'] = {'uniform': -1}
        elif mutation == 'even_cycle_stride':
            mixed['lengths'] = {'cycle': {'stride': 4, 'offset': 0}}
        elif mutation == 'cycle_offset':
            mixed['lengths'] = {'cycle': {'stride': 5, 'offset': 16}}
        elif mutation == 'boundary_above_capacity':
            workload['boundary_lengths'][-1] = 262209
        elif mutation == 'plan_rows':
            uniform['plan_rows'] += 1
        elif mutation == 'extra_field':
            uniform['tolerance'] = 1
        elif mutation == 'page_size_scalar':
            workload['scalars']['page_size'] = 32
        else:
            workload['outputs']['out_page_indices']['param'] = 'out'
        with pytest.raises(ValueError):
            contract.case_manifest(workload)


@pytest.mark.parametrize('source', [
    'import torch\ndef f(x, k):\n return torch.topk(x, k)',
    'import torch as t\ndef f(x):\n return t.argsort(x)',
    'from torch import sort',
    'from torch import kthvalue as kv',
    'import sglang.kernels.ops.attention.dsv4.topk',
    'from sgl_kernel import topk_transform',
    'import aiter',
    'import task_reference',
    'import importlib',
])
def test_import_guard_rejects_library_selection_and_production_imports(source, monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        with pytest.raises(RuntimeError):
            contract.assert_source_independent(source)


def test_import_guard_allows_flydsl_and_plumbing(monkeypatch):
    source = ('import torch\nimport flydsl.expr as fx\n'
              'def build(**axes):\n    def launch(*args):\n        args[-1].fill_(-1)\n    return launch\n')
    with modules(TASK, monkeypatch) as contract:
        contract.assert_source_independent(source)


def test_plan_validation_requires_exactly_the_rows_above_threshold(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        inputs = importlib.import_module('task_inputs')
        lengths = torch.tensor([10, 300, 0], dtype=torch.int32)
        conservative = torch.zeros((4, 2), dtype=torch.int32)
        conservative[0, 0] = 2**31 - 1
        assert inputs.plan_matches(lengths, conservative)
        routed = torch.zeros((4, 2), dtype=torch.int32)
        routed[0] = torch.tensor([100, 1])
        routed[1] = torch.tensor([1, 300])
        assert inputs.plan_matches(lengths, routed)
        missing = routed.clone()
        missing[0, 1] = 0
        assert not inputs.plan_matches(lengths, missing)
        stale = routed.clone()
        stale[1, 1] = 299
        assert not inputs.plan_matches(lengths, stale)
        wrong_row = routed.clone()
        wrong_row[1, 0] = 0
        assert not inputs.plan_matches(lengths, wrong_row)


def test_case_inputs_apply_declared_lengths_and_redraws_keep_them(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch) as contract:
        inputs_module = importlib.import_module('task_inputs')
        cases = {c['case_id']: c for c in inputs_module.CASES}
        bundle = inputs_module.build_case_inputs(cases['batch_4_bundle'], device='cpu')
        assert bundle['seq_lens'].tolist() == [0, 65552, 131104, 262208]
        mixed = inputs_module.build_case_inputs(cases['batch_4_mixed'], device='cpu')
        assert mixed['seq_lens'].tolist() == [513, 65536, 1, 512]
        assert torch.equal(mixed['metadata'], bundle['metadata'])
        assert torch.equal(mixed['scores'], bundle['scores']) and mixed['page_size'] == 64
        evidence = inputs_module.check_case_inputs(mixed, cases['batch_4_mixed'], thorough=True)
        assert evidence['plan_matches_lengths'] and evidence['valid_scores_tie_free']
        held = {k: mixed[k].clone() for k in ('seq_lens', 'metadata', 'scores', 'page_tables')}
        draws = inputs_module.call_varying_draws(mixed, [5, 6])
        assert set(draws[0]) == {'scores', 'page_tables'}
        assert all(torch.equal(mixed[k], v) for k, v in held.items())
        inputs_module.redraw_call_varying_inputs(mixed, seed=5)
        assert torch.equal(mixed['seq_lens'], held['seq_lens']) and torch.equal(mixed['metadata'], held['metadata'])
        assert not torch.equal(mixed['scores'], held['scores'])
        del contract


@pytest.mark.parametrize('fault', ['above_capacity', 'not_declared', 'plan', 'page_table', 'tie', 'nan'])
def test_case_input_validation_rejects_illegal_inputs(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        case = next(c for c in inputs_module.CASES if c['case_id'] == 'batch_2_len_1024')
        inputs = inputs_module.build_case_inputs(case, device='cpu')
        if fault == 'above_capacity':
            inputs['seq_lens'][0] = 262209
        elif fault == 'not_declared':
            inputs['seq_lens'][1] = 1023
        elif fault == 'plan':
            inputs['metadata'][0] = torch.tensor([512, 0])
        elif fault == 'page_table':
            inputs['page_tables'][0, 0] = -1
        elif fault == 'tie':
            inputs['scores'][0, 1] = inputs['scores'][0, 0]
        else:
            inputs['scores'][1, 3] = float('nan')
        with pytest.raises(RuntimeError):
            inputs_module.check_case_inputs(inputs, case, thorough=True)


def test_output_classification_and_destination_passing_calls(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        expected = torch.tensor([[4, 9, -1, -1], [7, 3, 5, 1]], dtype=torch.int32)
        assert measure.compare_output(expected.clone(), expected)['status'] == 'PASS'
        reordered = expected.clone()
        reordered[1] = reordered[1].flip(0)
        assert measure.compare_output(reordered, expected)['status'] == 'PASS'
        poisoned = expected.clone()
        poisoned[0, 2:] = measure.task_inputs.OUTPUT_POISON
        result = measure.compare_output(poisoned, expected)
        assert result['failure_kind'] == 'numerical_mismatch' and result['metrics']['poisoned_elements'] == 2
        for got in (expected.long(), expected[:, :3], None):
            assert measure.compare_output(got, expected)['failure_kind'] == 'output_contract'
        inputs = {'scores': torch.zeros((2, 8)), 'seq_lens': torch.zeros(2, dtype=torch.int32),
                  'metadata': torch.zeros((3, 2), dtype=torch.int32), 'page_size': 64,
                  'page_tables': torch.zeros((2, 1), dtype=torch.int32)}
        out = torch.empty((2, 4), dtype=torch.int32)
        seen = []
        assert measure.case_call(inputs, out, role='candidate', launch=lambda *a: seen.append(a))() is out
        assert [x is y for x, y in zip(seen[0], (*inputs.values(), out))] == [True] * 6
        with pytest.raises(RuntimeError, match='returns None'):
            measure.case_call(inputs, out, role='candidate', launch=lambda *a: out)()
        with pytest.raises(RuntimeError, match='no baseline fallback'):
            measure.case_call(inputs, out, role='candidate')
        with pytest.raises(ValueError, match='cannot invoke a candidate'):
            measure.case_call(inputs, out, role='baseline', launch=lambda *a: None)


def test_builder_receives_shape_axes(monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        seen = {}
        measure.build_launch(lambda **axes: seen.update(axes) or (lambda *a: None), _cases(measure)['batch_8_len_64'])
        assert seen == {'batch': 8, 'width': 262208, 'pages': 4097, 'k': 512, 'page_size': 64}


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
    """Writes the operator's result into the destination on every call."""
    def __init__(self, reference):
        self.reference, self.cost, self.calls = reference, None, 0

    def compute(self, args):
        self.cost = 1.0
        args[-1].copy_(self.reference.run(**dict(zip(INPUT_NAMES, args[:-1]))))

    def __call__(self, *args):
        self.calls += 1
        self.compute(args)


class _ValueMemo(_Honest):
    """Writes a stored, correct result for any scores and page table seen before."""
    def __init__(self, reference):
        super().__init__(reference)
        self.store = {}

    def __call__(self, *args):
        key = (args[0].numpy().tobytes(), args[4].numpy().tobytes())
        if key in self.store:
            self.cost = 0.1
            args[-1].copy_(self.store[key])
            return
        self.compute(args)
        self.store[key] = args[-1].clone()


class _SkipsPadding(_Honest):
    """Writes the selected slots but leaves the -1 padding to whatever was there."""
    def compute(self, args):
        self.cost = 1.0
        result = self.reference.run(**dict(zip(INPUT_NAMES, args[:-1])))
        args[-1].copy_(result.where(result >= 0, args[-1]))


class _SkipsEveryThirdCall(_Honest):
    def __call__(self, *args):
        self.calls += 1
        if self.calls % 3 == 0:
            self.cost = 0.01
            return
        self.compute(args)


def _cpu(measure, monkeypatch, torch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    build = measure.task_inputs.build_case_inputs
    monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda case, device=None: build(case, device='cpu'))


@pytest.mark.parametrize('kernel_type, case_id, status, failure_kind', [
    (_Honest, 'batch_1_len_8192', 'PASS', None),
    (_Honest, 'batch_2_len_63', 'PASS', None),
    (_ValueMemo, 'batch_1_len_8192', 'FAIL', 'timing_input_memoized'),
    (_SkipsPadding, 'batch_2_len_63', 'FAIL', 'numerical_mismatch'),
    (_SkipsEveryThirdCall, 'batch_1_len_8192', 'FAIL', 'numerical_mismatch'),
])
def test_protocol_rejects_known_timed_path_exploits_and_accepts_honest_timing(
        kernel_type, case_id, status, failure_kind, monkeypatch):
    """Known exploit behaviours kept as regression fixtures, end to end through
    ``time_case`` with the real bundle initializer, reference and comparator."""
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        monkeypatch.setattr(measure, 'choose_checked_samples', lambda repetition, count: list(range(count)))
        kernel = kernel_type(measure.task_reference)
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        result = measure.time_case(_cases(measure)[case_id], role='candidate', launch=kernel)
        assert result['status'] == status, result.get('reason')
        assert result.get('failure_kind') == failure_kind
        metadata = result['metadata']
        assert metadata['input_validation']['plan_matches_lengths']
        assert metadata['timed_output_correctness']['metadata']['checked_invocations'] == (
            measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS)


@pytest.mark.parametrize('mutated', ['scores', 'seq_lens', 'metadata', 'page_tables'])
def test_timed_guard_rejects_mutation_of_each_input(mutated, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)

        class Mutating(_Honest):
            def compute(self, args):
                super().compute(args)
                # The lowest bit keeps every input legal for the next call.
                args[INPUT_NAMES.index(mutated)].view(torch.uint8).flatten()[0] ^= 1

        kernel = Mutating(measure.task_reference)
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        with pytest.raises(RuntimeError, match=f'protected input tensor: {mutated}$'):
            measure.time_case(_cases(measure)['batch_2_len_63'], role='candidate', launch=kernel)


def test_batched_capture_is_rejected(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        helper = types.SimpleNamespace(
            TimedRun=lambda: types.SimpleNamespace(bound=True),
            benchmark_cuda_graph_or_events=lambda *a, **kw: (
                0.01, {'benchmark_method': 'cuda_graph', 'benchmark_effective_repeats': 8}))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
        result = measure.time_case(_cases(measure)['batch_1_len_8192'], role='candidate', launch=lambda *a: None)
        assert result['status'] == 'FAIL' and result['failure_kind'] == 'timing_protocol'


class _StaleLengths(_Honest):
    """Reads seq_lens once, on its first call, and reuses them afterwards."""
    def __init__(self, reference):
        super().__init__(reference)
        self.lengths = None

    def compute(self, args):
        if self.lengths is None:
            self.lengths = args[1].clone()
        args = list(args)
        args[1] = self.lengths
        super().compute(args)


# The batch-1 bundle lengths for seed 0 are [0], the same as len_0, so a launch
# frozen on its first call's lengths still passes those two cases.
@pytest.mark.parametrize('kernel_type, passing', [(_Honest, 17), (_StaleLengths, 2)])
def test_correctness_reuses_buffers_across_growing_lengths(kernel_type, passing, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        kernel, reuse, statuses, storage = kernel_type(measure.task_reference), {}, [], set()
        for case in measure.task_inputs.CASES:
            if case['batch'] != 1:
                continue
            result = measure.check_case(case, role='candidate', launch=kernel, reuse=reuse)
            statuses.append(result['status'])
            storage.add(next(iter(reuse.values()))[0]['scores'].data_ptr())
            assert len(result['metadata']['data_draw_seeds']) == 1 + measure.CORRECTNESS_DRAWS
        assert len(statuses) == 17 and statuses.count('PASS') == passing
        assert len(storage) == 1  # One set of buffers served every length.


def test_mixed_correctness_adds_fresh_lengths(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        drawn = []
        original = measure.fresh_lengths

        def recorded(*args):
            drawn.append(original(*args))
            return drawn[-1]
        monkeypatch.setattr(measure, 'fresh_lengths', recorded)
        result = measure.check_case(_cases(measure)['batch_2_mixed'], role='candidate',
                                    launch=_Honest(measure.task_reference), reuse={})
        assert result['status'] == 'PASS'
        assert result['metadata']['checked_invocations'] == 2 + measure.CORRECTNESS_DRAWS
        assert len(drawn) == 1 and all(0 <= n <= 262208 for n in drawn[0])
        assert original(64, 262208, BOUNDARY) != original(64, 262208, BOUNDARY)


def test_independent_controls_execute_real_callbacks_on_cpu(monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        records = validation.run_controls(measure, device='cpu')
        assert [r['name'] for r in records] == ['topk_small_known_answer', 'topk_production_known_answer',
                                                'comparator_positive_and_negative']
        assert all(r['status'] == 'PASS' and r['device'] == 'cpu' for r in records)
        assert set(records[2]['evidence']['rejected_controls']) == {
            'wrong_long_selection', 'short_row_reordered', 'padding_moved', 'missing_padding',
            'duplicated_slot', 'wrong_dtype', 'wrong_shape'}
        json.dumps(records, allow_nan=False)


@pytest.mark.parametrize('fault', ['unmapped', 'ignore_lengths', 'ascending', 'short_reversed'])
def test_known_answer_detects_specific_reference_faults(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        reference = measure.task_reference
        original = reference.topk_reference

        def faulty(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, out_raw_indices=None):
            if fault == 'unmapped':
                page_tables = None
            elif fault == 'ignore_lengths':
                seq_lens = torch.full_like(seq_lens, scores.shape[1])
            elif fault == 'ascending':
                scores = -scores
            original(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, out_raw_indices)
            if fault == 'short_reversed':
                for row, length in enumerate(seq_lens.tolist()):
                    if 1 < length <= out_page_indices.shape[1]:
                        out_page_indices[row, :length] = out_page_indices[row, :length].flip(0)
        monkeypatch.setattr(reference, 'topk_reference', faulty)
        monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'topk_small_known_answer' and records[-1]['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['always_accept', 'always_reject', 'sets_only'])
def test_controls_detect_broken_comparator(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')

        def broken(actual, expected):
            if fault == 'always_reject':
                raise AssertionError('reject everything')
            if fault == 'sets_only':
                assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
        monkeypatch.setattr(measure.task_compare, 'run', broken)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'comparator_positive_and_negative' and records[-1]['status'] == 'FAIL'


@pytest.mark.parametrize('exports, expected', [
    (('topk_transform_paged_v2', 'topk_transform_512_v2'), 'topk_transform_paged_v2'),
    (('topk_transform_512_v2',), 'topk_transform_512_v2'),
    ((), None),
])
def test_baseline_binds_the_installed_entry_point_name(exports, expected, monkeypatch):
    with modules(TASK, monkeypatch):
        baseline = importlib.import_module('task_baseline')
        fake = types.ModuleType(baseline.TRANSFORM_MODULE)
        calls = []
        for name in exports:
            setattr(fake, name, lambda name=name, **kw: calls.append((name, kw)))
        monkeypatch.setitem(sys.modules, baseline.TRANSFORM_MODULE, fake)
        if expected is None:
            with pytest.raises(ImportError):
                baseline.resolve_transform()
            return
        assert baseline.resolve_transform()[0] == expected
        assert baseline.run('s', 'l', 'm', 64, 't', 'o') is None
        assert calls == [(expected, {'scores': 's', 'seq_lens': 'l', 'page_tables': 't',
                                     'out_page_indices': 'o', 'page_size': 64, 'metadata': 'm'})]


@pytest.mark.parametrize('broken', [False, True])
def test_validate_task_runs_mandatory_controls_and_preserves_manifest(tmp_path, monkeypatch, broken):
    pytest.importorskip('torch')
    task = tmp_path / 'task'
    shutil.copytree(TASK, task)
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'task_validation')
        for source in contract.load_config()['baseline']['source_files']:
            (task / source).parent.mkdir(parents=True, exist_ok=True)
            (task / source).write_text('// fake materialization, for orchestration unit test only')
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


def test_exported_binding_is_destination_passing(tmp_path, monkeypatch):
    """Exercise the binding with host sentinels, not a claim of FlyDSL execution."""
    task = tmp_path / 'task'
    shutil.copytree(TASK, task)
    artifact = tmp_path / 'unpacked'
    artifact.mkdir()
    (artifact / 'candidate.py').write_text(
        'CALLS = []\ndef builder(**axes):\n    def launch(*args):\n        CALLS.append((axes, args))\n    return launch\n')
    with modules(task, monkeypatch) as contract:
        exporter = importlib.import_module('export_solution')
        workload = contract.load_workload()
        path = artifact / 'sikl_entry.py'
        path.write_text(exporter.tensor_entry({'file': 'candidate.py', 'symbol': 'builder'}, workload))
        spec = importlib.util.spec_from_file_location('topk_binding_unit_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = types.SimpleNamespace(shape=(8, 1024))
        tables = types.SimpleNamespace(shape=(8, 16))
        assert module.run(scores, 'lens', 'plan', 64, tables, 'out') is None
        candidate = sys.modules['_sikl_solution_candidate']
        assert candidate.CALLS == [({'batch': 8, 'width': 1024, 'pages': 16, 'k': 512, 'page_size': 64},
                                    (scores, 'lens', 'plan', 64, tables, 'out'))]
        assert json.loads((task / 'solution.json').read_text())['spec']['destination_passing_style'] is True
