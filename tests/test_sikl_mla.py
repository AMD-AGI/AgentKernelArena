"""CPU evidence for the sparse flash MLA SIKL tasks' contracts, guards and timing protocol, not GPU validation."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import types

import pytest

from src.task_protocol import CaseManifest, parse_command_result

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tasks/Aiter-task'
MAIN_ONLY = SUITE / 'flash_mla_with_kvcache_dsv4_fp8_10011_q1_h64_d512_p256_k128'
C128 = SUITE / 'flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256'
C4 = SUITE / 'flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep64_ek512'
TASKS = [MAIN_ONLY, C128, C4]
MODULE_NAMES = ['task_contract', 'task_inputs', 'task_compare', 'task_initialize',
                'task_reference', 'task_baseline', 'task_measure', 'task_validation', 'evaluate', 'export_solution']
SPARSE = [0, 1, 63, 64, 65, 127, 128]
EXTRA = [0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1024, 2048, 4096, 8192, 8255, 8256]


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


def _cases(module):
    return {case['case_id']: case for case in module.CASES}


@pytest.mark.parametrize('task, count, extra_grid, trajectory', [
    (MAIN_ONLY, 142, None, False),
    (C128, 415, EXTRA, True),
    (C4, 298, [e for e in EXTRA if e <= 512], False),
])
def test_manifest_covers_boundary_lengths_and_patterns_per_batch(task, count, extra_grid, trajectory, monkeypatch):
    with modules(task, monkeypatch) as contract:
        workload = contract.load_workload()
        rows = contract.case_manifest(workload)
        assert len(rows) == count
        assert workload['length_grid']['sparse'] == SPARSE
        assert workload['length_grid'].get('extra') == extra_grid
        for batch in (2**i for i in range(13)):
            ids = [c['case_id'] for c in workload['cases'] if c['batch'] == batch]
            assert ids[0] == f'batch_{batch}_bundle'
            assert (f'batch_{batch}_mixed' in ids) == (batch > 1)
            for special in ('holes', 'tail') + (('short_history',) if extra_grid else ()):
                assert f'batch_{batch}_{special}' in ids
            uniform = [c['lengths']['uniform'] for c in workload['cases']
                       if c['batch'] == batch and isinstance(c['lengths'], dict)
                       and set(c['lengths']) == {'uniform'}]
            pairs = {tuple(u.values()) for u in uniform}
            if extra_grid:
                assert {(s, max(extra_grid)) for s in SPARSE} | {(128, e) for e in extra_grid} | {(0, 0)} <= pairs
                assert ({(128, 64), (128, 256)} <= pairs) == trajectory
                assert [u['extra'] for u in uniform] == sorted(u['extra'] for u in uniform)
            else:
                assert pairs == {(s,) for s in SPARSE}
        for row in rows:
            assert row['checks'] == ['correctness', 'performance'] and row['dtype'] == 'bfloat16'
            assert row['shape'][1:] == [1, 64, 512] and row['params']['sm_scale'] == pytest.approx(1 / math.sqrt(512))
        assert len({row['params']['uuid'] for row in rows}) == count
        CaseManifest.from_result(parse_report({
            'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task', 'status': 'PASS',
            'cases': rows, 'metadata': {'candidate_state': 'unimplemented'}}))


def test_mixed_rows_cover_the_whole_length_grid(monkeypatch):
    with modules(C128, monkeypatch) as contract:
        workload = contract.load_workload()
        mixed = next(c for c in workload['cases'] if c['case_id'] == 'batch_4096_mixed')
        lengths = contract.row_lengths(mixed, workload)
        assert set(zip(lengths['sparse'], lengths['extra'])) == set(contract.grid_combinations(workload))
        small = next(c for c in workload['cases'] if c['case_id'] == 'batch_2_mixed')
        combos = contract.grid_combinations(workload)
        expected = [combos[(5 * r + 7) % len(combos)] for r in range(2)]
        assert list(zip(*contract.row_lengths(small, workload).values())) == expected


@pytest.mark.parametrize('mutation', [
    'sparse_above_width', 'extra_above_width', 'missing_pool', 'zero_hole_period', 'unknown_hole_pool',
    'tail_with_full_prefixes', 'even_cycle_stride', 'extra_field', 'nonpositive_scale', 'grid_above_width'])
def test_manifest_rejects_malformed_task_data(mutation, monkeypatch):
    with modules(C128, monkeypatch) as contract:
        workload = deepcopy(contract.load_workload())
        case = next(c for c in workload['cases'] if c['case_id'] == 'batch_2_s0_e0')
        if mutation == 'sparse_above_width':
            case['lengths'] = {'uniform': {'sparse': 129, 'extra': 0}}
        elif mutation == 'extra_above_width':
            case['lengths'] = {'uniform': {'sparse': 0, 'extra': 8257}}
        elif mutation == 'missing_pool':
            case['lengths'] = {'uniform': {'sparse': 0}}
        elif mutation == 'zero_hole_period':
            case['lengths'] = {'uniform': {'sparse': 64, 'extra': 64}, 'holes': {'sparse': 0}}
        elif mutation == 'unknown_hole_pool':
            case['lengths'] = {'uniform': {'sparse': 64, 'extra': 64}, 'holes': {'third': 2}}
        elif mutation == 'tail_with_full_prefixes':
            case['lengths'] = {'uniform': {'sparse': 128, 'extra': 8256}, 'tail': 'legal'}
        elif mutation == 'even_cycle_stride':
            case['lengths'] = {'cycle': {'stride': 2, 'offset': 0}}
        elif mutation == 'extra_field':
            case['rtol'] = 1
        elif mutation == 'nonpositive_scale':
            workload['scalars']['sm_scale'] = 0.0
        else:
            workload['length_grid']['extra'].append(8257)
        with pytest.raises(ValueError):
            contract.case_manifest(workload)


@pytest.mark.parametrize('source', [
    'import torch\ndef f(x):\n return torch.softmax(x, -1)',
    'import torch.nn.functional as F\ndef f(q, k, v):\n return F.scaled_dot_product_attention(q, k, v)',
    'import torch as t\ndef f(x):\n return t.exp(x)',
    'from torch import logsumexp',
    'def f(a, b):\n return a @ b',
    'import torch\ndef f(a, b):\n return torch.bmm(a, b)',
    'from sglang.kernels.ops.attention.dsa.tilelang_kernel import dpsk_v4_fp8_attention_fwd',
    'import tilelang',
    'import flash_mla',
    'import task_reference',
])
def test_import_guard_rejects_library_attention(source, monkeypatch):
    with modules(MAIN_ONLY, monkeypatch) as contract:
        with pytest.raises(RuntimeError):
            contract.assert_source_independent(source)


def test_import_guard_allows_flydsl_math_and_plumbing(monkeypatch):
    source = ('import torch\nfrom flydsl.expr import math as fmath\n'
              'def body(v):\n    return fmath.exp(v)\n'
              'def build(**axes):\n    return lambda q, *a: (torch.empty_like(q), torch.empty(q.shape[:-1]))\n')
    with modules(MAIN_ONLY, monkeypatch) as contract:
        contract.assert_source_independent(source)


def _prefix(indices, length):
    return indices[:length]


def test_case_application_sets_lengths_and_index_patterns(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        cases = _cases(inputs_module)
        state = inputs_module.BatchInputs(cases['batch_4_bundle'], device='cpu')
        bundle = state.prepare(cases['batch_4_bundle'])
        assert bundle['sparse_lens'].tolist()[:2] == [128, 0]
        reference_tables = bundle['sparse_indices'].clone()
        # Every case's prefixes are legal, its tails -1, and its lengths declared.
        for name in ('batch_4_s0', 'batch_4_s65', 'batch_4_s128', 'batch_4_holes', 'batch_4_tail', 'batch_4_mixed'):
            inputs = state.prepare(cases[name])
            evidence = inputs_module.check_case_inputs(inputs, cases[name], state)
            lengths, table = inputs['sparse_lens'].tolist(), inputs['sparse_indices'][:, 0]
            for row, length in enumerate(lengths):
                prefix, tail = table[row, :length], table[row, length:]
                if name == 'batch_4_holes':
                    assert (prefix[2::3] == -1).all() and (prefix[0::3] >= 0).all() and (prefix[1::3] >= 0).all()
                else:
                    assert (prefix >= 0).all()
                assert ((tail >= 0) if name == 'batch_4_tail' else (tail == -1)).all()
            if name == 'batch_4_holes':
                assert evidence['sparse']['holes'] == 4 * (128 // 3)
        # The bundle case reproduces the initializer's own tables exactly.
        assert torch.equal(state.prepare(cases['batch_4_bundle'])['sparse_indices'], reference_tables)
        # Prefix entries the initializer kept are unchanged by applying lengths.
        full = state.prepare(cases['batch_4_s128'])['sparse_indices']
        kept = reference_tables >= 0
        assert torch.equal(full[kept], reference_tables[kept])


def test_draws_vary_call_varying_operands_and_hold_the_rest(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(C128, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        case = _cases(inputs_module)['batch_2_s128_e4096']
        state = inputs_module.BatchInputs(case, device='cpu')
        inputs = state.prepare(case)
        # Packed pools hold bytes that read as FP8 NaN, so compare storage bytes.
        raw = lambda tensor: tensor.view(torch.uint8)
        held = {name: raw(inputs[name]).clone() for name in inputs_module.PERSISTENT_INPUTS}
        assert set(held) == {'sparse_lens', 'extra_kv_cache', 'extra_sparse_lens', 'sinks'}
        live = {name: raw(inputs[name]).clone() for name in inputs_module.CALL_VARYING}
        first, second = state.draws(case, [11, 12])
        assert set(first) == set(inputs_module.CALL_VARYING)
        assert all(torch.equal(raw(inputs[n]), v) for n, v in {**held, **live}.items())  # Drawing leaves live buffers.
        for name in inputs_module.CALL_VARYING:
            assert not torch.equal(raw(first[name]), raw(second[name])), name
        inputs_module.load_draw(inputs, first)
        inputs_module.check_case_inputs(inputs, case, state)
        assert all(torch.equal(raw(inputs[n]), v) for n, v in held.items())
        assert (inputs['extra_sparse_indices'][:, 0, :4096] >= 0).all()
        assert (inputs['extra_sparse_indices'][:, 0, 4096:] == -1).all()
        short = _cases(inputs_module)['batch_2_short_history']
        inputs = state.prepare(short)
        assert inputs['extra_sparse_lens'].tolist() == [1, 1]
        assert (inputs['extra_sparse_indices'][:, 0, 0] == -1).all()
        assert inputs_module.effective_entries(inputs).tolist() == [64, 64]


@pytest.mark.parametrize('fault', ['undeclared_hole', 'outside_pool', 'tail', 'lengths'])
def test_case_input_validation_rejects_inconsistent_tables(fault, monkeypatch):
    pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        case = _cases(inputs_module)['batch_2_s65']
        state = inputs_module.BatchInputs(case, device='cpu')
        inputs = state.prepare(case)
        if fault == 'undeclared_hole':
            inputs['sparse_indices'][0, 0, 10] = -1
        elif fault == 'outside_pool':
            inputs['sparse_indices'][0, 0, 10] = inputs_module.capacity(inputs, 'sparse')
        elif fault == 'tail':
            inputs['sparse_indices'][1, 0, 100] = 0
        else:
            inputs['sparse_lens'][1] = 64
        with pytest.raises(RuntimeError):
            inputs_module.check_case_inputs(inputs, case, state)


def test_output_classification(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        output = torch.ones((2, 1, 2, 4), dtype=torch.bfloat16)
        output[1] = 0
        lse = torch.tensor([[[0.5, 0.25]], [[math.inf, math.inf]]])
        expected = (output, lse)
        assert measure.compare_output((output.clone(), lse.clone()), expected)['status'] == 'PASS'
        assert measure.compare_output({'output': output.clone(), 'lse': lse.clone()}, expected)['status'] == 'PASS'
        far = output.clone()
        far[0] += 1
        assert measure.compare_output((far, lse.clone()), expected)['failure_kind'] == 'numerical_mismatch'
        finite = lse.clone()
        finite[1] = 0
        result = measure.compare_output((output.clone(), finite), expected)
        assert result['failure_kind'] == 'numerical_mismatch'
        for got in [(output.float(), lse), (output, lse * float('nan')), (output, -lse), output,
                    (output * float('nan'), lse), (output, lse.double())]:
            assert measure.compare_output(got, expected)['failure_kind'] == 'output_contract'
        with pytest.raises(ValueError):
            measure.compare_output(expected, (output * float('nan'), lse))


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
    def __init__(self, reference, names):
        self.reference, self.names, self.cost, self.calls, self.out = reference, names, None, 0, None

    def compute(self, args):
        self.cost = 1.0
        self.out = self.reference.run(**dict(zip(self.names, args)))
        return self.out

    def __call__(self, *args):
        self.calls += 1
        return self.compute(args)


class _ValueMemo(_Honest):
    """Returns a stored, correct result for any queries and index tables seen before."""
    def __init__(self, reference, names):
        super().__init__(reference, names)
        self.store = {}

    def __call__(self, *args):
        import torch
        key = (args[0].view(torch.uint8).numpy().tobytes(), args[2].numpy().tobytes())
        if key in self.store:
            self.cost = 0.1
            return tuple(t.clone() for t in self.store[key])
        self.store[key] = tuple(t.clone() for t in self.compute(args))
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
    real = measure.task_inputs.BatchInputs
    monkeypatch.setattr(measure.task_inputs, 'BatchInputs', lambda case, device=None: real(case, device='cpu'))


@pytest.mark.parametrize('kernel_type, status, failure_kind', [
    (_Honest, 'PASS', None),
    (_ValueMemo, 'FAIL', 'timing_input_memoized'),
    (_SkipsEveryThirdCall, 'FAIL', 'numerical_mismatch'),
])
def test_protocol_rejects_known_timed_path_exploits_and_accepts_honest_timing(
        kernel_type, status, failure_kind, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        monkeypatch.setattr(measure, 'choose_checked_samples', lambda repetition, count: list(range(count)))
        kernel = kernel_type(measure.task_reference, list(measure.task_inputs.INPUTS))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        result = measure.time_case(_cases(measure.task_inputs)['batch_2_s65'], role='candidate', launch=kernel)
        assert result['status'] == status, result.get('reason')
        assert result.get('failure_kind') == failure_kind
        assert result['metadata']['timed_output_correctness']['metadata']['checked_invocations'] == (
            measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS)


@pytest.mark.parametrize('mutated', ['q', 'kv_cache', 'sparse_indices', 'sparse_lens', 'sinks'])
def test_timed_guard_rejects_mutation_of_each_input(mutated, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        names = list(measure.task_inputs.INPUTS)

        class Mutating(_Honest):
            def compute(self, args):
                out = super().compute(args)
                # The lowest bit of the first element keeps every input legal.
                args[names.index(mutated)].view(torch.uint8).flatten()[0] ^= 1
                return out

        kernel = Mutating(measure.task_reference, names)
        monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
        with pytest.raises(RuntimeError, match=f'protected input tensor: {mutated}$'):
            measure.time_case(_cases(measure.task_inputs)['batch_2_s65'], role='candidate', launch=kernel)


def test_batched_capture_is_rejected(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        helper = types.SimpleNamespace(
            TimedRun=lambda: types.SimpleNamespace(bound=True),
            benchmark_cuda_graph_or_events=lambda *a, **kw: (
                0.01, {'benchmark_method': 'cuda_graph', 'benchmark_effective_repeats': 8}))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
        result = measure.time_case(_cases(measure.task_inputs)['batch_1_s1'], role='candidate', launch=lambda *a: None)
        assert result['status'] == 'FAIL' and result['failure_kind'] == 'timing_protocol'


class _StaleLengths(_Honest):
    """Reads sparse_lens once, on its first call, and reuses them afterwards."""
    def compute(self, args):
        args = list(args)
        index = self.names.index('sparse_lens')
        if not hasattr(self, 'lengths'):
            self.lengths = args[index].clone()
        args[index] = self.lengths
        return super().compute(args)


# A launch frozen on the bundle lengths still passes the bundle case and s0,
# whose tables hold only -1 so any length selects nothing.
@pytest.mark.parametrize('kernel_type, passing', [(_Honest, 11), (_StaleLengths, 2)])
def test_correctness_reuses_buffers_and_launch_across_cases(kernel_type, passing, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        _cpu(measure, monkeypatch, torch)
        monkeypatch.setattr(measure, 'CORRECTNESS_DRAWS', 1)
        kernel, reuse, statuses = kernel_type(measure.task_reference, list(measure.task_inputs.INPUTS)), {}, []
        storage = set()
        for case in measure.task_inputs.CASES:
            if case['batch'] != 2:
                continue
            result = measure.check_case(case, role='candidate', launch=kernel, reuse=reuse)
            statuses.append(result['status'])
            storage.add(next(iter(reuse.values())).inputs['q'].data_ptr())
        assert len(statuses) == 11 and statuses.count('PASS') == passing
        assert len(storage) == 1
        mixed = measure.check_case(_cases(measure.task_inputs)['batch_2_mixed'], role='candidate',
                                   launch=_Honest(measure.task_reference, list(measure.task_inputs.INPUTS)),
                                   reuse=reuse)
        assert 'fresh_lengths' in mixed['metadata'] and mixed['metadata']['checked_invocations'] == 3


@pytest.mark.parametrize('task', [MAIN_ONLY, C128])
def test_independent_controls_execute_real_callbacks_on_cpu(task, monkeypatch):
    pytest.importorskip('torch')
    with modules(task, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        records = validation.run_controls(measure, device='cpu')
        assert [r['name'] for r in records] == ['mla_fp8_decode', 'mla_attention_known_answer',
                                                'comparator_positive_and_negative']
        assert all(r['status'] == 'PASS' and r['device'] == 'cpu' for r in records)
        selected = records[1]['evidence']['selected_slots']
        assert selected[3] == [] and selected[2] == ['main:3']
        assert selected[4] == (['main:1', 'main:2', 'extra:0', 'extra:1'] if task == C128 else ['main:1', 'main:2'])
        json.dumps(records, allow_nan=False)


@pytest.mark.parametrize('fault', ['zero_codes_as_zero', 'unscaled_groups', 'negative_selected',
                                   'ignore_lengths', 'sink_in_lse', 'no_sink', 'extra_pool_dropped'])
def test_known_answers_detect_specific_reference_faults(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(C128, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        reference = measure.task_reference
        decode, select, attention = reference._decode_slots, reference._selected_slots, reference._flash_mla_reference
        if fault == 'zero_codes_as_zero':
            # Standard FP8 decoding maps a zero code to zero; the format does not.
            # The fixtures' unset codes decode below 2**-5 and their set codes above.
            monkeypatch.setattr(reference, '_decode_slots', lambda cache, slots: torch.where(
                decode(cache, slots).abs() < 2**-5, 0.0, decode(cache, slots)))
        elif fault == 'unscaled_groups':
            # Decode every slot as if all its scale bytes were 127.
            def unscaled(cache, slots):
                raw = cache.view(torch.uint8).clone()
                page_size = raw.shape[1]
                flat = raw.view(raw.shape[0], -1)
                for within in range(page_size):
                    start = page_size * 576 + within * 8
                    flat[:, start:start + 7] = 127
                return decode(raw.view(torch.float8_e4m3fn), slots)
            monkeypatch.setattr(reference, '_decode_slots', unscaled)
        elif fault == 'negative_selected':
            def keeps_negative(indices, lengths, row, capacity):
                length = indices.shape[-1] if lengths is None else lengths[row].item()
                return indices[row, 0, :length].to(torch.int64).clamp(min=0)
            monkeypatch.setattr(reference, '_selected_slots', keeps_negative)
        elif fault == 'ignore_lengths':
            monkeypatch.setattr(reference, '_selected_slots',
                                lambda indices, lengths, row, capacity: select(indices, None, row, capacity))
        elif fault in ('sink_in_lse', 'no_sink', 'extra_pool_dropped'):
            def faulty(**kw):
                if fault == 'no_sink':
                    kw['sinks'] = None
                if fault == 'extra_pool_dropped':
                    kw.update(extra_kv_cache=None, extra_sparse_indices=None, extra_sparse_lens=None)
                output, lse = attention(**kw)
                if fault == 'sink_in_lse':
                    lse = torch.logaddexp(lse, kw['sinks'].float()[None, None, :])
                return output, lse
            monkeypatch.setattr(reference, '_flash_mla_reference', faulty)
        monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['status'] == 'FAIL'
        assert records[-1]['name'] in ('mla_fp8_decode', 'mla_attention_known_answer')


@pytest.mark.parametrize('fault', ['always_accept', 'always_reject', 'output_only'])
def test_controls_detect_broken_comparator(fault, monkeypatch):
    pytest.importorskip('torch')
    with modules(MAIN_ONLY, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        compare = measure.task_compare

        def broken(actual, expected):
            if fault == 'always_reject':
                raise AssertionError('reject everything')
            if fault == 'output_only':
                actual = actual[0] if isinstance(actual, tuple) else actual
                compare.DefaultCompare(atol=1e-2, rtol=1e-2, mode='additive')(actual, expected[0])
        monkeypatch.setattr(compare, 'run', broken)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'comparator_positive_and_negative' and records[-1]['status'] == 'FAIL'


@pytest.mark.parametrize('broken', [False, True])
def test_validate_task_runs_mandatory_controls_and_preserves_manifest(tmp_path, monkeypatch, broken):
    pytest.importorskip('torch')
    task = tmp_path / 'task'
    shutil.copytree(MAIN_ONLY, task)
    with modules(task, monkeypatch) as contract:
        runner = importlib.import_module('evaluate')
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        monkeypatch.setenv('ARENA_EVAL_PHASE', 'task_validation')
        for source in contract.load_config()['baseline']['source_files']:
            (task / source).parent.mkdir(parents=True, exist_ok=True)
            (task / source).write_text('# fake materialization, for orchestration unit test only')
        monkeypatch.setattr(runner, 'require_runtime', lambda w: None)
        monkeypatch.setattr(runner, 'runtime_evidence', lambda w: {'backend': 'injected CPU unit test'})
        original = validation.run_controls
        monkeypatch.setattr(validation, 'run_controls', lambda m, **kw: original(m, device='cpu', **kw))
        calls = []
        monkeypatch.setattr(runner, 'validate_case', lambda case, m, reuse: calls.append(case['case_id']) or {'status': 'PASS'})
        if broken:
            monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        result = parse_report(runner.run('task', 'validate-task'))
        assert [r['test_case_id'] for r in result.cases] == list(measure.task_inputs.CASE_IDS)
        assert result.metadata['candidate_state'] == 'unimplemented'
        if broken:
            assert not result.passed and result.metadata['validation_controls'][-1]['status'] == 'FAIL'
        else:
            assert result.passed and calls == list(measure.task_inputs.CASE_IDS)


@pytest.mark.parametrize('task', TASKS)
def test_real_cli_stub_fails_every_case(task, monkeypatch):
    import subprocess
    result = subprocess.run([sys.executable, 'scripts/evaluate.py', 'candidate', 'correctness'],
                            cwd=task, text=True, capture_output=True, timeout=60)
    assert result.returncode == 1
    report = parse_command_result(result.stdout, role='candidate', action='correctness', returncode=1)
    assert report.cases and all('unimplemented' in row['reason'] for row in report.cases)


@pytest.mark.parametrize('task', [MAIN_ONLY, C4])
def test_exported_binding_reads_axes_from_declared_inputs(task, tmp_path, monkeypatch):
    """Exercise the binding with host sentinels, not a claim of FlyDSL execution."""
    copy = tmp_path / 'task'
    shutil.copytree(task, copy)
    artifact = tmp_path / 'unpacked'
    artifact.mkdir()
    (artifact / 'candidate.py').write_text(
        'def builder(**axes):\n    return lambda *inputs: (axes, inputs)\n')
    with modules(copy, monkeypatch) as contract:
        exporter = importlib.import_module('export_solution')
        workload = contract.load_workload()
        path = artifact / 'sikl_entry.py'
        path.write_text(exporter.tensor_entry({'file': 'candidate.py', 'symbol': 'builder'}, workload))
        spec = importlib.util.spec_from_file_location('mla_binding_unit_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arguments = {name: f'<{name}>' for name in workload['inputs']}
        arguments['q'] = types.SimpleNamespace(shape=(8, 1, 64, 512))
        arguments['kv_cache'] = types.SimpleNamespace(shape=(10, 256, 1, 584))
        if 'extra_kv_cache' in arguments:
            arguments['extra_kv_cache'] = types.SimpleNamespace(shape=(20, 64, 1, 584))
        axes, inputs = module.run(**arguments)
        assert inputs == tuple(arguments[name] for name in workload['inputs'])
        assert axes['batch'] == 8 and axes['pages'] == 10 and axes['topk'] == 128 and 'one' not in axes
        assert axes.get('extra_pages') == (20 if 'extra_kv_cache' in arguments else None)
