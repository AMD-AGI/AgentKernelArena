"""CPU evidence for the mHC SIKL task's contract, guards and timing protocol, not GPU validation."""
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
TASK = ROOT / 'tasks/Aiter-task/mhc_fused_post_pre_flat_rmsnorm_c4_d4096'
MODULE_NAMES = ['task_contract', 'task_inputs', 'task_compare', 'task_initialize',
                'task_reference', 'task_baseline', 'task_measure', 'task_validation', 'evaluate', 'export_solution']
OUTPUTS = ('next_post_mix', 'next_comb_mix', 'layer_input', 'next_residual')
SCALARS = {'rms_eps': 1e-06, 'pre_eps': 1e-06, 'sinkhorn_eps': 1e-06,
           'post_multiplier': 2.0, 'sinkhorn_iters': 20, 'norm_eps': 1e-06}


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


def test_manifest_covers_every_declared_case_with_scalars(monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        workload = contract.load_workload()
        rows = contract.case_manifest(workload)
        assert [row['params']['tokens'] for row in rows] == [2**i for i in range(13)]
        assert [row['shape'] for row in rows] == [[2**i, 4, 4096] for i in range(13)]
        for row in rows:
            assert row['checks'] == ['correctness', 'performance']
            assert {k: row['params'][k] for k in SCALARS} == SCALARS
            assert row['params']['projection_size'] == 16384 and row['params']['uuid']
        assert list(workload['inputs']) == ['x', 'residual', 'post_mix', 'comb_mix', 'proj_weight',
                                            'mix_scale', 'mix_bias', 'rms_eps', 'pre_eps', 'sinkhorn_eps',
                                            'post_multiplier', 'sinkhorn_iters', 'norm_weight', 'norm_eps']
        assert tuple(workload['outputs']) == OUTPUTS
        manifest = CaseManifest.from_result(parse_report({
            'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task', 'status': 'PASS',
            'cases': rows, 'metadata': {'candidate_state': 'unimplemented'}}))
        assert len(manifest.cases) == 13


@pytest.mark.parametrize('mutation', [
    'constraint_violated', 'unsupported_constraint', 'scalar_dtype', 'missing_scalar',
    'undeclared_dimension', 'duplicate_uuid', 'extra_case_field', 'unknown_dtype'])
def test_manifest_rejects_malformed_task_data(mutation, monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        workload = deepcopy(contract.load_workload())
        if mutation == 'constraint_violated':
            workload['axes']['projection_size'] = 8192
        elif mutation == 'unsupported_constraint':
            workload['constraints'] = ['__import__("os").system("true") == 0']
        elif mutation == 'scalar_dtype':
            workload['scalars']['sinkhorn_iters'] = 20.0
        elif mutation == 'missing_scalar':
            del workload['scalars']['norm_eps']
        elif mutation == 'undeclared_dimension':
            workload['inputs']['x']['shape'] = ['tokens', 'width']
        elif mutation == 'duplicate_uuid':
            workload['cases'][1]['uuid'] = workload['cases'][0]['uuid']
        elif mutation == 'extra_case_field':
            workload['cases'][0]['atol'] = 1.0
        else:
            workload['outputs']['layer_input']['dtype'] = 'float16'
        with pytest.raises(ValueError):
            contract.case_manifest(workload)


@pytest.mark.parametrize('source', [
    'import torch\ndef f(x):\n return torch.sigmoid(x)',
    'import torch.nn.functional as F\ndef f(x):\n return F.softmax(x, -1)',
    'import torch as t\ndef f(x):\n return t.nn.functional.rms_norm(x, (4,))',
    'from torch import rsqrt',
    'from torch.nn.functional import softmax as sm',
    'import torch\ndef f(a, b):\n return torch.bmm(a, b)',
    'def f(a, b):\n return a @ b',
    'from aiter.ops.mhc import mhc_fused_post_pre',
    'import task_reference',
    'from . import task_baseline',
    'import importlib',
])
def test_import_guard_rejects_library_operator_computation(source, monkeypatch):
    with modules(TASK, monkeypatch) as contract:
        with pytest.raises(RuntimeError):
            contract.assert_source_independent(source)


def test_import_guard_allows_flydsl_math_intrinsics_and_plumbing(monkeypatch):
    source = '''from functools import lru_cache
import torch
import flydsl.expr as fx
from flydsl.expr import math as fmath

def body(v):
    return fmath.rsqrt(v) * fx.math.exp(v)

@lru_cache(None)
def build(**axes):
    def launch(*args):
        return tuple(torch.empty_like(args[0]) for _ in range(4))
    return launch
'''
    with modules(TASK, monkeypatch) as contract:
        contract.assert_source_independent(source)


def _small_outputs(torch, value=1.0):
    return (torch.full((2, 4, 1), value), torch.full((2, 4, 4), value),
            torch.full((2, 4), value, dtype=torch.bfloat16),
            torch.full((2, 4, 4), value, dtype=torch.bfloat16))


def test_outputs_are_named_and_contract_and_numerical_failures_are_distinct(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        expected = _small_outputs(torch)
        assert measure.compare_output(tuple(t.clone() for t in expected), expected)['status'] == 'PASS'
        named = dict(zip(OUTPUTS, (t.clone() for t in expected)))
        assert measure.compare_output(named, expected)['status'] == 'PASS'
        for index, name in enumerate(OUTPUTS):
            wrong = list(expected)
            wrong[index] = wrong[index] + 1
            result = measure.compare_output(tuple(wrong), expected)
            assert result['failure_kind'] == 'numerical_mismatch'
            assert result['reason'].startswith(name)
            assert result['metadata']['output_contract_passed'] is True
        contract_failures = [
            expected[:3],
            (*expected[:2], expected[2].float(), expected[3]),
            (*expected[:2], expected[2][:, :1], expected[3]),
            (expected[0] * float('nan'), *expected[1:]),
            {name: t for name, t in zip(OUTPUTS[:3], expected)},
            expected[0],
        ]
        for got in contract_failures:
            assert measure.compare_output(got, expected)['failure_kind'] == 'output_contract'
        with pytest.raises(ValueError, match='reference'):
            measure.compare_output(expected, (expected[0] * float('inf'), *expected[1:]))


def _simulated_graph_helper(kernel):
    """The canonical helper's contract on CPU: one replay per sample, prepared by
    ``prepare_fn`` before its start event, ``after_sample`` after its end event,
    ``rerun_ms`` through the same path. A sample's time is the simulated device
    cost the kernel reports for that invocation."""
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
    """Computes the operator on every call; one unit of simulated device time."""
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
    """Returns a stored, correct result for any call-varying operands seen before."""
    def __init__(self, reference, names):
        super().__init__(reference, names)
        self.store = {}

    def __call__(self, *args):
        key = tuple(tuple(a.float().flatten().tolist()) for a in args[:4])
        if key in self.store:
            self.cost = 0.1
            return tuple(t.clone() for t in self.store[key])
        self.store[key] = tuple(t.clone() for t in self.compute(args))
        return self.store[key]


class _StaleUntilDisturbed(_Honest):
    """Skips while its output buffers still hold what it last wrote."""
    def __init__(self, reference, names):
        super().__init__(reference, names)
        self.last = None

    def __call__(self, *args):
        if self.last is not None and all(o.equal(l) for o, l in zip(self.out, self.last)):
            self.cost = 0.01
            return self.out
        self.compute(args)
        self.last = tuple(t.clone() for t in self.out)
        return self.out


class _SkipsEveryThirdCall(_Honest):
    """Leaves its previous result in place on every third call."""
    def __call__(self, *args):
        self.calls += 1
        if self.out is not None and self.calls % 3 == 0:
            self.cost = 0.01
            return self.out
        return self.compute(args)


def _cpu_time_case(measure, monkeypatch, torch, kernel, case=None):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    build = measure.task_inputs.build_case_inputs
    monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda c: build(c, device='cpu'))
    monkeypatch.setitem(sys.modules, '_aka_benchmark', _simulated_graph_helper(kernel))
    return measure.time_case(case or measure.task_inputs.CASES[0], role='candidate', launch=kernel)


@pytest.mark.parametrize('kernel_type, status, failure_kind', [
    (_Honest, 'PASS', None),
    (_ValueMemo, 'FAIL', 'timing_input_memoized'),
    (_StaleUntilDisturbed, 'FAIL', 'numerical_mismatch'),
    (_SkipsEveryThirdCall, 'FAIL', 'numerical_mismatch'),
])
def test_protocol_rejects_known_timed_path_exploits_and_accepts_honest_timing(
        kernel_type, status, failure_kind, monkeypatch):
    """Known exploit behaviours kept as regression fixtures, end to end through
    ``time_case`` with the real bundle initializer, reference and comparator."""
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        # Consecutive indices include a third call, so the one-in-three skipper
        # is checked deterministically; production chooses them secretly.
        monkeypatch.setattr(measure, 'choose_checked_samples',
                            lambda repetition, count: list(range(count)))
        kernel = kernel_type(measure.task_reference, list(measure.task_inputs.INPUTS))
        result = _cpu_time_case(measure, monkeypatch, torch, kernel)
        assert result['status'] == status, result.get('reason')
        assert result.get('failure_kind') == failure_kind
        metadata = result['metadata']
        assert len(metadata['timed_draw_seeds']) == measure.TIMED_DRAWS
        assert len(metadata['unseen_draw_seeds']) == measure.UNSEEN_DRAWS
        assert metadata['timed_output_correctness']['metadata']['checked_invocations'] == (
            measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS)


def test_rotation_changes_only_call_varying_inputs(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        inputs_module = importlib.import_module('task_inputs')
        inputs = inputs_module.build_case_inputs(inputs_module.CASES[0], device='cpu')
        before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        draws = inputs_module.call_varying_draws(inputs, [11, 12])
        assert all(torch.equal(inputs[k], v) for k, v in before.items())
        assert set(draws[0]) == {'x', 'residual', 'post_mix', 'comb_mix'}
        assert all(not torch.equal(draws[0][k], draws[1][k]) for k in draws[0])
        inputs_module.load_draw(inputs, draws[1])
        for name in inputs_module.PERSISTENT_INPUTS:
            assert torch.equal(inputs[name], before[name])
        assert all(inputs[k] is not draws[1][k] and torch.equal(inputs[k], draws[1][k]) for k in draws[1])
        assert {k: inputs[k] for k in SCALARS} == SCALARS


@pytest.mark.parametrize('mutated', ['x', 'residual', 'post_mix', 'comb_mix',
                                     'proj_weight', 'mix_scale', 'mix_bias', 'norm_weight'])
def test_timed_guard_rejects_mutation_of_each_tensor_input(mutated, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        names = list(measure.task_inputs.INPUTS)

        class Mutating(_Honest):
            def __call__(self, *args):
                out = self.compute(args)
                args[names.index(mutated)].view(torch.uint8).flatten()[0] ^= 0x55
                return out

        with pytest.raises(RuntimeError, match=f'protected input tensor: {mutated}$'):
            _cpu_time_case(measure, monkeypatch, torch, Mutating(measure.task_reference, names))


def test_batched_capture_is_rejected(monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        helper = types.SimpleNamespace(
            TimedRun=lambda: types.SimpleNamespace(bound=True),
            benchmark_cuda_graph_or_events=lambda *a, **kw: (
                0.01, {'benchmark_method': 'cuda_graph', 'benchmark_effective_repeats': 8}))
        monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
        monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda c: {})
        monkeypatch.setattr(measure.task_inputs, 'PERSISTENT_INPUTS', ())
        monkeypatch.setattr(measure.task_inputs, 'call_varying_draws', lambda i, seeds: [{} for _ in seeds])
        monkeypatch.setattr(measure, 'case_call', lambda *a, **kw: lambda: None)
        result = measure.time_case({}, role='candidate', launch=lambda: None)
        assert result['status'] == 'FAIL' and result['failure_kind'] == 'timing_protocol'


def test_roles_never_fall_back_and_builder_gets_shape_axes_only(monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        with pytest.raises(RuntimeError, match='no baseline fallback'):
            measure.case_call({}, role='candidate')
        with pytest.raises(ValueError, match='cannot invoke a candidate'):
            measure.case_call({}, role='baseline', launch=lambda *a: None)
        seen = {}
        measure.build_launch(lambda **axes: seen.update(axes) or (lambda *a: None),
                             measure.task_inputs.CASES[-1])
        assert seen == {'tokens': 4096, 'streams': 4, 'hidden_size': 4096}
        with pytest.raises(NotImplementedError):
            measure.build_launch(lambda **kw: (_ for _ in ()).throw(NotImplementedError()),
                                 measure.task_inputs.CASES[0])


def test_independent_controls_execute_real_callbacks_on_cpu(monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        records = validation.run_controls(measure, device='cpu')
        assert [r['name'] for r in records] == ['mhc_fused_post_pre_fixture', 'comparator_positive_and_negative']
        assert all(r['status'] == 'PASS' and r['device'] == 'cpu' for r in records)
        assert set(records[1]['evidence']['rejected_controls']) == {
            *(f'outside_numerical_gate:{name}' for name in OUTPUTS),
            'wrong_sign', 'wrong_shape', 'wrong_dtype', 'nonfinite_output', 'missing_output'}
        json.dumps(records, allow_nan=False)


@pytest.mark.parametrize('fault', ['post_transpose', 'scale_groups', 'bias_offset',
                                   'sinkhorn_iterations', 'output_norm', 'post_multiplier'])
def test_known_answer_detects_specific_reference_faults(fault, monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        reference = measure.task_reference
        post, pre = reference._mhc_post_reference, reference._mhc_pre_reference
        if fault == 'post_transpose':
            monkeypatch.setattr(reference, '_mhc_post_reference',
                                lambda x, residual, post_mix, comb_mix: post(x, residual, post_mix, comb_mix.mT))
        else:
            index, change = {
                'scale_groups': (2, lambda v: v.flip(0)),
                'bias_offset': (3, lambda v: v.roll(1)),
                'post_multiplier': (7, lambda v: 1.0),
                'sinkhorn_iterations': (8, lambda v: 1),
                'output_norm': (9, lambda v: None),
            }[fault]

            def faulty(*args):
                args = list(args)
                args[index] = change(args[index])
                return pre(*args)
            monkeypatch.setattr(reference, '_mhc_pre_reference', faulty)
        # A no-op comparator cannot hide a wrong reference: the known-answer
        # assertion is independent of that comparator.
        monkeypatch.setattr(measure.task_compare, 'run', lambda *a, **kw: None)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'mhc_fused_post_pre_fixture'
        assert records[-1]['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['always_accept', 'always_reject', 'first_output_only'])
def test_controls_detect_broken_comparator(fault, monkeypatch):
    pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        measure = importlib.import_module('task_measure')
        validation = importlib.import_module('task_validation')
        compare = measure.task_compare

        def broken(actual, expected):
            if fault == 'always_reject':
                raise AssertionError('reject everything')
            if fault == 'first_output_only':
                compare.DefaultCompare(atol=1e-2, rtol=1e-2, mode='additive')(
                    actual[OUTPUTS[0]], expected[OUTPUTS[0]])
        monkeypatch.setattr(compare, 'run', broken)
        records = []
        with pytest.raises(RuntimeError):
            validation.run_controls(measure, device='cpu', records=records)
        assert records[-1]['name'] == 'comparator_positive_and_negative'
        assert records[-1]['status'] == 'FAIL'


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
        controls = result.metadata['validation_controls']
        if broken:
            assert not result.passed and controls[-1]['status'] == 'FAIL'
            assert all(r['status'] == 'FAIL' for r in result.cases)
        else:
            assert result.passed and all(r['status'] == 'PASS' for r in controls)
            assert calls == list(measure.task_inputs.CASE_IDS)


def test_reference_output_contract_is_checked_per_declared_output(monkeypatch):
    torch = pytest.importorskip('torch')
    with modules(TASK, monkeypatch):
        runner = importlib.import_module('evaluate')
        measure = importlib.import_module('task_measure')
        build = measure.task_inputs.build_case_inputs
        monkeypatch.setattr(measure.task_inputs, 'build_case_inputs', lambda c: build(c, device='cpu'))
        original = measure.task_reference.run
        # validate_case requires CUDA outputs; a CPU reference fails its contract.
        with pytest.raises(RuntimeError, match='next_post_mix'):
            runner.validate_case(measure.task_inputs.CASES[0], measure)
        monkeypatch.setattr(measure.task_reference, 'run', lambda **kw: original(**kw)[:3])
        with pytest.raises(AssertionError):
            runner.validate_case(measure.task_inputs.CASES[0], measure)
        del torch


def test_exported_binding_passes_declared_inputs_in_order(tmp_path, monkeypatch):
    """Exercise the binding with host sentinels, not a claim of FlyDSL execution."""
    task = tmp_path / 'task'
    shutil.copytree(TASK, task)
    artifact = tmp_path / 'unpacked'
    (artifact / 'nested').mkdir(parents=True)
    (artifact / 'nested/candidate.py').write_text(
        'def arbitrary_builder(**axes):\n    return lambda *inputs: (axes, inputs)\n')
    with modules(task, monkeypatch) as contract:
        exporter = importlib.import_module('export_solution')
        workload = contract.load_workload()
        wrapper = exporter.tensor_entry({'file': 'nested/candidate.py', 'symbol': 'arbitrary_builder'}, workload)
        path = artifact / 'sikl_entry.py'
        path.write_text(wrapper)
        spec = importlib.util.spec_from_file_location('mhc_binding_unit_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arguments = {name: f'<{name}>' for name in workload['inputs']}
        arguments['residual'] = types.SimpleNamespace(shape=(8, 4, 16))
        axes, inputs = module.run(**arguments)
        assert axes == {'tokens': 8, 'streams': 4, 'hidden_size': 16}
        assert inputs == tuple(arguments[name] for name in workload['inputs'])
        accepted = dict(pass_compilation=True, pass_correctness=True, pass_tool_gate=True,
                        workload_consistent=True, benchmark_method_consistent=True,
                        valid_baseline_cases=13, valid_optimized_cases=13,
                        best_optimized_execution_time=0.01)
        for key, value in [('pass_correctness', False), ('valid_optimized_cases', 12)]:
            with pytest.raises(ValueError):
                exporter.accepted_result({**accepted, key: value}, 13)
