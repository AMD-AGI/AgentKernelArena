"""CPU observations of real protected wrappers; native HIP qualification is separate."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('14044_PositionWiseFeedForward', '14069_TransformerFFNLayer',
         '5334_GateGRUSelectionLayer', '8325_MaskedLanguageModel')


def load(path):
    spec = importlib.util.spec_from_file_location('hip5_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parts(name):
    root = ROOT / 'tasks/torch2hip/gpumode' / name
    module = load(next((root / 'pytorch_code_module').glob('*.py')))
    functional = load(next((root / 'pytorch_code_functional').glob('*.py')))
    replay = load(root / 'eval_tools/replay_validation.py')
    runner = load(root / 'eval_tools/evaluate.py')
    checks = load(root / 'eval_tools/correctness_check.py')
    return root, module, functional, replay, runner, checks


def models(name, source, functional, checks):
    args, kwargs = source.get_init_inputs()
    torch.manual_seed(0)
    baseline = getattr(source, name.split('_', 1)[1])(*args, **kwargs).eval()
    candidate = getattr(functional, name.split('_', 1)[1])(*args, **kwargs).eval()
    aligned, evidence = checks._align_state_dict(baseline, candidate)
    assert aligned, evidence
    return baseline, candidate


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    canonical = load(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', canonical)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)


def benchmark(mode, *, bad_first=False, omit_observer=False, wrong_count=False,
              replay_error=None, work_counter=None):
    calls = []
    def bench(invoke, **kwargs):
        assert kwargs['warmup'] == 10
        assert 'max_graph_repeats' not in kwargs
        timed = kwargs['timed_run']
        prepare = kwargs.get('prepare_fn')
        def prepared_invoke():
            if prepare is not None:
                prepare()
            return invoke()
        for _ in range(kwargs['warmup']):
            prepared_invoke()
        if mode == 'graph':
            # Existing canonical prepare schedule: estimate capture, two estimate
            # replays, final capture, prime final graph. None are scored samples.
            for _ in range(4):
                captured = prepared_invoke()
            if prepare is not None:
                prepare()
            captured.copy_(invoke())
        else:
            captured = None
        postscore = False
        post_calls = 0
        def replay():
            nonlocal post_calls
            if postscore and replay_error is not None:
                post_calls += 1
                replay_error(post_calls)
            if mode == 'graph':
                captured.copy_(invoke())
                return captured
            return invoke()
        def prepared_replay():
            if prepare is not None:
                prepare()
            return replay()
        timed._bind(prepared_replay, captured)
        for index in range(kwargs['repetition']):
            if prepare is not None:
                prepare()
            if timed.before_sample is not None:
                timed.before_sample(1)
            before = work_counter() if work_counter is not None else None
            output = replay()
            after = work_counter() if work_counter is not None else None
            if bad_first and index == 0:
                output = output + 2
            timed.outputs = output
            calls.append({'index': index, 'computations': None if before is None else after-before})
            if timed.after_sample is not None and not omit_observer:
                timed.after_sample(output)
        postscore = True
        return .125, {'benchmark_samples': kwargs['repetition'] + int(wrong_count),
                      'benchmark_effective_repeats': 1,
                      'benchmark_method': 'cuda_graph' if mode == 'graph' else 'cuda_event_fallback',
                      'benchmark_timed_run_kind': 'captured_graph' if mode == 'graph' else 'eager_callable'}
    return bench, calls


def install(replay, runner, mode, **options):
    bench, calls = benchmark(mode, **options)
    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
                           benchmark_cuda_graph_or_events=bench, _compare_results=torch.allclose)
    replay.install(perf, runner.output_contract)
    return perf, calls


def snapshots(model, inputs):
    return copy.deepcopy(inputs), copy.deepcopy(model.state_dict())


def assert_restored(model, inputs, before):
    for actual, expected in zip(inputs, before[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, expected in before[1].items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'event'))
@pytest.mark.parametrize('role', ('baseline', 'candidate'))
def test_all_declared_cases_use_same_measured_unit_changed_oracle_and_restore(name, mode, role):
    root, source, functional, replay, runner, checks = parts(name)
    baseline, candidate = models(name, source, functional, checks)
    model = baseline if role == 'baseline' else candidate
    rows = json.loads((root / 'workload.json').read_text())['cases']
    # Some protected wrappers use positional defaults; inspect the actual one.
    import inspect
    fn = inspect.signature(type(candidate).forward).parameters['fn'].default
    for index, values in enumerate(source.get_inputs()):
        inputs = list(values)
        runner.check_case_identity(rows[index], inputs)
        if name == '11709_InnerProd':
            load(root / 'eval_tools/case_controls.py').configure_models((baseline, candidate), inputs)
        before = snapshots(model, inputs)
        perf, calls = install(replay, runner, mode)
        with torch.no_grad():
            elapsed, evidence = perf.cal_hip_latency(model, inputs, None if role == 'baseline' else fn,
                                                     n_iter=2)
        assert elapsed == .125 and len(calls) == 2
        assert evidence['validated_sample_count'] == 2
        assert evidence['changed_input_replay_count'] == len(inputs)
        assert evidence['restored_input_replay_valid'] is True
        assert {record['operand'] for record in evidence['changed_input_replays']} == set(range(len(inputs)))
        assert all(record['original_value'] != record['replay_value'] and
                   record['reference_max_abs_delta'] > 0 for record in evidence['changed_input_replays'])
        assert_restored(model, inputs, before)


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'event'))
@pytest.mark.parametrize('defect', ('none', 'bad_first', 'no_observer', 'wrong_count', 'cached_result',
                                  'input_mutation', 'weight_mutation', 'changed_replay_exception'))
def test_actual_wrapper_finite_failures_all_100_samples_and_finally_restore(name, mode, defect):
    root, source, functional, replay, runner, checks = parts(name)
    _, model = models(name, source, functional, checks)
    inputs = list(next(source.get_inputs()))
    if name == '11709_InnerProd':
        load(root / 'eval_tools/case_controls.py').configure_models((model,), inputs)
    before = snapshots(model, inputs)
    import inspect
    correct = inspect.signature(type(model).forward).parameters['fn'].default
    cached = None
    def candidate(*args, **kwargs):
        nonlocal cached
        if cached is None or defect != 'cached_result':
            output = correct(*args, **kwargs)
            cached = output.detach().clone()
        else:
            output = cached.clone()
        if defect == 'input_mutation':
            args[0].zero_()
        if defect == 'weight_mutation':
            next(model.parameters()).zero_()
        return output
    def replay_error(post_calls):
        if defect == 'changed_replay_exception' and post_calls >= 2:
            raise RuntimeError('changed-input replay sentinel')
    perf, calls = install(replay, runner, mode, bad_first=defect == 'bad_first',
                          omit_observer=defect == 'no_observer', wrong_count=defect == 'wrong_count',
                          replay_error=replay_error)
    with torch.no_grad():
        if defect == 'none':
            _, evidence = perf.cal_hip_latency(model, inputs, candidate)
            assert evidence['validated_sample_count'] == len(calls) == 100
        else:
            with pytest.raises((AssertionError, ValueError, RuntimeError)):
                perf.cal_hip_latency(model, inputs, candidate)
    assert_restored(model, inputs, before)
    if defect == 'bad_first':
        assert len(calls) == 1  # A later good sample must not hide the first bad sample.


@pytest.mark.parametrize('name', ('14044_PositionWiseFeedForward', '14069_TransformerFFNLayer'))
def test_eval_dropout_and_transformer_none_state_contract_is_preserved(name):
    _, source, functional, replay, runner, checks = parts(name)
    _, model = models(name, source, functional, checks)
    inputs = list(next(source.get_inputs()))
    import inspect
    fn = inspect.signature(type(model).forward).parameters['fn'].default
    seen = []
    def candidate(*args, **kwargs):
        if name.startswith('14044_'):
            assert args[-1] == 0.0  # Effective dropout; reference is in eval mode.
        else:
            assert args[5:11] == (1, 0.0, 'gelu', 'SAME', False, None)
            assert args[11] is model
        seen.append(True)
        return fn(*args, **kwargs)
    perf, _ = install(replay, runner, 'graph')
    with torch.no_grad():
        perf.cal_hip_latency(model, inputs, candidate, n_iter=2)
    assert seen and model.training is False
    if name.startswith('14069_'):
        assert not functional.INCREMENTAL_STATE_INSTANCE_ID
        assert not hasattr(model, '_instance_id')


def test_historical_inverse_rejects_any_unreviewed_replay_byte():
    migration = load(ROOT / 'tests/test_hip_task_migration_v2.py')
    import hashlib
    for relative, record in migration._HIP5_MEASURED_DRAW_PINS.items():
        raw = (ROOT / 'tasks' / relative).read_bytes()
        restored = migration._hip5_measured_draw_historical_content(relative, raw)
        assert hashlib.sha256(restored).hexdigest() == record['before_sha256']
        with pytest.raises(AssertionError):
            migration._hip5_measured_draw_historical_content(relative, raw + b'\n# unreviewed\n')


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'event'))
def test_paired_roles_have_identical_fresh_draw_seeds_bytes_and_fixed_parameters(name, mode):
    root, source, functional, replay, runner, checks = parts(name)
    baseline, candidate = models(name, source, functional, checks)
    inputs = list(next(source.get_inputs()))
    if name == '11709_InnerProd':
        load(root / 'eval_tools/case_controls.py').configure_models((baseline,candidate), inputs)
    import inspect
    fn = inspect.signature(type(candidate).forward).parameters['fn'].default
    evidence = []
    for model, hip_fn in ((baseline,None),(candidate,fn)):
        values = copy.deepcopy(inputs)
        before = snapshots(model, values)
        perf,_ = install(replay,runner,mode)
        with torch.no_grad():
            _,metadata = perf.cal_hip_latency(model,values,hip_fn,n_iter=3)
        assert_restored(model, values, before)
        assert metadata['measured_input_draw_count'] == metadata['measured_input_distinct_draws'] == 3
        evidence.append(metadata)
    assert evidence[0]['measured_input_seed_base'] == evidence[1]['measured_input_seed_base']
    assert evidence[0]['measured_input_draws'] == evidence[1]['measured_input_draws']
    assert all(all(seed is not None for seed in record['seeds'])
               for record in evidence[0]['measured_input_draws'])


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'event'))
def test_legal_compute_on_change_cache_must_recompute_all_100_measured_invocations(name,mode):
    root,source,functional,replay,runner,checks = parts(name)
    _,model = models(name,source,functional,checks)
    inputs = list(next(source.get_inputs()))
    if name == '11709_InnerProd':
        load(root / 'eval_tools/case_controls.py').configure_models((model,), inputs)
    import inspect
    correct = inspect.signature(type(model).forward).parameters['fn'].default
    cached_inputs = None
    cached_output = None
    computations = 0
    def candidate(*args,**kwargs):
        nonlocal cached_inputs,cached_output,computations
        operands = args[:len(inputs)]
        if cached_inputs is None or any(not torch.equal(a,b) for a,b in zip(operands,cached_inputs)):
            cached_output = correct(*args,**kwargs).detach().clone()
            cached_inputs = copy.deepcopy(operands)
            computations += 1
        return cached_output.clone()
    perf,calls = install(replay,runner,mode,work_counter=lambda:computations)
    before = snapshots(model,inputs)
    with torch.no_grad():
        _,metadata = perf.cal_hip_latency(model,inputs,candidate)
    assert metadata['validated_sample_count'] == metadata['measured_input_distinct_draws'] == 100
    assert [record['computations'] for record in calls] == [1]*100
    assert_restored(model,inputs,before)


class _CanonicalCudaStub:
    """CPU tensors through the real canonical control flow; no GPU timings."""
    def __init__(self, monkeypatch, mode):
        from contextlib import contextmanager, nullcontext
        self.mode = mode
        self.active_graph = None
        self.capture_count = 0
        self.graph_replay_count = 0
        self.elapsed_reads = 0
        self.in_event_interval = False
        owner = self

        class Stream:
            def wait_stream(self, _other): pass
            def synchronize(self): pass

        self.stream = Stream()

        class Event:
            def __init__(self, enable_timing=True):
                assert enable_timing
                self.start = None
            def record(self, _stream=None):
                self.start = not owner.in_event_interval
                owner.in_event_interval = not owner.in_event_interval
            def synchronize(self): pass
            def elapsed_time(self, _end):
                owner.elapsed_reads += 1
                # Both estimate replays execute before the empty estimate gate.
                return 1e-8 if mode == 'empty_estimate' else .125

        class Graph:
            def __init__(self):
                self.calls = []
            def replay(self):
                owner.graph_replay_count += 1
                assert self.calls
                for invoke, output in self.calls:
                    output.copy_(invoke())

        @contextmanager
        def graph_capture(graph):
            owner.capture_count += 1
            if ((mode == 'first_capture_error' and owner.capture_count == 1) or
                    (mode == 'final_capture_error' and owner.capture_count == 2)):
                raise ValueError('CPU capture-error sentinel after preparation')
            owner.active_graph = graph
            try:
                yield graph
            finally:
                owner.active_graph = None

        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
        monkeypatch.setattr(torch.cuda, 'Stream', Stream)
        monkeypatch.setattr(torch.cuda, 'current_stream', lambda: self.stream)
        monkeypatch.setattr(torch.cuda, 'stream', lambda _stream: nullcontext())
        monkeypatch.setattr(torch.cuda, 'Event', Event)
        monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
        monkeypatch.setattr(torch.cuda, 'graph', graph_capture)
        monkeypatch.delenv('AKA_BENCHMARK_FORCE_EVENT', raising=False)
        if mode == 'forced_event':
            monkeypatch.setenv('AKA_BENCHMARK_FORCE_EVENT', '1')

    def invoke(self, fn):
        output = fn()
        if self.active_graph is not None:
            self.active_graph.calls.append((fn, output))
        return output


def _install_actual_canonical(replay, runner, monkeypatch, mode, *, bad_first=False):
    canonical = load(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', canonical)
    device = _CanonicalCudaStub(monkeypatch, mode)
    evidence = {'before_callbacks': 0, 'after_callbacks': 0}

    def actual_benchmark(invoke, **kwargs):
        # Do not replace canonical capture, graph replay, Event samples, fallback
        # handling, TimedRun binding or observation code. Only CUDA primitives
        # and device arithmetic are modeled on CPU.
        measured = kwargs['prepare_fn'].__self__
        evidence['controller'] = measured
        collector = kwargs['timed_run']
        before = collector.before_sample
        after = collector.after_sample

        def before_sample(calls):
            assert not device.in_event_interval
            state = snapshots(measured.module, measured.inputs)
            before(calls)
            assert_restored(measured.module, measured.inputs, state)
            evidence['before_callbacks'] += 1

        def after_sample(output):
            assert not device.in_event_interval
            state = snapshots(measured.module, measured.inputs)
            original_output = output.detach().clone()
            if bad_first and evidence['after_callbacks'] == 0:
                # Inject an erroneous modeled device result, outside the trusted
                # production observer. A later good sample cannot conceal it.
                after(output + 2)
            else:
                after(output)
            torch.testing.assert_close(output, original_output, rtol=0, atol=0)
            assert_restored(measured.module, measured.inputs, state)
            evidence['after_callbacks'] += 1

        collector.before_sample = before_sample
        collector.after_sample = after_sample
        return canonical.benchmark_cuda_graph_or_events(
            lambda: device.invoke(invoke), **kwargs)

    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
                          benchmark_cuda_graph_or_events=actual_benchmark,
                          _compare_results=torch.allclose)
    replay.install(perf, runner.output_contract)
    return perf, device, evidence


def _canonical_case(name):
    root, source, functional, replay, runner, checks = parts(name)
    baseline, model = models(name, source, functional, checks)
    inputs = list(next(source.get_inputs()))
    if name == '11709_InnerProd':
        load(root / 'eval_tools/case_controls.py').configure_models((baseline, model), inputs)
    import inspect
    fn = inspect.signature(type(model).forward).parameters['fn'].default
    return baseline, model, inputs, fn, replay, runner


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'explicit_event', 'forced_event'))
def test_actual_canonical_graph_and_explicit_event_check_all_100_draws_and_restore(name, mode, monkeypatch):
    _, model, inputs, fn, replay, runner = _canonical_case(name)
    original = snapshots(model, inputs)
    perf, device, observations = _install_actual_canonical(replay, runner, monkeypatch, mode)
    with torch.no_grad():
        elapsed, evidence = perf.cal_hip_latency(model, inputs, fn,
                                               use_cuda_graph=mode != 'explicit_event')
    assert elapsed == .125  # Constant stub, not a measured GPU or CPU time.
    graph = mode == 'graph'
    assert evidence['benchmark_timed_run_kind'] == ('captured_graph' if graph else 'eager_callable')
    assert observations['before_callbacks'] == observations['after_callbacks'] == 100
    assert evidence['validated_sample_count'] == evidence['measured_input_distinct_draws'] == 100
    first = 15 if graph else 10
    assert [draw['prepare_index'] for draw in evidence['measured_input_draws']] == list(range(first, first+100))
    assert all(all(seed is not None for seed in draw['seeds']) for draw in evidence['measured_input_draws'])
    assert device.capture_count == (2 if graph else 0)
    if mode == 'forced_event':
        assert evidence['benchmark_fallback_reason'] == 'forced_event_baseline'
    elif mode == 'explicit_event':
        assert evidence['benchmark_fallback_reason'] == 'cuda_graph_disabled'
    assert_restored(model, inputs, original)


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode,preparations', (('empty_estimate', 13),
                                           ('first_capture_error', 11),
                                           ('final_capture_error', 14)))
def test_actual_canonical_automatic_fallback_with_collector_fails_closed_and_restores(name, mode, preparations, monkeypatch):
    _, model, inputs, fn, replay, runner = _canonical_case(name)
    original = snapshots(model, inputs)
    perf, device, observations = _install_actual_canonical(replay, runner, monkeypatch, mode)
    with torch.no_grad(), pytest.raises(RuntimeError, match='timed_run cannot validate') as error:
        perf.cal_hip_latency(model, inputs, fn)
    assert error.value.__cause__ is not None
    if mode == 'empty_estimate':
        assert 'empty_cuda_graph_capture' in str(error.value.__cause__)
        assert device.graph_replay_count == 2
    else:
        assert 'CPU capture-error sentinel' in str(error.value.__cause__)
    assert observations['controller'].preparations == preparations
    assert observations['controller'].checked == 0
    assert observations['before_callbacks'] == observations['after_callbacks'] == 0
    # No eager measurement silently follows a failed capture/estimate.
    assert device.elapsed_reads == (2 if mode != 'first_capture_error' else 0)
    assert_restored(model, inputs, original)


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('mode', ('graph', 'explicit_event'))
def test_actual_canonical_first_bad_observed_output_is_rejected_and_restored(name, mode, monkeypatch):
    _, model, inputs, fn, replay, runner = _canonical_case(name)
    original = snapshots(model, inputs)
    perf, _, observations = _install_actual_canonical(replay, runner, monkeypatch, mode, bad_first=True)
    expected_error = RuntimeError if mode == 'graph' else ValueError
    expected_message = 'timed_run cannot validate' if mode == 'graph' else 'protected reference'
    with torch.no_grad(), pytest.raises(expected_error, match=expected_message):
        perf.cal_hip_latency(model, inputs, fn, use_cuda_graph=mode != 'explicit_event')
    assert observations['before_callbacks'] == 1
    assert observations['after_callbacks'] == observations['controller'].checked == 0
    assert_restored(model, inputs, original)


def test_actual_canonical_graph_vs_explicit_event_records_different_scored_draw_ids(monkeypatch):
    _, model, inputs, fn, replay, runner = _canonical_case(TASKS[0])
    records = []
    for mode in ('graph', 'explicit_event'):
        perf, _, _ = _install_actual_canonical(replay, runner, monkeypatch, mode)
        with torch.no_grad():
            _, evidence = perf.cal_hip_latency(model, inputs, fn,
                                               use_cuda_graph=mode != 'explicit_event')
        records.append(evidence)
    assert records[0]['measured_input_seed_base'] == records[1]['measured_input_seed_base']
    assert records[0]['measured_input_draws'][0]['prepare_index'] == 15
    assert records[1]['measured_input_draws'][0]['prepare_index'] == 10
    assert records[0]['measured_input_draws'][0]['seeds'] != records[1]['measured_input_draws'][0]['seeds']
    # Equal preparation IDs, where they overlap, are exactly reproducible even
    # across kinds. Do not claim that their 100 scored seed sequences are equal.
    graph = {draw['prepare_index']: draw for draw in records[0]['measured_input_draws']}
    event = {draw['prepare_index']: draw for draw in records[1]['measured_input_draws']}
    assert all(graph[index] == event[index] for index in graph.keys() & event.keys())
