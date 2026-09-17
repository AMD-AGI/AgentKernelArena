"""CPU routing/oracle/dispatch controls; no FlyDSL or GPU qualification claim."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/silu_and_mul_fq_kernel'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch, tmp_path):
    # Execute the real protected function bodies, replacing GPU allocation and
    # the candidate builder with explicitly CPU-only test doubles.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    package = types.ModuleType('scripts')
    package.__path__ = [str(TASK / 'scripts')]
    monkeypatch.setitem(sys.modules, 'scripts', package)
    replay = load(TASK/'scripts/replay_checks.py', 'scripts.replay_checks')
    routing = load(TASK/'scripts/routing_checks.py', 'scripts.routing_checks')
    monkeypatch.setitem(sys.modules, 'scripts.replay_checks', replay)
    monkeypatch.setitem(sys.modules, 'scripts.routing_checks', routing)
    scope = dict(math=math, json=json, Path=Path, os=__import__('os'),
                 _CANDIDATE_DIR=str(tmp_path), KERNEL_FILE='kernel.py',
                 prepare_check=replay.prepare_check, verify_timed_run=replay.verify_timed_run)
    tree = ast.parse((TASK/'test_kernel_harness.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope[target.id] = value
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(functions, type_ignores=[]), str(TASK/'test_kernel_harness.py'), 'exec'), scope)
    shape_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ALL_SHAPES' for t in node.targets))
    exec(compile(ast.Module([shape_node], type_ignores=[]), '<original shapes>', 'exec'), scope)
    scope['HARNESS_SHAPES'] = scope['ALL_SHAPES']
    for name in ('randn', 'arange', 'zeros', 'tensor'):
        original = getattr(torch, name)
        def allocate(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, allocate)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    try:
        yield scope, routing
    finally:
        torch.set_num_threads(prior_threads)


def buffers(h):
    inputs = h['_make_inputs'](h['ALL_SHAPES'][0], seed=42)
    return inputs, h['_alloc_outputs'](inputs)


def cpu_candidate(h, inputs, outputs, behavior='correct', record=None):
    """Numerical CPU double, not a FlyDSL implementation or GPU proof."""
    payload, scale_bytes = outputs
    width, rows = inputs['inter_dim'], inputs['rows']
    real = h['_torch_ref_silu_mul'](inputs['x'], width)
    _, exponents = h['reference_mxfp4'](real, width // 32)
    grid = real.view(rows, width//32, 32) * torch.pow(
        2., 127.-torch.tensor(exponents, dtype=torch.float32)).unsqueeze(-1)
    grid = grid.reshape(rows, width)
    code, _ = h['_nearest_e2m1_code'](grid)
    code = code.to(torch.uint8) | ((grid < 0).to(torch.uint8) * 8)
    packed = code[:, ::2] | (code[:, 1::2] << 4)
    # Separate implementation of the public ID decode for the candidate double.
    ids = inputs['sorted_ids'].tolist()
    topk = rows // inputs['token_num']
    order = [(value & ((1 << 24)-1))*topk + (value >> 24) for value in ids]
    if record is not None:
        record.append({'ids':inputs['sorted_ids'].clone(), 'x':inputs['x'].clone(),
                       'x_ptr':inputs['x'].data_ptr(), 'ids_ptr':inputs['sorted_ids'].data_ptr()})
    if behavior == 'raise':
        inputs['sorted_ids'].zero_()
        raise RuntimeError('CPU candidate launch failed')
    if behavior == 'input_mutation':
        inputs['x'].add_(1)
    if behavior == 'sorted_payload':
        packed = packed[order]
    elif behavior == 'wrong_payload':
        packed.zero_()
    if behavior != 'no_payload_write':
        payload.copy_(packed)
    if behavior in ('identity_scales', 'cached_identity_ids'):
        scale_order = list(range(rows))
    elif behavior == 'wrong_slot':
        scale_order = [(value & ((1 << 24)-1))*topk for value in ids]
    else:
        scale_order = order
    offsets = torch.tensor(h['_scale_tiled_offsets'](rows, width//32), dtype=torch.long)
    if behavior != 'no_scale_write':
        scale_bytes[offsets] = torch.tensor(exponents[scale_order], dtype=torch.uint8)
    return outputs


def test_routing_reference_has_independent_known_sorted_scale_coordinates(harness):
    h, routing = harness
    inputs = dict(rows=4, token_num=2, num_sorted_rows=4, inter_dim=32, scale_cols=1,
                  x=torch.zeros(4, 64), sorted_ids=torch.tensor([0x1000001, 0, 0x1000000, 1], dtype=torch.int32),
                  num_valid_ids=torch.tensor([4], dtype=torch.int32))
    exact = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., -4.]*4)
    real = exact[None, :] * torch.tensor([1., 2., 4., 8.])[:, None]
    order, expected_real, expected_deq, expected_scales = routing.routing_reference(
        inputs, lambda x, width: real, h['reference_mxfp4'])
    assert order.tolist() == [3, 0, 1, 2]
    assert expected_scales.tolist() == [[130], [127], [128], [129]]
    torch.testing.assert_close(expected_real, torch.stack([exact*8, exact, exact*2, exact*4]), rtol=0, atol=0)
    torch.testing.assert_close(expected_deq, expected_real, rtol=0, atol=0)


def test_control_crosses_slots_tiles_and_has_distinct_magnitudes(harness):
    h, routing = harness
    inputs, outputs = buffers(h)
    routing.set_routing_control_(inputs)
    order = routing.input_rows_in_sorted_order(inputs)
    positions = torch.arange(128)
    assert torch.equal(order.sort().values, positions)
    assert bool((order % 2 != positions % 2).all())
    assert bool((order // 32 != positions // 32).all())
    peaks = inputs['x'][:, 1024:1056].abs().amax(dim=1)
    assert len(peaks.unique()) == 128
    real = h['_torch_ref_silu_mul'](inputs['x'], 1024)
    _, scales = h['reference_mxfp4'](real, 32)
    assert (scales != scales[order.numpy()]).sum() > scales.size // 2


@pytest.mark.parametrize('behavior', ['correct', 'identity_scales', 'cached_identity_ids', 'sorted_payload',
                                     'wrong_slot', 'wrong_payload', 'no_payload_write', 'no_scale_write',
                                     'input_mutation', 'raise'])
def test_candidate_executed_routing_control_and_finally_restore(harness, behavior):
    h, routing = harness
    inputs, outputs = buffers(h)
    originals = {name:value.clone() for name,value in inputs.items() if isinstance(value, torch.Tensor)}
    pointers = {name:inputs[name].data_ptr() for name in originals}
    calls = []
    def invoke():
        return cpu_candidate(h, inputs, outputs, behavior, calls)
    args = (inputs, outputs, invoke, h['_torch_ref_silu_mul'], h['reference_mxfp4'], h['decode_kernel_fp4'])
    if behavior == 'correct':
        assert routing.check_routing_call(*args)['routing_correctness'] == 'PASS'
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            routing.check_routing_call(*args)
    assert len(calls) == 1
    assert not torch.equal(calls[0]['ids'], originals['sorted_ids'])
    for name, original in originals.items():
        assert inputs[name].data_ptr() == pointers[name]
        assert torch.equal(inputs[name], original)


@pytest.mark.parametrize('behavior', ['correct', 'identity_scales'])
def test_real_timedrun_binding_executes_changed_ids_without_replacement(harness, behavior):
    from src.tools.perf.aka_benchmark import TimedRun
    h, routing = harness
    inputs, outputs = buffers(h)
    original_ids = inputs['sorted_ids'].clone()
    calls = []
    def measured():
        return cpu_candidate(h, inputs, outputs, behavior, calls)
    timed = TimedRun()
    timed._bind(measured, measured())
    args = (timed, inputs, h['_torch_ref_silu_mul'], h['reference_mxfp4'], h['decode_kernel_fp4'])
    if behavior == 'correct':
        assert routing.verify_routing_replay(*args)['routing_timed_replay'] == 'PASS'
    else:
        with pytest.raises(AssertionError, match='Routing E8M0 scales'):
            routing.verify_routing_replay(*args)
    assert len(calls) == 2 and calls[0]['ids_ptr'] == calls[1]['ids_ptr']
    assert not torch.equal(calls[0]['ids'], calls[1]['ids'])
    assert torch.equal(inputs['sorted_ids'], original_ids)


@pytest.mark.parametrize('behavior', ['correct', 'identity_scales'])
def test_public_correctness_binding_runs_candidate_routing_control(harness, behavior):
    h, routing = harness
    calls = []
    def builder(width, topk, **kwargs):
        assert (width, topk, kwargs['quant_mode']) == (1024, 2, 'fp4')
        def launch(x, payload, scales, ids, valid, expert, bias, tokens, count, stream):
            inputs = dict(x=x, sorted_ids=ids, num_valid_ids=valid, topk_ids=expert, bias=bias,
                          rows=tokens*topk, token_num=tokens, inter_dim=width)
            return cpu_candidate(h, inputs, (payload, scales), behavior, calls)
        return launch
    h['_load_kernel'] = lambda *args: types.SimpleNamespace(build_silu_and_mul_fq_module=builder)
    actions = load(TASK/'scripts/task_actions.py', 'silu_task_actions')
    original_cases = []
    def old_correctness(shapes, verbose):
        original_cases.extend(shapes)
        return {'correct':True, 'num_correct':len(shapes)}
    public = types.SimpleNamespace(run_correctness=old_correctness, run_routing_correctness=h['run_routing_correctness'])
    if behavior == 'correct':
        actions.check(public)
    else:
        with pytest.raises(AssertionError, match='Routing E8M0 scales'):
            actions.check(public)
    assert original_cases == h['ALL_SHAPES'] and len(calls) == 1


def test_no_partial_valid_or_padding_policy_is_invented(harness):
    h, routing = harness
    inputs, outputs = buffers(h)
    inputs['num_valid_ids'].fill_(127)
    with pytest.raises(AssertionError, match='all-valid'):
        routing.input_rows_in_sorted_order(inputs)


def test_original_case_file_and_action_timing_preserved():
    assert hashlib.sha256((TASK/'cases.json').read_bytes()).hexdigest() == '5c48da0714d4090bba7febcb059171ff341d55506e11bd4530cbbd442345ef75'
    actions = load(TASK/'scripts/task_actions.py', 'silu_original_cases')
    expected = [(tokens, 1024, 2, 'fp4') for tokens in [64, 128, 256, 512, 1024]]
    assert actions.CORRECTNESS_CASES == actions.PERFORMANCE_CASES == expected
    calls = []
    actions.performance(types.SimpleNamespace(arena_benchmark=lambda **kwargs: calls.append(kwargs)))
    assert calls == [{'shapes':expected, 'warmup':10, 'iters':100}]


@pytest.mark.parametrize('behavior', ['correct', 'identity_scales'])
@pytest.mark.parametrize('timing_kind', ['captured_graph', 'eager_callable'])
def test_actual_arena_benchmark_replays_control_after_unchanged_timed_values(harness, behavior, timing_kind):
    from src.tools.perf.aka_benchmark import TimedRun
    h, routing = harness
    calls, benchmark_calls = [], []
    inputs_seen = []
    phase = {'value':'setup'}
    def builder(width, topk, **kwargs):
        def launch(x, payload, scales, ids, valid, expert, bias, tokens, count, stream):
            inputs = dict(x=x, sorted_ids=ids, num_valid_ids=valid, topk_ids=expert, bias=bias,
                          rows=tokens*topk, token_num=tokens, inter_dim=width)
            inputs_seen.append(inputs)
            calls.append(phase['value'])
            return cpu_candidate(h, inputs, (payload, scales), behavior)
        return launch
    h['_load_kernel'] = lambda *args: types.SimpleNamespace(build_silu_and_mul_fq_module=builder)
    original, _ = buffers(h)
    def benchmark(fn, warmup, repetition, timed_run=None):
        benchmark_calls.append((warmup, repetition, timed_run is not None))
        phase['value'] = 'measured'
        if timed_run is not None:
            assert torch.equal(inputs_seen[-1]['x'], original['x'])
            assert torch.equal(inputs_seen[-1]['sorted_ids'], original['sorted_ids'])
            output = fn()
            def bound():
                phase['value'] = 'bound_replay'
                try:
                    return fn()
                finally:
                    phase['value'] = 'outside_timing'
            timed_run._bind(bound, output)
        else:
            fn()
        phase['value'] = 'outside_timing'
        return .25, {'benchmark_method':'cuda_graph' if timing_kind == 'captured_graph' else 'cuda_event_fallback',
                     'benchmark_timed_run_kind':timing_kind}
    h['TimedRun'] = TimedRun
    h['benchmark_cuda_graph_or_events'] = benchmark
    if behavior == 'correct':
        result = h['arena_benchmark'](shapes=h['ALL_SHAPES'][:1], warmup=10, iters=100, verbose=False)
        assert len(result) == 1 and result[0]['test_case_id'] == 'test_case_0'
        assert result[0]['execution_time_ms'] == .25
        assert result[0]['routing_timed_replay'] == 'PASS'
        assert benchmark_calls == [(0, 100, True), (0, 100, False)]
    else:
        with pytest.raises(AssertionError, match='Routing E8M0 scales'):
            h['arena_benchmark'](shapes=h['ALL_SHAPES'][:1], warmup=10, iters=100, verbose=False)
        assert benchmark_calls == [(0, 100, True)]
    assert calls.count('setup') == 11  # Original first compile launch +10 warmups.
    assert calls.count('measured') == 1
    assert calls.count('bound_replay') == 2  # Original sign control +new routing control.
    assert torch.equal(inputs_seen[-1]['x'], original['x'])
    assert torch.equal(inputs_seen[-1]['sorted_ids'], original['sorted_ids'])


def test_public_cpu_reference_controls_include_routing_known_answers(harness, monkeypatch):
    monkeypatch.syspath_prepend(str(TASK/'scripts'))
    monkeypatch.setitem(sys.modules, 'reference_support',
                        load(TASK/'scripts/reference_support.py', 'reference_support'))
    controls = load(TASK/'scripts/reference_controls.py', 'silu_reference_controls')
    results = controls.run()
    assert len(results) == 5
    assert all(row['known_answer'] == 'PASS' and row['negative_output'] == 'rejected' for row in results)
