"""CPU adversarial controls for task-owned gates; these are not GPU qualification."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def helpers(task):
    path = ROOT / 'tasks' / task / 'scripts/replay_checks.py'
    spec = importlib.util.spec_from_file_location('repair_control_' + task.replace('/', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(task, names, **globals_):
    tree = ast.parse((ROOT / 'tasks' / task / 'test_kernel_harness.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    ns = dict(torch=torch, **globals_)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'protected-task-functions', 'exec'), ns)
    return SimpleNamespace(**ns)


def timed(fn):
    state = SimpleNamespace(bound=True, outputs=fn(), calls=0)
    def replay():
        state.calls += 1
        result = fn()
        state.outputs.copy_(result)
        return state.outputs
    state.rerun = replay
    return state


def test_independent_attention_probes_reject_stale_qk_and_restore_inputs():
    h = helpers('flydsl2flydsl/flash_attn_func_kernel')
    q = torch.tensor([[1., -1.], [.5, 2.]])
    k = torch.tensor([[2., 1.], [-1., .25]])
    v = torch.tensor([[3., -2.], [.25, 1.]])
    original = tuple(x.clone() for x in (q, k, v))
    reference = lambda: (q @ k.T).softmax(-1) @ v
    cached = (q @ k.T).softmax(-1)
    current = timed(lambda: cached @ v)
    with pytest.raises(AssertionError, match='Attention max error'):
        h.verify_timed_run(current, inputs=(q, k, v), originals=original,
                           expected=reference(), perturb=h.scaled_input_probes(q, k, v),
                           reference=reference,
                           compare=lambda a, b: h.compare_output(a, b, .01))
    assert all(torch.equal(a, b) for a, b in zip((q, k, v), original))


def test_norm_probes_reject_eps_omission_without_tightening_original_gate():
    h = helpers('triton2flydsl/aiter/rmsnorm')
    x = torch.tensor([[1., -2., 3., -4.]])
    original = (x.clone(),)
    reference = lambda: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    current = timed(lambda: x * torch.rsqrt(x.square().mean(-1, keepdim=True)))
    # The normal input alone cannot distinguish this semantic omission.
    h.allclose_output(current.outputs, reference(), atol=.01, rtol=.01)
    with pytest.raises(AssertionError, match='Numerical mismatch'):
        h.verify_timed_run(current, inputs=(x,), originals=original,
                           expected=reference(), perturb=h.scaled_input_probes(x, factors=(.5, 1e-4)),
                           reference=reference,
                           compare=lambda a, b: h.allclose_output(a, b, atol=.01, rtol=.01))
    assert torch.equal(x, original[0])


def test_each_operand_probe_restores_prior_input_and_noop_is_rejected():
    h = helpers('triton2flydsl/aiter/fused_silu_mul')
    a, b = torch.tensor([2., 4.]), torch.tensor([3., 5.])
    originals = (a.clone(), b.clone())
    seen = []
    def reference():
        seen.append((a.clone(), b.clone()))
        return a * b
    current = timed(lambda: a * b)
    result = h.verify_timed_run(current, inputs=(a, b), originals=originals,
        expected=a*b, perturb=h.scaled_input_probes(a, b), reference=reference,
        compare=torch.testing.assert_close)
    assert result['replay_input_draws'] == 2
    assert torch.equal(seen[0][1], originals[1])
    assert torch.equal(seen[1][0], originals[0])
    with pytest.raises(AssertionError, match='did not change'):
        h.verify_timed_run(timed(lambda: a*b), inputs=(a, b), originals=originals,
            expected=a*b, perturb=lambda: None, reference=lambda: a*b, single_invocations=2,
            compare=torch.testing.assert_close)


@pytest.mark.parametrize('columns', [89999, 131072])
def test_wide_softmax_rejects_zero_and_uniform_probabilities(columns):
    h = helpers('triton2flydsl/aiter/softmax')
    f = functions('triton2flydsl/aiter/softmax', ['_compare_probabilities'],
                  allclose_output=h.allclose_output)
    x = torch.linspace(-3., 3., columns).reshape(1, columns)
    expected = x.softmax(-1).to(torch.bfloat16)
    f._compare_probabilities(expected.clone(), expected)
    for wrong in (torch.zeros_like(expected), torch.full_like(expected, 1 / columns)):
        # Demonstrate the prior absolute gate's acceptance, then reject it.
        assert torch.allclose(wrong, expected, atol=.01, rtol=.01)
        with pytest.raises(AssertionError, match='probability'):
            f._compare_probabilities(wrong, expected)


def test_dynamic_scale_zero_rejected_even_when_quantized_codes_are_right():
    names = ['_checked_quant_output', '_compare_token_outputs']
    f = functions('triton2flydsl/aiter/dynamic_quant_fp8', names)
    x = torch.tensor([[1., -2.]], dtype=torch.bfloat16)
    ref = torch.tensor([[63, -127]], dtype=torch.int8)
    ref_scale = torch.tensor([2 / 127])
    assert torch.allclose(torch.zeros(1), ref_scale, atol=.1, rtol=.1)
    with pytest.raises(AssertionError, match='quantizer values/scales'):
        f._compare_token_outputs((ref.clone(), torch.zeros(1)), (ref, ref_scale),
                                ref.clone(), x, torch.zeros(1), torch.int8)


def test_routing_checks_weights_of_ids_absent_from_reference_topk():
    f = functions('torch2flydsl/moe_topk_softmax_kernel', ['_require_weight_association'])
    gating = torch.linspace(-1., 1., 256).reshape(1, 256)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = gating.softmax(-1).gather(1, ids.long())
    f._require_weight_association(weights, ids, gating, 1.)
    for wrong in (weights * 1.25, torch.zeros_like(weights), torch.full_like(weights, 1e5)):
        with pytest.raises(AssertionError, match='selected expert'):
            f._require_weight_association(wrong, ids, gating, 1.)


def test_rope_mapped_replay_checks_and_restores_nonzero_untouched_cache():
    h = helpers('flydsl2flydsl/fused_rope_cache_kernel')
    inp = {name: torch.ones(2, 1, 2) for name in ('Q', 'K', 'V')}
    inp.update(positions=torch.arange(2), cos_cache=torch.ones(2, 2), sin_cache=torch.zeros(2, 2),
               slot_mapping=torch.arange(2), k_scale=torch.ones(1), v_scale=torch.ones(1), BS=2,
               key_cache=torch.zeros(3, 2, 1, 2), value_cache=torch.zeros(3, 2, 1, 2))
    def invoke(ignore_mapping=False, clear_state=False):
        slots = torch.arange(2) if ignore_mapping else inp['slot_mapping']
        if clear_state:
            inp['key_cache'].zero_(); inp['value_cache'].zero_()
        valid = slots >= 0
        inp['key_cache'][slots[valid] // 2, slots[valid] % 2] = inp['K'][valid]
        inp['value_cache'][slots[valid] // 2, slots[valid] % 2] = inp['V'][valid]
        return inp['Q'].clone(), inp['K'].clone(), inp['key_cache'], inp['value_cache']
    for bad in (False, True):
        inp['key_cache'].zero_(); inp['value_cache'].zero_()
        check = h.prepare_check(inp, lambda x, *args: x.clone(), .01, .01)
        state = SimpleNamespace(bound=True, outputs=invoke(bad, bad), rerun=lambda: invoke(bad, bad))
        if bad:
            with pytest.raises(AssertionError):
                h.verify_timed_run(state, **check)
        else:
            h.verify_timed_run(state, **check)
        assert torch.count_nonzero(inp['key_cache']) == 0
        assert torch.equal(inp['slot_mapping'], torch.arange(2))


def test_paged_attention_future_key_control_distinguishes_causal_mask():
    f = functions('flydsl2flydsl/pa_decode_fp8_kernel', ['reference_masked_attention'])
    q = torch.ones(4, 1, 128)
    k = torch.full((1027, 1, 128), .01)
    v = torch.full_like(k, .01)
    k[-1].fill_(4.); v[-1].fill_(1.)
    causal = f.reference_masked_attention(q, k, v, 128 ** -.5, torch.bfloat16, is_causal=True)
    wrong = f.reference_masked_attention(q, k, v, 128 ** -.5, torch.bfloat16, is_causal=False)
    assert (causal[:3].float() - .01).abs().max() < .005
    assert (wrong[:3].float() - causal[:3].float()).abs().min() > .9


def test_candidate_audit_does_not_advance_scored_preparation_stream():
    path = ROOT / 'tasks/triton2flydsl/aiter/gemm_a8w8/scripts/candidate_checks.py'
    spec = importlib.util.spec_from_file_location('gemm_audit_stream_control', path)
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    ns = {'__name__': 'flydsl.compiler.cpu_launch_fixture'}
    exec('class FakeCompiledKernel:\n def __call__(self, x): return x.clone()\n', ns)
    launch = ns['FakeCompiledKernel']()
    outcomes = []
    for candidate in (False, True):
        draws, current = [], [-1]
        def prepare():
            current[0] += 1
        def benchmark(fn, *, prepare_fn):
            for _ in range(3):
                prepare_fn(); fn(); draws.append(current[0])
            return draws
        h = SimpleNamespace(ARENA_FINAL_CANDIDATE=candidate, ENTRY='gemm_a8w8',
            _load_source=lambda: SimpleNamespace(gemm_a8w8=launch),
            benchmark_cuda_graph_or_events=benchmark)
        with policy.audit_candidate_calls(h, performance=True):
            fn = h._load_source().gemm_a8w8
            result = h.benchmark_cuda_graph_or_events(lambda: fn(torch.ones(2)), prepare_fn=prepare)
        outcomes.append(result)
    assert outcomes == [[0, 1, 2], [0, 1, 2]]


def test_sorting_actual_replay_controls_force_multi_block_expert_run():
    h = helpers('torch2flydsl/moe_sorting_kernel')
    ids = torch.stack((torch.arange(64) % 8, (torch.arange(64) + 1) % 8), dim=1).to(torch.int32)
    weights = torch.linspace(.1, .9, 128).reshape(64, 2)
    def reference(_model, shape, ids, weights):
        # A CPU stand-in returns complete current routing data, not a GPU result.
        return ids.clone(), weights.clone()
    def exact(shape, expected, actual, verbose=False):
        return (all(torch.equal(a, b) for a, b in zip(actual, expected)), 'routing changed')
    f = functions('torch2flydsl/moe_sorting_kernel', ['_sorting_replay_validator'],
                  _ref_outputs=reference, _exact_check=exact, verify_timed_pair=h.verify_timed_pair)
    validator = f._sorting_replay_validator(None, {'E': 8}, ids, weights)
    original = (ids.clone(), weights.clone())
    seen = []
    state = SimpleNamespace(bound=True, outputs=reference(None, {}, ids, weights))
    def replay():
        seen.append(ids.clone())
        result = reference(None, {}, ids, weights)
        for out, value in zip(state.outputs, result): out.copy_(value)
        return state.outputs
    state.rerun = replay
    def isolated(*, before_replay=None):
        if before_replay is not None: before_replay(state.outputs)
        return replay()  # CPU stand-in is one logical call.
    state.single_invocation = isolated
    validator(state)
    assert torch.equal(ids, original[0]) and torch.equal(weights, original[1])
    concentrated = [draw for draw in seen if torch.equal(draw, draw[:1].expand_as(draw))]
    assert concentrated and (concentrated[0] == concentrated[0][0, 0]).sum() == 64
    assert any((draw == 0).sum() == 33 for draw in seen)  # partially filled second block


@pytest.mark.parametrize('dtype', ['fp8', 'fp16', 'bf16'])
def test_preshuffle_compile_forwards_actual_input_dtype(monkeypatch, dtype):
    f = functions('flydsl2flydsl/preshuffle_gemm_v2_kernel', ['_compile_and_run_once'],
                  _kernel_args=lambda *args: args)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: 'cpu-stream-fixture')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    received = []
    mod = SimpleNamespace(compile_preshuffle_gemm_v2=lambda **kwargs: received.append(kwargs) or 'kernel')
    compiler = SimpleNamespace(compile=lambda *args: 'compiled')
    a = SimpleNamespace(shape=(256, 256))
    f._compile_and_run_once(mod, compiler, None, a, None, None, None,
                            256, 256, (128, 128, 64), in_dtype=dtype)
    assert received[0]['in_dtype'] == dtype


@pytest.mark.parametrize('behavior', ['correct', 'partial', 'mutate'])
def test_routing_correctness_checks_each_call_before_repeated_timing(monkeypatch, behavior):
    h = helpers('flydsl2flydsl/topk_gating_softmax_kernel')
    for name in ('rand', 'empty', 'arange'):
        factory = getattr(torch, name)
        def cpu(*args, _factory=factory, **kwargs):
            if 'device' in kwargs: kwargs['device'] = 'cpu'
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    f = functions('flydsl2flydsl/topk_gating_softmax_kernel', ['run_test', 'reference_topk'],
                  DTYPE_FP32=torch.float32, WARMUP_ITERS=10, BENCH_ITERS=100,
                  _torch_dtype=lambda dtype: torch.float32, os=SimpleNamespace(environ={}),
                  require_topk_output=h.require_topk_output, require_unchanged=h.require_unchanged)
    calls, benchmarks = [0], [0]
    def build(**kwargs):
        def launch(gating, weights, ids, tokens, count, stream=None):
            _, ref_w, ref_id, ref_token = f.reference_topk(gating, 2, True)
            rows = slice(calls[0] % 2 * 2, calls[0] % 2 * 2 + 2) if behavior == 'partial' else slice(None)
            calls[0] += 1
            for out, ref in zip((weights, ids, tokens), (ref_w, ref_id, ref_token)):
                out[rows].copy_(ref[rows])
            if behavior == 'mutate': gating[:, 0].add_(1)
        return launch
    def benchmark(fn, **kwargs):
        benchmarks[0] += 1
        fn(); fn()  # A repeated timing could heal the partial writer's buffer.
        return None, 1.
    namespace = f.run_test.__globals__
    namespace.update(build_topk_gating_softmax_module=build, run_perftest=benchmark)
    if behavior != 'correct':
        with pytest.raises(AssertionError): f.run_test(4, 8, 2, 'f32')
        assert calls[0] == 1 and benchmarks[0] == 0
    else:
        assert f.run_test(4, 8, 2, 'f32')[0] is True
        assert benchmarks[0] == 1


@pytest.mark.parametrize('partial_writer', [False, True])
def test_actual_scored_routing_driver_also_checks_isolated_graph(monkeypatch, tmp_path, partial_writer):
    import os
    from src.tools.perf.aka_benchmark import TimedRun
    h = helpers('flydsl2flydsl/topk_gating_softmax_kernel')
    for name in ('rand', 'empty', 'arange'):
        factory = getattr(torch, name)
        def cpu(*args, _factory=factory, **kwargs):
            if 'device' in kwargs: kwargs['device'] = 'cpu'
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    f = functions('flydsl2flydsl/topk_gating_softmax_kernel', ['reference_topk', 'arena_benchmark'],
        DTYPE_FP32=torch.float32, _torch_dtype=lambda _: torch.bfloat16,
        run_test=lambda *a, **kw: (True, None), TimedRun=TimedRun,
        verify_topk_timed_run=h.verify_topk_timed_run, require_topk_output=h.require_topk_output,
        _os=os, _THIS=str(tmp_path))
    def build(**options):
        calls = [0]
        def launch(gating, weights, ids, tokens, count, stream=None):
            _, w, i, t = f.reference_topk(gating.float(), options['topk'], True)
            half = count // 2
            selected = slice((calls[0] % 2) * half, (calls[0] % 2 + 1) * half) if partial_writer else slice(None)
            calls[0] += 1
            for out, ref in zip((weights, ids, tokens), (w, i, t)):
                out[selected].copy_(ref[selected])
        return launch
    observed, isolated_calls = [], []
    def benchmark(fn, *, repetition, timed_run, prepare_fn=None, **kwargs):
        one_call = prepare_fn is not None
        observed.append(one_call)
        def replay():
            if prepare_fn is not None: prepare_fn()
            result = fn()
            if not one_call: result = fn()  # Repeated calls can heal partial writes.
            return result
        for _ in range(repetition):
            if prepare_fn is not None: prepare_fn()
            if callable(timed_run.before_sample): timed_run.before_sample(1 if one_call else 2)
            output = fn()
            if not one_call: output = fn()
            if callable(timed_run.after_sample): timed_run.after_sample(output)
        timed_run._bind(replay, output)
        def isolated(before_replay):
            isolated_calls.append(True)
            if prepare_fn is not None: prepare_fn()
            if before_replay is not None: before_replay(timed_run.outputs)
            return fn()
        timed_run._single_invocation = isolated
        return .125, {'benchmark_method': 'cuda_graph', 'benchmark_samples': repetition}
    f.arena_benchmark.__globals__.update(build_topk_gating_softmax_module=build,
                                         benchmark_cuda_graph_or_events=benchmark)
    if partial_writer:
        with pytest.raises(AssertionError): f.arena_benchmark(warmup=0, iters=2)
        assert observed == [False] and len(isolated_calls) == 1
    else:
        rows = f.arena_benchmark(warmup=0, iters=2)
        assert observed == [False, False] and len(isolated_calls) >= 4
        assert all(row['benchmark_samples'] == 2 for row in rows)
        assert all(row['single_invocations_per_probe'] == 2 for row in rows)


def test_single_call_gate_rejects_alternating_compute_that_passes_grouped_graph():
    h = helpers('flydsl2flydsl/flash_attn_func_kernel')
    x = torch.tensor([[1., -2.]])
    original = (x.clone(),)
    output = x * 3
    calls = [0]
    def one(*, before_replay=None):
        if before_replay is not None: before_replay(output)
        calls[0] += 1
        if calls[0] % 2 == 0: output.copy_(x * 3)
        return output
    def grouped():
        one(); return one()
    for audit in (0, 2):
        x.copy_(original[0]); output.copy_(x * 3); calls[0] = 0
        state = SimpleNamespace(bound=True, outputs=output, rerun=grouped, single_invocation=one)
        check = lambda: h.verify_timed_run(state, inputs=(x,), originals=original,
            expected=x*3, perturb=lambda: x.mul_(.5), reference=lambda: x*3,
            compare=torch.testing.assert_close, single_invocations=audit)
        if audit:
            with pytest.raises(AssertionError): check()
        else:
            assert check()['replay_correctness'] == 'PASS'
        assert torch.equal(x, original[0])


def test_changed_input_is_checked_before_grouped_replay_can_refresh_stale_state():
    h = helpers('flydsl2flydsl/flash_attn_func_kernel')
    x = torch.tensor([[1., -2.]])
    original = (x.clone(),)
    output, cached = x * 3, x * 3
    def delayed(*, before_replay=None):
        if before_replay is not None: before_replay(output)
        result = cached.clone()
        cached.copy_(x * 3)
        return result
    def grouped():
        delayed()
        output.copy_(delayed())
        return output
    for audit in (0, 2):
        x.copy_(original[0]); output.copy_(x * 3); cached.copy_(x * 3)
        state = SimpleNamespace(bound=True, outputs=output, rerun=grouped, single_invocation=delayed)
        def check():
            return h.verify_timed_run(state, inputs=(x,), originals=original,
                expected=x*3, perturb=lambda: x.mul_(.5), reference=lambda: x*3,
                compare=torch.testing.assert_close, single_invocations=audit)
        if audit:
            with pytest.raises(AssertionError): check()
        else:
            assert check()['replay_correctness'] == 'PASS'
        assert torch.equal(x, original[0])


def test_nonuniform_probe_rejects_cached_quantized_weights_with_fresh_scales():
    h = helpers('torch2flydsl/gemm_a8w8_blockscale_kernel')
    a = torch.tensor([[1., 2., 3., 4.]])
    w = torch.tensor([[1., 2., 3., 4.], [-4., -3., -2., -1.]])
    original = (a.clone(), w.clone())
    def code(value):
        return (value / value.abs().amax(-1, keepdim=True) * 127).round()
    cached_codes = code(w)
    def dequant(codes, value):
        return codes * (value.abs().amax(-1, keepdim=True) / 127)
    reference = lambda: a @ dequant(code(w), w).T
    wrong = lambda: a @ dequant(cached_codes, w).T
    for audit in (0, 2):
        for value, old in zip((a, w), original): value.copy_(old)
        state = timed(wrong)
        def isolated(*, before_replay=None):
            if before_replay is not None: before_replay(state.outputs)
            return wrong()
        state.single_invocation = isolated
        def check():
            return h.verify_timed_run(state, inputs=(a, w), originals=original,
                expected=reference(), perturb=lambda: (a.neg_(), w.mul_(.5)),
                reference=reference, compare=torch.testing.assert_close, single_invocations=audit)
        if audit:
            with pytest.raises(AssertionError): check()
        else:
            assert check()['replay_correctness'] == 'PASS'
        assert all(torch.equal(value, old) for value, old in zip((a, w), original))


def test_mean_shift_distinguishes_variance_from_second_moment_at_original_tolerance():
    h = helpers('flydsl2flydsl/layernorm_kernel')
    x = torch.tensor([[-1., 1., -1., 1.]])
    original = (x.clone(),)
    centered = lambda: x - x.mean(-1, keepdim=True)
    reference = lambda: centered() / (centered().square().mean(-1, keepdim=True) + 1e-5).sqrt()
    wrong = lambda: centered() / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    state = timed(wrong)
    torch.testing.assert_close(state.outputs, reference(), atol=.01, rtol=.01)
    with pytest.raises(AssertionError):
        h.verify_timed_run(state, inputs=(x,), originals=original, expected=reference(),
            perturb=lambda: x.add_(1), reference=reference,
            compare=lambda actual, wanted: torch.testing.assert_close(actual, wanted, atol=.01, rtol=.01))
    assert torch.equal(x, original[0])


def test_opted_in_controllers_check_isolated_call_before_grouped_state_can_heal_it():
    files = [p for p in (ROOT / 'tasks').rglob('scripts/replay_checks.py')
             if 'def check_single_invocations(' in p.read_text()]
    assert len(files) == 39
    for path in files:
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith('verify_'): continue
            single = [node.lineno for node in ast.walk(fn) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'check_single_invocations']
            rerun = [node.lineno for node in ast.walk(fn) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'rerun']
            if single and rerun: assert max(single) < min(rerun), (path, fn.name)


def test_finite_high_magnitude_norm_input_detects_half_precision_squaring():
    x = torch.full((2, 8), 1024., dtype=torch.float16)
    reference = (x.float() / (x.float().square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(x.dtype)
    wrong = (x.float() / (x.square().float().mean(-1, keepdim=True) + 1e-5).sqrt()).to(x.dtype)
    assert torch.isfinite(x).all() and torch.equal(reference, torch.ones_like(x))
    with pytest.raises(AssertionError): torch.testing.assert_close(wrong, reference, atol=.01, rtol=.01)


def test_nonuniform_control_preserves_broadcast_scale_layout():
    h = helpers('flydsl2flydsl/pa_decode_fp8_kernel')
    storage = torch.tensor([2., 4., 6.])
    scale = storage[:, None].expand(3, 8)
    before = scale.clone()
    probe, = h.nonuniform_input_probes(scale)
    probe()
    assert scale.stride() == (1, 0) and torch.equal(storage, torch.tensor([1., 4., 3.]))
    h._restore_readonly_tensor(scale, before)
    assert torch.equal(scale, before)


def test_preshuffle_probe_changes_real_packed_operand_and_reference_together():
    h = helpers('flydsl2flydsl/preshuffle_gemm_v2_kernel')
    f = functions('flydsl2flydsl/preshuffle_gemm_v2_kernel', ['_logical_b_from_packed'])
    a = torch.arange(128).float().reshape(2, 64) / 100
    logical = torch.arange(1024).float().remainder(17).reshape(16, 64) / 10
    packed = logical.reshape(1, 16, 1, 4, 16).permute(0, 2, 3, 1, 4).contiguous()
    torch.testing.assert_close(f._logical_b_from_packed(packed, 16, 64), logical)
    inputs = (a, logical, packed)
    originals = tuple(x.clone() for x in inputs)
    actual = lambda: a @ f._logical_b_from_packed(packed, 16, 64).T
    state = timed(actual)
    def isolated(*, before_replay=None):
        if before_replay is not None: before_replay(state.outputs)
        return actual()
    state.single_invocation = isolated
    # logical is an independent frozen oracle control, not an operator argument.
    result = h.verify_timed_run(state, inputs=inputs, originals=originals,
        expected=a @ logical.T, perturb=lambda: a.neg_(), reference=actual,
        compare=torch.testing.assert_close, single_invocations=2, probe_inputs=(a, packed))
    assert result['replay_correctness'] == 'PASS'
    assert all(torch.equal(x, old) for x, old in zip(inputs, originals))


@pytest.mark.parametrize('correct', [False, True])
def test_event_samples_cannot_hide_99_wrong_outputs_before_a_correct_last_one(monkeypatch, correct):
    from contextlib import nullcontext
    from src.tools.perf import aka_benchmark as benchmark
    h = helpers('torch2flydsl/gemm_a8w8_blockscale_kernel')
    class Event:
        def record(self): pass
        def synchronize(self): pass
        def elapsed_time(self, other): return 1.
    stream = SimpleNamespace(wait_stream=lambda other: None)
    cuda = SimpleNamespace(is_available=lambda: True, Event=lambda **kw: Event(),
        current_stream=lambda: stream, stream=lambda value: nullcontext(), synchronize=lambda: None)
    monkeypatch.setattr(benchmark, 'torch', SimpleNamespace(cuda=cuda))
    x = torch.tensor([1., 2.]); expected = x * 3; calls = [0]
    def invoke():
        calls[0] += 1
        return expected.clone() if correct or calls[0] == 100 else torch.zeros_like(expected)
    timed_run = benchmark.TimedRun()
    timed_run.after_sample = h.sample_output_validator(inputs=(x,), originals=(x.clone(),),
        expected=expected, compare=torch.testing.assert_close)
    if correct:
        samples = benchmark.benchmark_cuda_event_samples(invoke, repetition=100, timed_run=timed_run)
        assert len(samples) == calls[0] == timed_run.after_sample.count == 100
    else:
        with pytest.raises(AssertionError):
            benchmark.benchmark_cuda_event_samples(invoke, repetition=100, timed_run=timed_run)
        assert calls[0] == 1 and not timed_run.bound


@pytest.mark.parametrize('task', ['batched_gemm_bf16_kernel', 'gemm_a8w8_kernel',
                                 'gemm_a8w8_blockscale_kernel', 'gemm_a8wfp4_kernel'])
def test_both_event_entrypoints_bind_task_numerical_sample_callback(task):
    p = ROOT / 'tasks/torch2flydsl' / task / 'test_kernel_harness.py'
    tree = ast.parse(p.read_text())
    for name in ('run_benchmark', 'arena_benchmark'):
        fn = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == name)
        bindings = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == 'after_sample' for t in n.targets)]
        assert len(bindings) == 1
        source = ast.get_source_segment(p.read_text(), bindings[0])
        assert 'sample_output_validator(' in source or 'replay_validate.sample_output' in source


@pytest.mark.parametrize('mode', ['static', 'dyn_tensor', 'dyn_token'])
@pytest.mark.parametrize('correct', [False, True])
def test_all_quantization_entrypoints_reject_stale_same_address_codes(mode, correct):
    h = helpers('triton2flydsl/aiter/dynamic_quant_fp8')
    f = functions('triton2flydsl/aiter/dynamic_quant_fp8',
        ['_reference_quant', '_checked_quant_output', '_check_repeated_quant'],
        _dtype_max=lambda dtype: 127., nonuniform_input_probes=h.nonuniform_input_probes,
        require_unchanged=h.require_unchanged)
    x = torch.linspace(-4., 3., 64).reshape(2, 32).to(torch.bfloat16)
    original = x.clone()
    scale = torch.tensor([.125]) if mode == 'static' else torch.zeros(1 if mode == 'dyn_tensor' else 2)
    cached, _ = f._reference_quant(x, torch.int8, mode, scale if mode == 'static' else None)
    qx = cached.clone()
    def invoke(out, current, scales):
        expected, scale_ref = f._reference_quant(current, torch.int8, mode, scales if mode == 'static' else None)
        out.copy_(expected if correct else cached)
        if mode == 'static': return out
        scales.copy_(scale_ref.reshape_as(scales))
        return out, scales
    mod = SimpleNamespace(static_per_tensor_quant_fp8_i8=invoke,
        dynamic_per_tensor_quant_fp8_i8=invoke, dynamic_per_token_quant_fp8_i8=invoke)
    if correct: f._check_repeated_quant(mod, mode, x, torch.int8, qx, scale)
    else:
        with pytest.raises(AssertionError): f._check_repeated_quant(mod, mode, x, torch.int8, qx, scale)
    assert torch.equal(x, original)


@pytest.mark.parametrize('behavior', ['correct', 'ignore_out', 'stale_fp16', 'stale_out'])
def test_silu_original_dtypes_reuse_input_and_honor_supplied_output(monkeypatch, behavior):
    h = helpers('triton2flydsl/aiter/fused_silu_mul')
    cached = {}
    def reference(x):
        gate, up = x.float().chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)
    def invoke(x, out=None):
        result = reference(x)
        if behavior == 'stale_fp16' and x.dtype == torch.float16:
            result = cached.setdefault(x.dtype, result.clone())
        if out is None:
            cached['default'] = result.clone()
            return result
        if behavior == 'ignore_out': return result
        if behavior == 'stale_out': result = cached['default']
        out.copy_(result)
        return out
    f = functions('triton2flydsl/aiter/fused_silu_mul', ['run_correctness', '_checked_elementwise_output'],
        _load_source=lambda: SimpleNamespace(fused_silu_mul=invoke),
        TEST_SHAPES=[{'name': 'small', 'rows': 2, 'last': 8}], DTYPES=['bf16', 'fp16'],
        _torch_dtype=lambda name: torch.bfloat16 if name == 'bf16' else torch.float16,
        _make_inputs=lambda rows, last, dtype: torch.linspace(-3., 2., rows*last).reshape(rows, last).to(dtype),
        _torch_silu_mul=reference, nonuniform_input_probes=h.nonuniform_input_probes,
        require_unchanged=h.require_unchanged)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    assert f.run_correctness(verbose=False) is (behavior == 'correct')


@pytest.mark.parametrize('behavior', ['correct', 'stale_fp16', 'omit_eps_fp16'])
def test_rmsnorm_correctness_only_fp16_also_checks_repeat_and_eps_transition(monkeypatch, behavior):
    h = helpers('triton2flydsl/aiter/rmsnorm')
    cached = {}
    def invoke(x, weight, eps):
        denominator = x.float().square().mean(-1, keepdim=True)
        if not (behavior == 'omit_eps_fp16' and x.dtype == torch.float16): denominator = denominator + eps
        out = (x.float() * denominator.rsqrt() * weight.float()).to(x.dtype)
        if behavior == 'stale_fp16' and x.dtype == torch.float16:
            return cached.setdefault(x.dtype, out.clone())
        return out
    f = functions('triton2flydsl/aiter/rmsnorm', ['run_correctness', '_torch_rmsnorm', '_checked_elementwise_output'],
        _load_source=lambda: SimpleNamespace(rms_norm=invoke), EPS=1e-5,
        TEST_SHAPES=[{'name': 'small', 'M': 2, 'N': 8}], DTYPES=['bf16', 'fp16'],
        _torch_dtype=lambda name: torch.bfloat16 if name == 'bf16' else torch.float16,
        _make_inputs=lambda m, n, dtype: (torch.linspace(-2., 2., m*n).reshape(m, n).to(dtype), torch.ones(n, dtype=dtype)),
        nonuniform_input_probes=h.nonuniform_input_probes, check_reused_calls=h.check_reused_calls,
        require_unchanged=h.require_unchanged, allclose_output=h.allclose_output)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    assert f.run_correctness(verbose=False) is (behavior == 'correct')


@pytest.mark.parametrize('defect', ['none', 'stride', 'fp16'])
def test_mxfp8_actual_domain_controls_check_strides_and_fp16(monkeypatch, defect):
    h = helpers('triton2flydsl/aiter/dynamic_mxfp8_quant')
    f = functions('triton2flydsl/aiter/dynamic_mxfp8_quant',
        ['run_domain_controls', '_mx_reference', '_compare_mx_pair', '_checked_mx_pair', '_torch_mxfp8_quant_from_fp32'],
        QUANT_BLOCK_SIZE=32, _E8M0_MASK_INT32=-8388608, require_unchanged=h.require_unchanged)
    for name in ('linspace', 'full', 'zeros'):
        factory = getattr(torch, name)
        def cpu(*args, _factory=factory, **kwargs):
            if 'device' in kwargs: kwargs['device'] = 'cpu'
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    seen = []
    def invoke(x, scale=None):
        seen.append((x.stride(-1), x.dtype))
        if defect == 'stride' and not x.is_contiguous(): raise AssertionError('contiguous-only shortcut')
        values, expected_scale = f._mx_reference(x)
        if defect == 'fp16' and x.dtype == torch.float16:
            values.view(torch.uint8).zero_()
        if scale is not None:
            scale.copy_(expected_scale.reshape_as(scale))
            expected_scale = scale.reshape_as(expected_scale)
        return values, expected_scale
    if defect != 'none':
        with pytest.raises(AssertionError): f.run_domain_controls(SimpleNamespace(dynamic_mxfp8_quant=invoke))
    else:
        f.run_domain_controls(SimpleNamespace(dynamic_mxfp8_quant=invoke))
        assert (2, torch.float16) in seen
    assert any(stride == 2 for stride, dtype in seen)


@pytest.mark.parametrize('ignore_scale', [False, True])
def test_prefill_actual_correctness_controls_check_explicit_softmax_scale(monkeypatch, ignore_scale):
    h = helpers('triton2flydsl/sglang/prefill_attention')
    cfg = {'seqs': [4], 'head': 2, 'kv_head': 1, 'd': 4, 'causal': False}
    torch.manual_seed(17)
    q = (torch.randn(4, 2, 4) * 2).to(torch.bfloat16)
    k = (torch.randn(4, 1, 4) * 2).to(torch.bfloat16)
    v = torch.randn(4, 1, 4).to(torch.bfloat16)
    out = torch.empty_like(q)
    bsl, bseq = torch.tensor([0], dtype=torch.int32), torch.tensor([4], dtype=torch.int32)
    source = functions('triton2flydsl/sglang/prefill_attention',
        ['run_correctness', 'reference', '_checked_attention_output', '_compare_attention_output'],
        TEST_SHAPES=[cfg], _shape_of=lambda value: value,
        make_inputs=lambda *a: (q, k, v, out, bsl, bseq, 4),
        _retry_oom=lambda fn: fn(), require_unchanged=h.require_unchanged,
        require_tensor_contract=h.require_tensor_contract)
    scales = []
    def invoke(q, k, v, out, bsl, bseq, length, *, is_causal, sm_scale=None):
        scales.append(sm_scale)
        wanted = source.reference(q, k, v, cfg, sm_scale=None if ignore_scale else sm_scale)
        out.copy_(wanted.to(out.dtype))
    source.run_correctness.__globals__['load_module'] = lambda: SimpleNamespace(context_attention_fwd=invoke)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    assert source.run_correctness()[0] is (not ignore_scale)
    assert any(scale is not None for scale in scales)


def test_gdn_replay_includes_unique_nonconsecutive_state_indices():
    h = helpers('triton2flydsl/sglang/gdn_chunk_fwd_h')
    received = {}
    source = functions('triton2flydsl/sglang/gdn_chunk_fwd_h', ['_recurrent_replay_validator'],
        reference_h=lambda inp: (inp['idx'].clone(), inp['w'].clone(), inp['g'].clone()),
        _compare_recurrent_output=lambda *a: None,
        verify_timed_pair=lambda timed, **kwargs: received.update(kwargs))
    inp = dict(B=2, pool=16, k=torch.ones(2, 2), w=torch.ones(2, 2),
               u=torch.ones(2, 2), g=-torch.ones(2, 2), idx=torch.arange(2, dtype=torch.int32),
               init=torch.zeros(16, 2))
    validator = source._recurrent_replay_validator(inp, inp['init'].clone())
    validator(None)
    seen = []
    for change in received['perturb']:
        for value, original in zip(received['inputs'], received['originals']): value.copy_(original)
        change(); seen.append(inp['idx'].clone())
    selected = next(draw for draw in seen if draw.tolist() == [1, 3])
    assert selected.unique().numel() == 2 and bool(((selected >= 0) & (selected < 16)).all())
    wrong = selected[0] - torch.arange(2)
    assert not torch.equal(selected, wrong)


def test_constant_value_attention_probe_rejects_missing_tail_at_original_fp4_gate():
    h = helpers('triton2flydsl/aiter/fav3_sage_mxfp4')
    received = {}
    source = functions('triton2flydsl/aiter/fav3_sage_mxfp4',
        ['_sage_replay_validator', '_attention_reference', '_checked_sage_output', '_compare_sage_output',
         '_compare', '_fp8_frac_exceeding'],
        scaled_input_probes=h.scaled_input_probes, require_tensor_contract=h.require_tensor_contract,
        NORM_MAX_ERR_TOL=.25, MAX_DIFF_PCT=1.5, ATOL_FP8=.3, RTOL_FP8=.25,
        verify_timed_run=lambda timed, **kwargs: received.update(kwargs))
    q, k, v = (torch.randn(1, 384, 4, 128, dtype=torch.bfloat16) for _ in range(3))
    source._sage_replay_validator(q, k, v, 128 ** -.5, True)(None)
    received['perturb'][0]()
    wanted = received['reference']()
    torch.testing.assert_close(wanted, torch.ones_like(wanted))
    complete = wanted.to(torch.bfloat16)
    source._compare_sage_output(complete, wanted)
    missing = complete.clone(); missing[:, 256:] = 0
    with pytest.raises(AssertionError, match='original SageAttention'):
        source._compare_sage_output(missing, wanted)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_preshuffle_actual_public_dtype_controls_reject_transposed_output(monkeypatch, dtype):
    # Intercept the actual protected control's launch; GPU execution is covered
    # separately. The control must reject an incorrect output transpose.
    selected = 'bf16' if dtype == torch.bfloat16 else 'fp16'
    source = functions('flydsl2flydsl/preshuffle_gemm_v2_kernel', ['run_domain_controls'],
        _load_kernel=lambda *a: object(), _CANDIDATE_DIR='unused',
        require_unchanged=helpers('flydsl2flydsl/preshuffle_gemm_v2_kernel').require_unchanged,
        ATOL=.02, RTOL=.02)
    import sys
    monkeypatch.setitem(sys.modules, 'flydsl', SimpleNamespace(compiler=object()))
    monkeypatch.setitem(sys.modules, 'flydsl.compiler', object())
    for name in ('linspace', 'arange', 'full'):
        factory = getattr(torch, name)
        def cpu(*args, _factory=factory, **kwargs):
            if 'device' in kwargs: kwargs['device'] = 'cpu'
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    seen = []
    def launch(mod, compiler, out, a, packed, sa, sb, m, n, tile, in_dtype='fp8'):
        b = packed.permute(0, 3, 1, 2, 4).reshape(n, a.shape[1])
        correct = (a.float() @ b.float().T).to(out.dtype)
        out.copy_(correct.T if in_dtype == selected else correct)
        seen.append(in_dtype)
    source.run_domain_controls.__globals__['_compile_and_run_once'] = launch
    with pytest.raises(AssertionError): source.run_domain_controls()
    assert selected in seen


def test_fresh_captures_cannot_prime_the_scored_graph_with_changed_operands():
    h = helpers('torch2flydsl/fmoe_fp8_blockscale_g1u1_kernel')
    x = torch.tensor([[1., -2.]])
    original = (x.clone(),)
    cached = x * 3
    output = cached.clone()
    def capture(*, before_replay=None):
        cached.copy_(x * 3)
        if before_replay is not None:
            before_replay(output)
        output.copy_(cached)
        return output
    def scored_graph():
        # Wrong: its required input-dependent arithmetic ran at capture only.
        output.copy_(cached)
        return output
    state = SimpleNamespace(bound=True, outputs=output,
                            rerun=scored_graph, single_invocation=capture)
    with pytest.raises(AssertionError):
        h.verify_timed_run(state, inputs=(x,), originals=original, expected=x*3,
            perturb=lambda: x.mul_(.5), reference=lambda: x*3,
            compare=torch.testing.assert_close, single_invocations=2)
    assert torch.equal(x, original[0])


def test_activation_actual_probe_rejects_premature_fp16_input_narrowing():
    h = helpers('torch2flydsl/gelu_tanh_and_mul_kernel')
    f = functions('torch2flydsl/gelu_tanh_and_mul_kernel', ['_activation_replay_validator', '_checked_activation_output'],
                  require_unchanged=h.require_unchanged, verify_timed_run=h.verify_timed_run,
                  require_tensor_contract=h.require_tensor_contract,
                  scaled_input_probes=h.scaled_input_probes,
                  normalized_output=h.normalized_output, REL_TOL=.01)
    x = torch.tensor([[1., -2., .5, 1.]], dtype=torch.bfloat16)
    reference = lambda z: (torch.nn.functional.gelu(z[..., :2].float(), approximate='tanh') * z[..., 2:].float()).to(z.dtype)
    current = timed(lambda: reference(x.to(torch.float16).to(torch.bfloat16)))
    def isolated(*, before_replay=None):
        if before_replay is not None: before_replay(current.outputs)
        current.outputs.copy_(reference(x.to(torch.float16).to(torch.bfloat16)))
        return current.outputs
    current.single_invocation = isolated
    with pytest.raises(AssertionError, match='Non-finite'):
        f._activation_replay_validator(x, reference)(current)


def test_static_quantizer_must_read_changed_scale_at_the_same_address():
    h = helpers('triton2flydsl/aiter/dynamic_quant_fp8')
    f = functions('triton2flydsl/aiter/dynamic_quant_fp8',
        ['_reference_quant', '_checked_quant_output', '_check_repeated_quant'],
        _dtype_max=lambda dtype: 127., nonuniform_input_probes=h.nonuniform_input_probes,
        require_unchanged=h.require_unchanged)
    x = torch.linspace(-4., 3., 64).reshape(2, 32).to(torch.bfloat16)
    scale = torch.tensor([.125])
    qx = torch.empty_like(x, dtype=torch.int8)
    captured_scale = scale.clone()
    def stale_scale(out, current, actual_scale):
        out.copy_(f._reference_quant(current, torch.int8, 'static', captured_scale)[0])
        return out
    with pytest.raises(AssertionError):
        f._check_repeated_quant(SimpleNamespace(static_per_tensor_quant_fp8_i8=stale_scale),
                                'static', x, torch.int8, qx, scale)
    torch.testing.assert_close(scale, captured_scale)


def test_actual_unified_softcap_remains_finite_for_large_finite_scores():
    path = ROOT / 'tasks/triton2flydsl/aiter/unified_attention/unified_attention.py'
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'apply_softcap')
    fn.decorator_list = []
    ns = {'tl': SimpleNamespace(math=SimpleNamespace(exp2=torch.exp2), abs=torch.abs, where=torch.where)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    score = torch.tensor([-1e6, -1000., -1., 0., 1., 1000., 1e6])
    actual = ns['apply_softcap'](score, 10.)
    expected = 10 * torch.tanh(score / 10 * __import__('math').log(2))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('task', ['mla', 'fav3_sage_mxfp4'])
def test_failed_performance_adapter_preserves_original_per_case_evidence(task, tmp_path):
    path = ROOT / 'tasks/triton2flydsl/aiter' / task / 'scripts/task_actions.py'
    spec = importlib.util.spec_from_file_location('raw_case_evidence_' + task, path)
    actions = importlib.util.module_from_spec(spec); spec.loader.exec_module(actions)
    records = [{'test_case_id': 'perf1', 'execution_time_ms': 1.25,
                'benchmark_method': 'cuda_graph'},
               {'test_case_id': 'perf2', 'execution_time_ms': -1.,
                'benchmark_fallback_reason': 'Numerical mismatch: boundary counterexample'}]
    harness = SimpleNamespace(__file__=str(tmp_path/'test_kernel_harness.py'), run_performance=lambda: records)
    with pytest.raises(RuntimeError, match='perf2: Numerical mismatch: boundary counterexample'):
        actions.performance(harness)
    saved = __import__('json').loads((tmp_path/'build/raw_performance_cases.json').read_text())
    assert saved == records
    harness.run_performance = lambda: records[:1]
    assert actions.performance(harness) == records[:1]


def test_paged_attention_restores_broadcast_scale_views_without_changing_layout():
    h = helpers('flydsl2flydsl/pa_decode_fp8_kernel')
    storage = torch.tensor([2.])
    scale = storage.expand(2, 8)
    original = scale.clone()
    stride, pointer = scale.stride(), scale.data_ptr()
    h.restore_input(scale, original)
    storage.fill_(4.)
    h.restore_input(scale, original)
    assert torch.equal(scale, original)
    assert scale.stride() == stride and scale.data_ptr() == pointer


def test_mla_oracle_retains_fp32_scores_instead_of_rounding_them_to_bf16():
    f = functions('triton2flydsl/aiter/mla', ['_ref_masked_attention'])
    gen = torch.Generator().manual_seed(44)
    q = torch.randn(1, 2, 192, generator=gen).to(torch.bfloat16)
    k = torch.randn(512, 1, 192, generator=gen).to(torch.bfloat16)
    v = k[..., :128]
    scale = 192 ** -.5
    actual = f._ref_masked_attention(q, k, v, scale)
    expanded_k = k.repeat_interleave(2, 1)
    expanded_v = v.repeat_interleave(2, 1)
    score64 = torch.einsum('qhd,khd->hqk', q.double(), expanded_k.double()) * scale
    probabilities = score64.softmax(-1).to(q.dtype)
    expected = torch.einsum('hqk,khd->qhd', probabilities.double(), expanded_v.double()).to(q.dtype)
    torch.testing.assert_close(actual, expected, atol=.004, rtol=.01)
    # The specification's QK scores precede a softmax and remain FP32. Preserve
    # that distinction independently of the later BF16 probability/output casts.
    score32 = torch.einsum('qhd,khd->hqk', q.float(), expanded_k.float())
    premature = torch.einsum('qhd,khd->hqk', q, expanded_k).float()
    assert not torch.equal(score32, premature)


@pytest.mark.parametrize('task,scales', [('flydsl2flydsl/fp8_gemm_4wave_kernel',('A_scale','B_scale')),
                                        ('flydsl2flydsl/preshuffle_gemm_v2_kernel',('scale_a','scale_b'))])
def test_actual_gemm_performance_callbacks_support_multiple_probes_and_bound_names(task, scales):
    h = helpers(task)
    path = ROOT/'tasks'/task/'test_kernel_harness.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='arena_benchmark')
    call = next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_timed_run')
    expression = next(k.value for k in call.keywords if k.arg=='perturb')
    a = torch.arange(1.,9.).reshape(2,4)
    b = torch.arange(1.,13.).reshape(3,4)
    sa,sb = torch.tensor([.5,1.]),torch.tensor([1.,.5,.25])
    inputs=(a,b,sa,sb);originals=tuple(x.clone() for x in inputs)
    ns=dict(A=a, **{('B_T' if scales[0]=='A_scale' else 'B'):b}, **dict(zip(scales,(sa,sb))))
    probes=eval(compile(ast.Expression(expression),'actual-gemm-probes','eval'),ns)
    reference=lambda: (a@b.T)*sa[:,None]*sb[None,:]
    current=timed(reference)
    def isolated(*,before_replay=None):
        if before_replay is not None:before_replay(current.outputs)
        current.outputs.copy_(reference());return current.outputs
    current.single_invocation=isolated
    h.verify_timed_run(current,inputs=inputs,originals=originals,expected=reference(),
                      perturb=probes,reference=reference,compare=torch.testing.assert_close,single_invocations=2)
    assert all(torch.equal(x,y) for x,y in zip(inputs,originals))


def test_sage_actual_e2m1_rounding_selects_even_code_at_every_midpoint():
    path = ROOT/'tasks/triton2flydsl/aiter/fav3_sage_mxfp4/fav3_sage_mxfp4.py'
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == '_e2m1_rne_magnitude')
    fn.decorator_list = []
    ns = {'tl': SimpleNamespace(uint32=torch.int64)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    codes = ns[fn.name]
    assert codes(midpoints).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert codes(midpoints - .001).tolist() == list(range(7))
    assert codes(midpoints + .001).tolist() == list(range(1, 8))
    assert codes(torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 100.])).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7]


def test_sage_actual_residual_expansion_reconstructs_all_four_contractions():
    path = ROOT/'tasks/triton2flydsl/aiter/fav3_sage_mxfp4/fav3_sage_mxfp4.py'
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'sage_quant_mxfp4')
    # Exercise the actual feature/scale concatenations, without GPU quantization.
    names = {'q_fp4', 'k_fp4', 'q_scale', 'k_scale'}
    statements = [n for n in fn.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr == 'cat']
    assert len(statements) == 4
    gen = torch.Generator().manual_seed(305)
    q0, q1 = (torch.randn(2, 32, generator=gen) for _ in range(2))
    k0, k1 = (torch.randn(3, 32, generator=gen) for _ in range(2))
    qs0, qs1, ks0, ks1 = (torch.rand(n, 1, generator=gen) for n in (2, 2, 3, 3))
    ns = dict(torch=torch, q_fp4=q0, q1=q1, k_fp4=k0, k1=k1,
              q_scale=qs0, qs1=qs1, k_scale=ks0, ks1=ks1)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), ns)
    expanded_q = ns['q_fp4'] * ns['q_scale'].repeat_interleave(32, -1)
    expanded_k = ns['k_fp4'] * ns['k_scale'].repeat_interleave(32, -1)
    expected = (q0*qs0 + q1*qs1) @ (k0*ks0 + k1*ks1).T
    torch.testing.assert_close(expanded_q @ expanded_k.T, expected)


@pytest.mark.parametrize('unstable', [False, True])
def test_topk_correctness_only_fp32_case_checks_large_logits(monkeypatch, unstable):
    task = 'flydsl2flydsl/topk_gating_softmax_kernel'
    h = helpers(task)
    tree = ast.parse((ROOT/'tasks'/task/'test_kernel_harness.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_test')
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'run_perftest')
    fn.body = fn.body[:end] + [ast.Return(value=ast.Constant(True))]
    ast.fix_missing_locations(fn)
    seen = []
    def build(**kwargs):
        def launch(logits, weights, indices, token_indices, num_tokens, **kwargs):
            seen.append(float(logits.max()))
            probs = logits.exp() / logits.exp().sum(-1, keepdim=True) if unstable else logits.softmax(-1)
            w, idx = probs.topk(weights.shape[1], dim=-1)
            weights.copy_(w/w.sum(-1, keepdim=True)); indices.copy_(idx)
            token_indices.copy_(torch.arange(weights.shape[1])[None, :]*num_tokens + torch.arange(num_tokens)[:, None])
        return launch
    for name in ('rand', 'empty'):
        factory = getattr(torch, name)
        def cpu(*args, _factory=factory, **kwargs):
            if 'device' in kwargs: kwargs['device'] = 'cpu'
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    f = functions(task, ['reference_topk'], DTYPE_FP32=torch.float32)
    ns = dict(torch=torch, build_topk_gating_softmax_module=build,
              _torch_dtype=lambda dtype: torch.float32, DTYPE_FP32=torch.float32,
              reference_topk=f.reference_topk, require_unchanged=h.require_unchanged,
              require_topk_output=h.require_topk_output)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'topk-public-fp32-control', 'exec'), ns)
    if unstable:
        with pytest.raises(AssertionError): ns['run_test'](4, 8, 2, 'f32')
    else:
        assert ns['run_test'](4, 8, 2, 'f32')
    assert max(seen) > 990


def test_dynamic_quantizer_cannot_resize_the_caller_scale_to_hide_invalid_rank():
    f = functions('triton2flydsl/aiter/dynamic_quant_fp8',
        ['_checked_quant_output', '_compare_token_outputs'])
    x = torch.ones(2, 4)
    qx = torch.ones(2, 4, dtype=torch.int8)
    scale = torch.ones(1, 2)  # Wrong rank, but broadcasts like the old comparison.
    with pytest.raises(AssertionError, match='caller scale shape'):
        f._compare_token_outputs((qx, scale), (qx.clone(), torch.ones(2)), qx, x, scale, torch.int8)


@pytest.mark.parametrize('value', [None, torch.float8_e4m3fnuz, torch.float32, torch.float8_e4m3fn])
def test_mqa_public_dtype_export_is_checked(value):
    f = functions('triton2flydsl/aiter/fp8_mqa_logits', ['_require_public_dtype'])
    mod = SimpleNamespace() if value is None else SimpleNamespace(e4m3_dtype=value)
    if value == torch.float8_e4m3fn:
        f._require_public_dtype(mod, torch.float8_e4m3fn)
    else:
        with pytest.raises(AssertionError, match='public e4m3_dtype'):
            f._require_public_dtype(mod, torch.float8_e4m3fn)


@pytest.mark.parametrize('pair', [False, True])
def test_sigmoid_routing_helpers_accept_the_actual_tuple_probe_interface(pair):
    h = helpers('triton2flydsl/aiter/moe_routing_sigmoid_top1')
    x = torch.tensor([1., -2.]); original = x.clone()
    ref = (lambda: (x*3, x*5)) if pair else (lambda: x*3)
    state = SimpleNamespace(bound=True, outputs=ref())
    def replay():
        actual = ref()
        if pair:
            for out, value in zip(state.outputs, actual): out.copy_(value)
        else: state.outputs.copy_(actual)
        return state.outputs
    state.rerun = replay
    def isolated(*, before_replay=None):
        if before_replay is not None: before_replay(state.outputs)
        return replay()
    state.single_invocation = isolated
    def compare(actual, wanted):
        if pair:
            for a, b in zip(actual, wanted): torch.testing.assert_close(a, b)
        else: torch.testing.assert_close(actual, wanted)
    check = h.verify_timed_pair if pair else h.verify_timed_run
    check(state, inputs=(x,), originals=(original,), expected=ref(),
          perturb=(lambda: x.mul_(.5), lambda: x.neg_()), reference=ref,
          compare=compare, single_invocations=2)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize('ignore_epsilon', [False, True])
def test_fused_add_rmsnorm_calls_nondefault_epsilon_at_unchanged_tolerance(monkeypatch, ignore_epsilon):
    task = 'triton2flydsl/aiter/fused_add_rmsnorm'
    h = helpers(task)
    seen = []
    def invoke(out, x, residual, summed, weight, epsilon):
        seen.append(epsilon)
        summed.copy_(x + residual)
        eps = 1e-5 if ignore_epsilon else epsilon
        out.copy_((summed.float() * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype))
        return out
    f = functions(task, ['run_correctness', '_run_kernel', '_torch_rmsnorm', '_checked_pair_output'],
        EPS=1e-5, _load_source=lambda: SimpleNamespace(rmsnorm2d_fwd_with_add=invoke),
        TEST_SHAPES=[{'name': 'small', 'M': 2, 'N': 8}], DTYPES=['fp16'],
        _torch_dtype=lambda dt: torch.float16,
        _make_inputs=lambda m,n,dt: (torch.arange(m*n).reshape(m,n).to(dt)/10, torch.ones(m,n,dtype=dt),torch.ones(n,dtype=dt)),
        nonuniform_input_probes=h.nonuniform_input_probes, check_reused_calls=h.check_reused_calls,
        require_unchanged=h.require_unchanged, require_tensor_contract=h.require_tensor_contract,
        allclose_output=h.allclose_output)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    assert f.run_correctness(verbose=False) is (not ignore_epsilon)
    assert 1e-3 in seen


def test_unified_replay_changes_relative_page_mapping_without_changing_shape():
    task = 'triton2flydsl/aiter/unified_attention'
    h = helpers(task);received = {}
    f = functions(task, ['_unified_replay_validator'],
        ref_paged_attn=lambda q,k,v,*a,**kw:q.clone(),
        _checked_unified_output=lambda *a:None, _compare_unified_output=lambda *a:None,
        scaled_input_probes=h.scaled_input_probes,
        verify_timed_run=lambda timed,**kwargs:received.update(kwargs))
    q=torch.ones(1,1,16);k=torch.ones(8,16,1,16);v=k.clone();out=q.clone()
    pages=torch.tensor([[0,1,2,3]],dtype=torch.int32);starts=torch.tensor([0,1]);length=torch.tensor([64])
    f._unified_replay_validator(q,k,v,out,pages,starts,length,.25,1,1,64,0,0)(None)
    candidates=[]
    for probe in received['perturb']:
        for value,original in zip(received['inputs'],received['originals']):value.copy_(original)
        probe();candidates.append(pages.clone())
    assert any(p.tolist()==[[3,2,1,0]] for p in candidates)


def test_strided_softmax_rows_reject_reversal_and_cross_row_reuse():
    f = functions('triton2flydsl/aiter/softmax', ['_reference_softmax', '_compare_probabilities'],
                  allclose_output=helpers('triton2flydsl/aiter/softmax').allclose_output)
    store = torch.zeros(3, 1026, dtype=torch.bfloat16)
    for row,peak in enumerate((1,17,129)):store[row,2*peak]=6
    expected = f._reference_softmax(store[:,::2])
    f._compare_probabilities(expected, expected)
    for wrong in (expected.flip(0), expected.float().mean(0,keepdim=True).expand_as(expected).to(expected.dtype)):
        with pytest.raises(AssertionError): f._compare_probabilities(wrong, expected)


def test_extend_prefix_control_rejects_reconstructing_addresses_from_first_index():
    task = 'triton2flydsl/sglang/extend_attention'
    h = helpers(task);captured = {}
    f = functions(task, ['_extend_replay_validator','reference','_checked_extend_output','_compare_extend_output'],
        require_tensor_contract=h.require_tensor_contract,scaled_input_probes=h.scaled_input_probes,
        verify_timed_run=lambda timed,**kwargs:captured.update(kwargs))
    q,k,v=(torch.ones(2,1,4,dtype=torch.bfloat16) for _ in range(3))
    kb=vb=torch.ones(4,1,4,dtype=torch.bfloat16)
    kb=kb.clone();vb=vb.clone();qo=torch.tensor([0,2]);kvp=torch.tensor([0,4]);indices=torch.arange(4)
    cfg=dict(prefix=[4],extend=[2],head=1,kv_head=1,Lq=4,Lv=4,causal=False)
    f._extend_replay_validator((q,k,v,kb,vb,qo,kvp,indices),cfg)(None)
    captured['perturb'][-1]()
    wanted=captured['reference']()
    assert indices[0]==0 and indices.tolist()!=list(range(4))
    reconstructed=torch.arange(indices.numel())+indices[0]
    wrong=f.reference(q,k,v,kb,vb,reconstructed,cfg).to(torch.bfloat16)
    with pytest.raises(AssertionError,match='original extend gate'):
        f._compare_extend_output(wrong,wanted)


def test_empty_silu_call_does_not_require_or_fabricate_a_gpu_launch():
    path=ROOT/'tasks/triton2flydsl/aiter/fused_silu_mul/scripts/candidate_checks.py'
    spec=importlib.util.spec_from_file_location('empty_silu_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    observed=set()
    out=audit.checked_candidate_invocation(lambda x:x.new_empty((*x.shape[:-1],x.shape[-1]//2)),observed,torch.empty(0,8))
    assert out.shape==(0,4) and not observed
    with pytest.raises(RuntimeError,match='No FlyDSL kernel'):
        audit.checked_candidate_invocation(lambda x:x.new_empty(1,4),observed,torch.empty(1,8))


@pytest.mark.parametrize('narrow_int8', [False, True])
def test_actual_large_int8_domain_control_rejects_fp8_narrowing(monkeypatch,narrow_int8):
    task='triton2flydsl/aiter/gemm_a8w8';h=helpers(task)
    f=functions(task,['run_domain_controls','_torch_ref','_checked_scaled_gemm_output'],
                _fp8_e4m3_dtype=lambda:torch.float8_e4m3fn,require_unchanged=h.require_unchanged)
    for name in ('arange','linspace','full','zeros','empty'):
        factory=getattr(torch,name)
        def cpu(*args,_factory=factory,**kwargs):
            if 'device' in kwargs:kwargs['device']='cpu'
            return _factory(*args,**kwargs)
        monkeypatch.setattr(torch,name,cpu)
    seen=[]
    def invoke(x,w,xs,ws,bias,dtype,y=None):
        seen.append((x.dtype,float(x.float().abs().max())))
        a=x.to(torch.float8_e4m3fn) if narrow_int8 and x.dtype==torch.int8 else x
        b=w.to(torch.float8_e4m3fn) if narrow_int8 and w.dtype==torch.int8 else w
        out=f._torch_ref(a,b,xs,ws,bias,dtype)
        if y is not None:y.copy_(out);return y
        return out
    if narrow_int8:
        with pytest.raises(AssertionError):f.run_domain_controls(SimpleNamespace(gemm_a8w8=invoke))
    else:f.run_domain_controls(SimpleNamespace(gemm_a8w8=invoke))
    assert any(dt==torch.int8 and maximum>16 for dt,maximum in seen)


def test_dynamic_quantizer_cannot_change_caller_dtype_and_return_a_correct_copy():
    f=functions('triton2flydsl/aiter/dynamic_quant_fp8',['_checked_quant_output'])
    x=torch.ones(2,4);qx=torch.ones(2,4,dtype=torch.int8)
    caller=torch.ones(2,dtype=torch.float64);returned=caller.float()
    assert torch.equal(caller,returned)  # Equality alone misses a dtype change.
    with pytest.raises(AssertionError,match='Caller scale dtype/device'):
        f._checked_quant_output(qx,qx,x,torch.int8,returned,caller,expected_scale_shape=(2,))


@pytest.mark.parametrize('missing_return',[False,True])
def test_fused_add_rmsnorm_checks_the_actual_return_value(missing_return):
    task='triton2flydsl/aiter/fused_add_rmsnorm';h=helpers(task)
    f=functions(task,['_run_kernel'],EPS=1e-5,require_tensor_contract=h.require_tensor_contract)
    def invoke(out,x,residual,residual_out,weight,epsilon):
        out.copy_(x);residual_out.copy_(x+residual)
        return None if missing_return else out
    x=torch.ones(2,8);r=x.clone();w=torch.ones(8)
    mod=SimpleNamespace(rmsnorm2d_fwd_with_add=invoke)
    if missing_return:
        with pytest.raises(AssertionError,match='Tensor'):f._run_kernel(mod,x,r,w)
    else:
        out,res=f._run_kernel(mod,x,r,w);torch.testing.assert_close(out,x)


def test_sigmoid_router_checks_negative_winning_logits():
    task='triton2flydsl/aiter/moe_routing_sigmoid_top1';h=helpers(task);received={}
    f=functions(task,['_pair_replay_validator','_torch_routing_ref','_compare_routing_pair','_checked_pair_output'],
        verify_timed_pair=lambda timed,**kwargs:received.update(kwargs))
    x=torch.ones(2,16,dtype=torch.float16);w=torch.ones(16,4,dtype=torch.float16)
    f._pair_replay_validator(x,w,4,False)(None)
    received['perturb'][-1]()
    expected=received['reference']()
    torch.testing.assert_close(expected[1],torch.full((2,1),float(torch.sigmoid(torch.tensor(-1.)))))
    wrong=(torch.zeros(2,1,dtype=torch.int32),torch.full((2,1),.5))
    with pytest.raises(AssertionError,match='router weights'):received['compare'](wrong,expected)


def test_quantized_gemm_rejects_fp16_narrowing_of_finite_bf16_operands():
    task = 'torch2flydsl/gemm_a8w8_kernel'
    h = helpers(task)
    received = {}
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, a, w): return (a.float() @ w.float().T).to(torch.bfloat16)
    f = functions(task, ['_gemm_replay_validator', '_checked_quant_gemm_output',
                        '_compare_quant_gemm_output', '_norm_worst'],
        require_unchanged=h.require_unchanged, require_tensor_contract=h.require_tensor_contract,
        sample_output_validator=h.sample_output_validator, TOL=.01,
        verify_timed_run=lambda timed, **kwargs: received.update(kwargs))
    a = torch.ones((2, 8), dtype=torch.bfloat16)
    w = torch.ones((3, 8), dtype=torch.bfloat16)
    def narrowed(): return (a.half().float() @ w.half().float().T).to(torch.bfloat16)
    f._gemm_replay_validator(SimpleNamespace(Model=Model), a, w)(None)
    received['compare'](narrowed(), received['reference']())
    received['perturb'][-1]()
    expected = received['reference']()
    assert bool(torch.isfinite(expected).all())
    with pytest.raises(AssertionError, match='Non-finite'):
        received['compare'](narrowed(), expected)


@pytest.mark.parametrize('ignore_k_stride', [False, True])
def test_sigmoid_routing_checks_nonunit_activation_stride(monkeypatch, ignore_k_stride):
    task = 'triton2flydsl/aiter/moe_routing_sigmoid_top1'; h = helpers(task)
    def make_inputs(m,k,n):
        g=torch.Generator().manual_seed(19)
        return (torch.randn(m,k,generator=g,dtype=torch.float16),
                torch.randn(k,n,generator=g,dtype=torch.float16)*.01)
    f=functions(task,['run_domain_controls','_torch_routing_ref','_checked_pair_output','_compare_routing_pair'],
        _make_inputs=make_inputs,require_unchanged=h.require_unchanged)
    factory=torch.full
    monkeypatch.setattr(torch,'full',lambda *args,**kwargs:factory(*args,**{**kwargs,'device':'cpu'}))
    seen=[]
    def invoke(x,w,topk,fused_shared_experts):
        x=x.reshape(-1,x.shape[-1]);seen.append(x.stride(-1))
        if ignore_k_stride and x.stride(-1)!=1:
            x=x.as_strided(x.shape,(x.shape[-1],1))
        return f._torch_routing_ref(x,w,16,fused_shared_experts)[:2]
    if ignore_k_stride:
        with pytest.raises(AssertionError):f.run_domain_controls(SimpleNamespace(routing_sigmoid_top1=invoke))
    else:f.run_domain_controls(SimpleNamespace(routing_sigmoid_top1=invoke))
    assert 2 in seen


def test_moe_provided_closure_keeps_all_raw_preparation_in_every_invocation():
    seen=[]
    def provided(*args):
        seen.append(tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in args))
        return args[0].clone()
    f=functions('torch2flydsl/fmoe_g1u1_tkw1_kernel',['_make_prepared_baseline_op'],
        _provided_implementation=lambda:SimpleNamespace(fp8_moe=provided,fp8_dtype=lambda hidden:torch.float8_e4m3fn))
    model=SimpleNamespace(w1=torch.ones(2,4,4),w2=torch.ones(2,4,2),activation='silu')
    hidden=torch.ones(2,4);weights=torch.ones(2,1);ids=torch.zeros(2,1,dtype=torch.int32)
    op=f._make_prepared_baseline_op(model,hidden,weights,ids)
    assert seen==[]
    op();hidden.mul_(2);model.w1.mul_(3);model.w2.mul_(4);weights.mul_(.5);ids.fill_(1);op()
    assert len(seen)==2
    for index,current in enumerate((hidden,model.w1,model.w2,weights,ids)):
        torch.testing.assert_close(seen[1][index],current)
        assert not torch.equal(seen[0][index],seen[1][index])


def test_moe_independent_oracle_arithmetic_is_unchanged():
    import hashlib
    p=ROOT/'tasks/torch2flydsl/fmoe_g1u1_tkw1_kernel/model.py'
    # Pin original arithmetic bytes independently of Python's AST schema.
    source=p.read_text();doc_end=ast.parse(source).body[0].end_lineno
    arithmetic='\n'.join(source.splitlines()[doc_end:])
    assert hashlib.sha256(arithmetic.encode()).hexdigest() == '879652359fcd61f47b4ddabe1327f0d99a63f8c22c4128c5912dec054e85eeb2'


def test_moe_candidate_cannot_import_the_new_provided_baseline(tmp_path, monkeypatch):
    path=ROOT/'tasks/torch2flydsl/fmoe_g1u1_tkw1_kernel/task_runtime.py'
    spec=importlib.util.spec_from_file_location('moe_runtime_baseline_import_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    candidate=tmp_path/'kernel.py';candidate.write_text('from task_baseline import fp8_moe\n')
    (tmp_path/'task_baseline.py').write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError,match='Protected dependency'):
        module.candidate_source_closure([candidate])


def test_moe_actual_provided_source_is_declared_for_compile_and_protection():
    import yaml
    task=ROOT/'tasks/torch2flydsl/fmoe_g1u1_tkw1_kernel'
    cfg=yaml.safe_load((task/'config.yaml').read_text())
    assert cfg['baseline']['source_files']==['model.py','task_baseline.py']
    assert 'task_baseline.py' not in cfg['candidate']['editable']
    compile((task/'task_baseline.py').read_text(), 'task_baseline.py', 'exec')


@pytest.mark.parametrize('topk',[1,2])
def test_moe_live_probes_cover_zero_rows_and_large_expert_groups(topk):
    task='torch2flydsl/fmoe_g1u1_tkw1_kernel';h=helpers(task);captured={}
    model=SimpleNamespace(w1=torch.ones(16,8,4),w2=torch.ones(16,4,4),experts=16,
        forward_with_routing=lambda hidden,weights,ids,plan:hidden.clone())
    f=functions(task,['_fmoe_inputs','_fmoe_replay_validator'],
        _checked_fmoe_output=lambda actual,hidden:actual,require_unchanged=h.require_unchanged,
        scaled_input_probes=lambda *args:(),_compare_fmoe_output=lambda actual,expected:None,
        verify_timed_run=lambda timed,**kwargs:captured.update(kwargs))
    hidden=torch.ones(128,4);weights=torch.full((128,topk),1./topk)
    ids=torch.arange(topk,dtype=torch.int32)[None,:].expand(128,-1).clone()
    originals=tuple(v.clone() for v in (hidden,model.w1,model.w2,weights,ids))
    f._fmoe_replay_validator(model,hidden,weights,ids,())(None)
    for index,probe in enumerate(captured['perturb']):
        for value,old in zip((hidden,model.w1,model.w2,weights,ids),originals):value.copy_(old)
        probe()
        if index<3:
            assert not bool((hidden,model.w1,model.w2)[index].any())
        else:
            count=torch.bincount(ids.flatten().long(),minlength=model.experts)
            if index==5:
                assert sorted(count[count>0].tolist())==[33]*topk+[95]*topk
            else:
                assert int(count.max())==128 and int((count>32).sum())==topk
            assert int(ids.min())>=0 and int(ids.max())<model.experts
            torch.testing.assert_close(weights.sum(-1),torch.ones(128))
            for row in ids: assert torch.unique(row).numel()==topk
