"""CPU numerical/dispatch controls; actual Triton and graphs require GPU validation."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'
FIRST = ('triton_correct_attn_cp_out', 'triton_decode_attn_stage2',
         'triton_decode_attn_stage1', 'triton_decode_attn_grouped_stage1')
REMAINING = ('triton_chunked_prefill_paged_decode', 'triton_flash_prefill_attention',
             'triton_paged_prefix_prefill', 'triton_paged_prefix_prefill_alibi',
             'triton_unified_attention_2d', 'triton_unified_attention_3d')


def load(path):
    spec = importlib.util.spec_from_file_location('_attention_cpu_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=FIRST + REMAINING)
def contract(request, monkeypatch):
    monkeypatch.chdir(ROOT)
    name = request.param
    checks = load(TASKS / name / '_arena_checks.py')
    harness = load(TASKS / name / 'scripts/task_runner.py')
    if name == FIRST[0]:
        torch.manual_seed(71)
        args = (torch.randn(2, 2, 4), torch.randn(2, 2, 2), 1)
        kwargs = {'is_base_e': True}
    elif name == FIRST[1]:
        args = (*harness.make_stage1_outputs(1, 2, 1, 4, 8, 2, 2, 'cpu', torch.float32), 2)
        kwargs = {}
    elif name in FIRST:
        q, k, v, out, pages, lengths, scale = harness.make_inputs(1, 2, 1, 4, 8, 2, 2, 'cpu', torch.float32)
        args = (q, k, v, out, pages, lengths, 2, scale, 2)
        kwargs = {'logit_cap': 0.0}
    elif name == REMAINING[0]:
        args = harness.make_test_data(1, 4, 2, 1, 8, 2, 2, 'cpu', torch.float32)
        kwargs = {'filter_by_query_len': False}
    elif name == REMAINING[1]:
        args = (torch.randn(4, 2, 8), torch.randn(4, 1, 8), torch.randn(4, 1, 8),
                torch.zeros(4, 2, 8), torch.tensor([0], dtype=torch.int32), torch.tensor([4], dtype=torch.int32))
        kwargs = {'max_input_len': 4, 'is_causal': True}
    elif name in REMAINING[2:4]:
        kc, vc, pages, _, _ = harness.setup_paged_kv_cache(1, 4, 1, 8, 2, 'cpu', torch.float32)
        args = (torch.randn(2, 2, 8), torch.randn(2, 1, 8), torch.randn(2, 1, 8),
                torch.zeros(2, 2, 8), kc, vc, pages, torch.tensor([0, 2]), torch.tensor([6]))
        kwargs = {'max_input_len': 2}
        if name.endswith('_alibi'): kwargs['alibi_slopes'] = torch.tensor([0.5, 0.125])
    elif name == REMAINING[4]:
        args = harness.make_test_data(1, 2, 4, 2, 1, 8, 2, 'cpu', torch.float32)
        kwargs = {'sliding_window': 2, 'softcap': 2.0}
    else:
        import sys
        # The independent original reference uses only Triton's integer helper.
        monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()))
        args = harness.make_test_data(1, 1, 32, 2, 1, 8, 2, 'cpu', torch.float32)
        kwargs = {'num_segments': 2}
    return checks, harness, args, kwargs


def numerical_candidate(checks, harness):
    """Pure CPU oracle double, explicitly bypassed by numerical tests only."""
    def candidate(*args, **kwargs):
        values = dict(enumerate(args)) | kwargs
        wanted = checks.expected_outputs(harness, values)
        if checks.OUTPUT_KEYS:
            for key, expected in zip(checks.OUTPUT_KEYS, wanted):
                values[key].copy_(expected)
            return None
        return wanted
    return candidate


def install_numerical_double(monkeypatch, checks):
    monkeypatch.setattr(checks, 'checked_candidate_call', lambda module, fn, *args, **kwargs: fn(*args, **kwargs))


@pytest.mark.parametrize('fault', ['none', 'input_mutation', 'metadata_mutation', 'partial_write', 'wrong_auxiliary'])
def test_actual_correctness_wrapper_pristine_full_output_guards(contract, monkeypatch, fault):
    checks, harness, args, kwargs = contract
    install_numerical_double(monkeypatch, checks)
    good = numerical_candidate(checks, harness)
    def candidate(*a, **kw):
        if fault == 'input_mutation':
            a[checks.PERTURB_KEYS[0]].zero_()
        result = good(*a, **kw)
        outputs = tuple(a[k] for k in checks.OUTPUT_KEYS) if checks.OUTPUT_KEYS else result
        if fault == 'metadata_mutation':
            a[checks.PERTURB_KEYS[0]].transpose_(0, 1)
        if fault == 'partial_write':
            outputs[0].reshape(-1)[-1] = float('nan')
        if fault == 'wrong_auxiliary':
            outputs[-1].reshape(-1)[-1] += 10
        return result
    mod = SimpleNamespace(**{checks.SYMBOL: candidate})
    harness.load_module = lambda: mod
    with checks.checked_modules(harness):
        checked = getattr(harness.load_module(), checks.SYMBOL)
        if fault == 'none':
            checked(*args, **kwargs)
        else:
            with pytest.raises(AssertionError):
                checked(*args, **kwargs)
    assert getattr(mod, checks.SYMBOL) is candidate


@pytest.mark.parametrize('fault', ['none', 'stale', 'unwritten_tail', 'auxiliary', 'mutated_input'])
def test_exact_measured_outputs_perturbed_replay_and_restore(contract, monkeypatch, fault):
    checks, harness, args, kwargs = contract
    install_numerical_double(monkeypatch, checks)
    good = numerical_candidate(checks, harness)
    mod = SimpleNamespace(**{checks.SYMBOL: good})
    pristine = {i: t.clone() for i, t in enumerate(args) if isinstance(t, torch.Tensor)}
    def fn():
        getattr(mod, checks.SYMBOL)(*args, **kwargs)
    replay_count = []
    def benchmark(measured, *, timed_run, **options):
        assert options == {'warmup': 10, 'repetition': 100}
        outputs = measured()
        old = tuple(o.clone() for o in outputs)
        def replay():
            replay_count.append(1)
            new = old if fault == 'stale' else checks.expected_outputs(harness, dict(enumerate(args)) | kwargs)
            for out, wanted in zip(outputs, new):
                if fault == 'unwritten_tail':
                    out.reshape(-1)[:-1].copy_(wanted.reshape(-1)[:-1])
                else:
                    out.copy_(wanted)
            if fault == 'auxiliary': outputs[-1].reshape(-1)[-1] += 10
            if fault == 'mutated_input': args[checks.PERTURB_KEYS[0]].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    harness._TimedRun = SimpleNamespace
    if fault == 'none':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == .125 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert replay_count == [1]
    for key, original in pristine.items():
        assert torch.equal(checks._tensor_bytes(args[key]), checks._tensor_bytes(original))
    assert getattr(mod, checks.SYMBOL) is good


def test_readonly_shape_only_nan_bytes_are_compared_without_numeric_assumptions(contract):
    checks, harness, args, kwargs = contract
    plan = checks.CallPlan(harness, args, kwargs)
    plan.unchanged()
    if checks.SYMBOL == 'decode_softmax_reducev_fwd':
        args[1].fill_(float('nan'))
        plan = checks.CallPlan(harness, args, kwargs)
        plan.unchanged()
        args[1].zero_()
        with pytest.raises(AssertionError, match='read-only'):
            plan.unchanged()


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'alias', 'missing'])
def test_full_output_contract_rejects_metadata_and_aliases(contract, fault):
    checks, harness, args, kwargs = contract
    plan = checks.CallPlan(harness, args, kwargs)
    output = [o.clone() for o in plan.expected]
    if fault == 'shape': output[0] = output[0].unsqueeze(0)
    elif fault == 'dtype': output[0] = output[0].double()
    elif fault == 'alias':
        source = next(v for k, v in plan.values.items() if k not in checks.OUTPUT_KEYS
                      and isinstance(v, torch.Tensor) and v.dtype == output[0].dtype
                      and v.numel() >= output[0].numel())
        # Same storage, matching metadata, hence specifically an alias violation.
        output[0] = source.reshape(-1)[:output[0].numel()].view(output[0].shape)
    elif fault == 'missing': output.pop()
    with pytest.raises(AssertionError): plan.check(output)


@pytest.mark.parametrize('operation', [lambda x: x.mean(), lambda x: x @ x.t()])
def test_real_dispatch_guard_rejects_torch_compute_and_unused_kernel(contract, monkeypatch, operation):
    checks, _, _, _ = contract
    class JIT:
        def run(self): pass
    runtime = SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)
    compiler = SimpleNamespace(CompiledKernel=type('CompiledKernel', (), {}))
    monkeypatch.setattr(checks, '_triton_runtime', lambda: (runtime, compiler, JIT, ()))
    module = SimpleNamespace(**{checks.KERNEL: JIT()})
    with pytest.raises(AssertionError, match='non-preparation PyTorch'):
        checks.checked_candidate_call(module, operation, torch.ones(2, 2))
    with pytest.raises(AssertionError, match='No genuine declared'):
        checks.checked_candidate_call(module, lambda: torch.empty(2))
    assert runtime.launch_enter_hook is runtime.launch_exit_hook is None


def test_fake_jit_and_initializer_are_rejected(contract, monkeypatch):
    checks, _, _, _ = contract
    class JIT:
        def run(self): pass
    monkeypatch.setattr(checks, '_triton_runtime', lambda: (None, None, JIT, ()))
    fake = type('JITFunction', (), {'__module__': 'triton.runtime.jit'})()
    with pytest.raises(AssertionError, match='genuine Triton'):
        checks.checked_candidate_call(SimpleNamespace(**{checks.KERNEL: fake}), lambda: None)
    h = SimpleNamespace(load_module=lambda: torch.ones(4).mean(), run_correctness=lambda: None,
                        run_performance=lambda: None)
    checks.install(h)
    with pytest.raises(AssertionError, match='non-preparation'):
        h.load_module()


# Filled with reviewed base bytes; no git-history dependency in source archives.
ORIGINALS = {'triton_correct_attn_cp_out/source/triton_correct_attn_cp_out.py': '2fd2502550ac36e634a6cd01fe247b66704d4d48a9b369d96084d9c2d6f7d220',
 'triton_correct_attn_cp_out/config.yaml': 'd734d0a7b8f9ddf17111dec1ee435e81fc68865079a8d524099b7486e9b1914f',
 'triton_correct_attn_cp_out/workloads.json': 'c3dbea0d0cf60a99c050e4795a8ab4d10411ec2f8ea01e5e59e3ea50155502a1',
 'triton_correct_attn_cp_out/scripts/task_runner.py': '1c5b87ae973ffbf660cd520343c990c83e8bbb4f3b6d57e16f49e826408a12fd',
 'triton_decode_attn_stage2/source/_fwd_kernel_stage2.py': 'db986832bb69e49548e5ef3eddcc8b56aabaa6b98f8193acb7bcfbc44452e7d2',
 'triton_decode_attn_stage2/config.yaml': '7e1e7018cd00382729b75e71be8f2c211c146e957381472ddcae0d61ae7e6866',
 'triton_decode_attn_stage2/scripts/task_runner.py': '9b5ae812b0156ed5abb5cdfec11a35e96634e5d856a5a7ecd4d9520d5b2b9d91',
 'triton_decode_attn_stage1/source/_fwd_kernel_stage1.py': '025e681f2a2a11be27b5384aca729841d04d1ada3ecfb0757a7f713c37e95c1a',
 'triton_decode_attn_stage1/config.yaml': '793e6e10b6227e67ce8d50ca18f88ca87deb888631af904c5a4998bfea6768ba',
 'triton_decode_attn_stage1/scripts/task_runner.py': 'bdafdb2e9cdfb8abab438dc9f4486cbed58743358482637c9b4d2400f5f24ff2',
 'triton_decode_attn_grouped_stage1/source/_fwd_grouped_kernel_stage1.py': '3f0d728ac011a7b9bd5eee1ae9d3eb69ef1ad91367918c4660c05a743cac7e4f',
 'triton_decode_attn_grouped_stage1/config.yaml': '24c98bbcaff663b18709845a25e03d1608cf50e6068e65caacace55e37e47dbd',
 'triton_decode_attn_grouped_stage1/scripts/task_runner.py': 'd2dacb2807331c827edcbc97e8419ee17db41992470e6574be2262a9e4b425ad'}


def test_original_kernels_cases_gates_and_timers_preserved():
    for relative, digest in ORIGINALS.items():
        assert hashlib.sha256((TASKS / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('alibi', [False, True])
def test_paged_oracle_reconstructs_actual_cache_and_page_indirection(monkeypatch, alibi):
    monkeypatch.chdir(ROOT)
    task = TASKS / ('triton_paged_prefix_prefill' + ('_alibi' if alibi else ''))
    checks, h = load(task / '_arena_checks.py'), load(task / 'scripts/task_runner.py')
    torch.manual_seed(909)
    kc, vc, pages, dense_k, dense_v = h.setup_paged_kv_cache(2, 4, 2, 8, 2, 'cpu', torch.float32)
    q, k, v = torch.randn(4, 4, 8), torch.randn(4, 2, 8), torch.randn(4, 2, 8)
    starts, lengths, slopes = torch.tensor([0, 2, 4]), torch.tensor([6, 6]), torch.tensor([.5, .25, .125, .0625])
    a = dict(enumerate((q, k, v, torch.empty_like(q), kc, vc, pages, starts, lengths)))
    if alibi:
        a['alibi_slopes'] = slopes
        expected = h.reference_attention_alibi(q,k,v,dense_k,dense_v,starts,lengths,slopes,2,4,2,4,2,8)
    else:
        expected = h.reference_paged_attention(q,k,v,dense_k,dense_v,starts,lengths,2,4,2,4,2,8)
    torch.testing.assert_close(checks.expected_outputs(h, a)[0], expected)
    # Physically permute blocks and repair the table: the same operator must result.
    order = torch.arange(kc.shape[0]-1, -1, -1)
    permuted = a | {4:kc[order], 5:vc[order], 6:kc.shape[0]-1-pages}
    torch.testing.assert_close(checks.expected_outputs(h, permuted)[0], expected)
    # Changing real cache content, while the generator's old dense tensors stay
    # fixed, must affect this reference and invalidate a cached old answer.
    changed = permuted | {5:permuted[5]+3}
    assert not torch.allclose(checks.expected_outputs(h, changed)[0], expected)

ORIGINALS.update({'triton_chunked_prefill_paged_decode/source/triton_chunked_prefill_paged_decode.py': '3fca551b8f21dbfca500e5ee003112c88ddce8c364f1c2c8e16a0596d08bdf08', 'triton_chunked_prefill_paged_decode/config.yaml': '9c2299ba86847d816573c626c13821ac00d26ed527a83001ef2491b04a5e8865', 'triton_chunked_prefill_paged_decode/workloads.json': '4b1cf48ca65f51159d17a2e6de9ff6f2d1436296c6954e3c2325b6ace7fde6c3', 'triton_chunked_prefill_paged_decode/scripts/task_runner.py': '7c0a87b6374ea88b3f45cbe0d73005acc6716734fb51393f71f1b79711d2b71b', 'triton_flash_prefill_attention/source/triton_flash_prefill_attention.py': '7f0a141fd36716f848ab95b4be9df3655b5b6da11e07654466076b701c9d3b5b', 'triton_flash_prefill_attention/config.yaml': '2fbb4c96d9705a85dd42753c9a1638b67212d87639dbc53e36a072b9bee90fff', 'triton_flash_prefill_attention/workloads.json': '27e121b368b9055d59746c784d17954d5b33fe277b24872d2d9406e025698b49', 'triton_flash_prefill_attention/scripts/task_runner.py': '58cad2d8c55e3a37e6dbc18417d8885b205998d8300ae869b8d5adc3af2df961', 'triton_paged_prefix_prefill/source/triton_paged_prefix_prefill.py': 'c02a77c038191ab04f635d861e1395074ca77c049917fb0c2641b7a503affb76', 'triton_paged_prefix_prefill/config.yaml': '6140359155f3f66e00bb28378ae9675f04b48a3c51a1ab21ff546d5b963a8269', 'triton_paged_prefix_prefill/workloads.json': '4b8b6c0d12a4bdc6c6781ca188ed425ccd67e2d280ba7f9a37625f452389d92b', 'triton_paged_prefix_prefill/scripts/task_runner.py': '8a7f02a4464bd0c32fb19a5ed599a10d17324cb4b9c8004f1f48ec4bed51877b', 'triton_paged_prefix_prefill_alibi/source/triton_paged_prefix_prefill_alibi.py': '127615ac25ecda18318d2328176e92f0fd344c3eba78bb9a295d95fbd8ddc742', 'triton_paged_prefix_prefill_alibi/config.yaml': 'b6b06a0041f97d27c626a5c477ae7751cdaa29683130ebe9be687c91e888eb40', 'triton_paged_prefix_prefill_alibi/workloads.json': '45ec8809c90266475f1ff222d49e89e5eeadeab114e4e770471e9109cecd0e23', 'triton_paged_prefix_prefill_alibi/scripts/task_runner.py': '3958ecfc8e2db1508da60aeaef5e1e08af71f24069967a6ad558c695103258ed', 'triton_unified_attention_2d/source/triton_unified_attention_2d.py': '38efd85c3e716f0a689f0b464dc7a108f885d5b2bd5f8ca5674a138a271a259f', 'triton_unified_attention_2d/config.yaml': '31b6b7d72a8c6703ad5eb6f956febd0b5f4f64c682f11b643aed2328dcc2a294', 'triton_unified_attention_2d/workloads.json': '836278af56678a6bb601e7986b7acec804ba7d1fc14a873b690b4b77664be7dc', 'triton_unified_attention_2d/scripts/task_runner.py': '20c714a0b1b51a99ab03ee65c14533c84b4e7c089afbd37c51b66b3f16ba801b', 'triton_unified_attention_3d/source/triton_unified_attention_3d.py': '715800906a4d9244c2045871a6cc0168c4b68f8806e2d97dd66ea71d83ad79b8', 'triton_unified_attention_3d/config.yaml': 'b20d3a7328cf9dfdb3a6c404c86fd8d1431c23e55def172ba5e372bdf3016156', 'triton_unified_attention_3d/workloads.json': '0073fec446196304235c29632d1701083eb13710fcd0b6c9f69105da9951b578', 'triton_unified_attention_3d/scripts/task_runner.py': 'daa9393587de7abb630f2cc4b981c9b11d1e6546686980912c56f168c7c3b4a5'})

LEGACY_DECODE_MANIFESTS = {'triton_decode_attn_stage1': '3f2b49294c8dfbb1ccb626c1717ee04c1733f1d3dfe5eec9adce057d79d2cd94', 'triton_decode_attn_grouped_stage1': '51ef726f3c855fa16ad97211ca17904d5e154728c43f5a2d6741cd4c4e9e7f96', 'triton_decode_attn_stage2': '0da34b7bcc9922d25217f269f32ad4aa2903f32c465e446655b5d57160a2d591'}

def test_added_decode_controls_preserve_every_original_case_and_manifest_field():
    import json
    for name, digest in LEGACY_DECODE_MANIFESTS.items():
        manifest = json.loads((TASKS / name / 'workloads.json').read_text())
        extra = manifest['cases'][5:]
        assert extra and all('contract_case' in row['params'] for row in extra)
        manifest['cases'] = manifest['cases'][:5]
        assert hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == digest


@pytest.mark.parametrize('name', FIRST[1:])
def test_new_decode_control_execution_and_declared_shapes(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    checks = load(TASKS/name/'_arena_checks.py')
    h = load(TASKS/name/'scripts/task_runner.py')
    install_numerical_double(monkeypatch, checks)
    inputs = checks.control_inputs
    good = numerical_candidate(checks, h)
    h.load_module = lambda: SimpleNamespace(**{checks.SYMBOL:good})
    monkeypatch.setattr(checks, 'control_inputs', lambda h, case: inputs(h, case, 'cpu'))
    checks.install_controls(h)
    for case, config in checks.CONTRACT_CASES.items():
        args, kwargs = inputs(h, case, 'cpu')
        assert args[0].shape[0] == config['batch']
        assert args[5].tolist() == config['sequence_lengths']
        assert h.run_contract_correctness(case) == (True, None)
        if name.endswith('stage2'):
            assert args[0][1,:,2:,:-1].eq(7000).all()
            assert args[0][1,:,2:,-1].eq(80).all()
        else:
            assert kwargs['logit_cap'] == config['logit_cap']
            assert args[4][0,:4].tolist() == [7,6,5,4]
            if case == 'inactive_ragged_pages':
                assert args[3][~checks.output_write_mask(dict(enumerate(args)))].eq(23.5).all()


@pytest.mark.parametrize('name', FIRST[2:])
def test_stage1_cap_reference_is_not_uncapped_and_inactive_state_is_exact(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    checks, h = load(TASKS/name/'_arena_checks.py'), load(TASKS/name/'scripts/task_runner.py')
    args, kwargs = checks.control_inputs(h, 'inactive_ragged_pages', 'cpu')
    values = dict(enumerate(args)) | kwargs
    capped = checks.expected_outputs(h, values)[0]
    uncapped = checks.expected_outputs(h, values | {'logit_cap':0.0})[0]
    assert not torch.allclose(capped, uncapped, atol=.01, rtol=.01)
    plan = checks.CallPlan(h, args, kwargs)
    plan.check((capped,))
    poisoned = capped.clone()
    plan.poison((poisoned,))
    active = checks.output_write_mask(values)
    assert torch.isnan(poisoned[active]).all()
    assert poisoned[~active].eq(23.5).all()
    corrupted = capped.clone()
    corrupted[~active] += .001  # Smaller than numeric tolerance, but state must be exact.
    with pytest.raises(AssertionError, match='inactive caller-owned'):
        plan.check((corrupted,))


@pytest.mark.parametrize('name', FIRST[1:])
def test_protocol_extra_correctness_failure_cannot_pass_vacuously(name, monkeypatch):
    import json
    monkeypatch.chdir(ROOT)
    evaluator = load(TASKS/name/'_arena_eval.py')
    checks = load(TASKS/name/'_arena_checks.py')
    data = json.loads((TASKS/name/'workloads.json').read_text())
    calls = []
    def bad(case):
        calls.append(case)
        return False, 'real control rejected'
    h = SimpleNamespace(TEST_SHAPES=data['input_table'], CONTRACT_CASES=checks.CONTRACT_CASES,
                        run_correctness=lambda **kw:(True,None), run_contract_correctness=bad)
    monkeypatch.setattr(evaluator, 'load_harness', lambda:h)
    result = evaluator.evaluate('candidate','correctness')
    assert result['status'] == 'FAIL'
    assert calls == list(checks.CONTRACT_CASES)
    assert all(row['status']=='PASS' for row in result['cases'][:5])
    assert all(row['status']=='FAIL' for row in result['cases'][5:])
