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


def load(path):
    spec = importlib.util.spec_from_file_location('_attention_cpu_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=FIRST)
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
    else:
        q, k, v, out, pages, lengths, scale = harness.make_inputs(1, 2, 1, 4, 8, 2, 2, 'cpu', torch.float32)
        args = (q, k, v, out, pages, lengths, 2, scale, 2)
        kwargs = {'logit_cap': 0.0}
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
ORIGINALS = {'triton_correct_attn_cp_out/source/triton_correct_attn_cp_out.py': '2fd2502550ac36e634a6cd01fe247b66704d4d48a9b369d96084d9c2d6f7d220', 'triton_correct_attn_cp_out/config.yaml': 'd734d0a7b8f9ddf17111dec1ee435e81fc68865079a8d524099b7486e9b1914f', 'triton_correct_attn_cp_out/workloads.json': 'c3dbea0d0cf60a99c050e4795a8ab4d10411ec2f8ea01e5e59e3ea50155502a1', 'triton_correct_attn_cp_out/scripts/task_runner.py': '1c5b87ae973ffbf660cd520343c990c83e8bbb4f3b6d57e16f49e826408a12fd', 'triton_decode_attn_stage2/source/_fwd_kernel_stage2.py': 'db986832bb69e49548e5ef3eddcc8b56aabaa6b98f8193acb7bcfbc44452e7d2', 'triton_decode_attn_stage2/config.yaml': '7e1e7018cd00382729b75e71be8f2c211c146e957381472ddcae0d61ae7e6866', 'triton_decode_attn_stage2/workloads.json': 'b90deb7ebb2cc23bd0d02bd1c13a483019c1abfd5fbbd9908260c32e1eb48174', 'triton_decode_attn_stage2/scripts/task_runner.py': '9b5ae812b0156ed5abb5cdfec11a35e96634e5d856a5a7ecd4d9520d5b2b9d91', 'triton_decode_attn_stage1/source/_fwd_kernel_stage1.py': '025e681f2a2a11be27b5384aca729841d04d1ada3ecfb0757a7f713c37e95c1a', 'triton_decode_attn_stage1/config.yaml': '793e6e10b6227e67ce8d50ca18f88ca87deb888631af904c5a4998bfea6768ba', 'triton_decode_attn_stage1/workloads.json': 'c487c2d3937c9dd56b2986cf6114ccc690842a4851541c4b723ba85527112d44', 'triton_decode_attn_stage1/scripts/task_runner.py': 'bdafdb2e9cdfb8abab438dc9f4486cbed58743358482637c9b4d2400f5f24ff2', 'triton_decode_attn_grouped_stage1/source/_fwd_grouped_kernel_stage1.py': '3f0d728ac011a7b9bd5eee1ae9d3eb69ef1ad91367918c4660c05a743cac7e4f', 'triton_decode_attn_grouped_stage1/config.yaml': '24c98bbcaff663b18709845a25e03d1608cf50e6068e65caacace55e37e47dbd', 'triton_decode_attn_grouped_stage1/workloads.json': 'd26c8ce33f535dac630c02e3ae55e187ec9917ff9d32e2b49553d04153802d29', 'triton_decode_attn_grouped_stage1/scripts/task_runner.py': 'd2dacb2807331c827edcbc97e8419ee17db41992470e6574be2262a9e4b425ad'}


def test_original_kernels_cases_gates_and_timers_preserved():
    for relative, digest in ORIGINALS.items():
        assert hashlib.sha256((TASKS / relative).read_bytes()).hexdigest() == digest
