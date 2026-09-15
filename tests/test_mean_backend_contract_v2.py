"""CPU dispatch/launcher doubles test enforcement, not actual GPU execution.

The public GPU task validator separately qualifies the real Triton runtime.
"""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_mean'


def module_at(path):
    spec = importlib.util.spec_from_file_location('_mean_cpu_fixture_' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def checks(monkeypatch):
    monkeypatch.chdir(ROOT)
    return module_at(TASK / '_arena_checks.py')


@pytest.fixture
def cpu_runtime(checks, monkeypatch):
    """Explicit fake trusted runtime; kernel copying is not a Triton proof."""
    runtime = SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)
    compiler = ModuleType('cpu_fixture_compiler')
    class CompiledKernel:
        def __init__(self, fn):
            self.src = SimpleNamespace(fn=fn)
    compiler.CompiledKernel = CompiledKernel

    class JITFunction:
        def __init__(self):
            self.expected = None
            self.calls = 0

        def run(self, x, output, *, grid=(1,), warmup=False):
            kernel = CompiledKernel(self)
            if not warmup:
                runtime.launch_enter_hook(None)
                output.copy_(self.expected)
                self.calls += 1
                runtime.launch_exit_hook(None)
            return kernel

    class Wrapper:
        def __init__(self, fn):
            self.fn = fn

    monkeypatch.setattr(checks, '_triton_runtime', lambda: (runtime, compiler, JITFunction, (Wrapper,)))
    return runtime, compiler, JITFunction, Wrapper


def candidate_module(jit):
    def mean_dim(x, dim, keepdim=False, dtype=None):
        dtype = dtype or x.dtype
        x = x.to(dtype)
        shape = list(x.shape)
        dim %= x.ndim
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        output = torch.empty(shape, dtype=dtype, device=x.device)
        jit.run(x, output)
        return output
    return SimpleNamespace(mean_dim=mean_dim, mean_kernel=jit)


@pytest.mark.parametrize('compute', [
    lambda x: torch.mean(x, dim=-1),
    lambda x: x.mean(dim=-1),
    lambda x: x.sum(dim=-1) / x.shape[-1],
    lambda x: torch.matmul(x, x.t()),
])
def test_unused_declared_jit_cannot_hide_torch_compute(checks, cpu_runtime, compute):
    _, _, JIT, _ = cpu_runtime
    mod = candidate_module(JIT())
    x = torch.arange(6.).reshape(2, 3)
    with pytest.raises(AssertionError, match='non-preparation PyTorch operation'):
        checks.checked_candidate_call(mod, compute, x)


def test_fake_same_named_kernel_is_rejected(checks, cpu_runtime):
    FakeJIT = type('JITFunction', (), {'__module__': 'triton.runtime.jit'})
    mod = SimpleNamespace(mean_kernel=FakeJIT())
    with pytest.raises(AssertionError, match='genuine Triton JITFunction'):
        checks.checked_candidate_call(mod, lambda: torch.empty(2))


@pytest.mark.parametrize('mode', ['unused', 'warmup_only', 'other_kernel', 'forged_hook'])
def test_only_actual_declared_launch_counts(checks, cpu_runtime, mode):
    runtime, _, JIT, _ = cpu_runtime
    jit = JIT()
    mod = candidate_module(jit)
    x = torch.arange(6.).reshape(2, 3)
    other = JIT()
    other.expected = torch.tensor([1., 4.])
    def shortcut():
        output = torch.empty(2)
        if mode == 'warmup_only': jit.run(x, output, warmup=True)
        elif mode == 'other_kernel': other.run(x, output)
        elif mode == 'forged_hook':
            runtime.launch_enter_hook(None)
            runtime.launch_exit_hook(None)
        return output
    with pytest.raises(AssertionError, match='No genuine declared mean_kernel Triton launch'):
        checks.checked_candidate_call(mod, shortcut)
    assert runtime.launch_enter_hook is runtime.launch_exit_hook is None


def test_preparation_and_hook_restoration_with_cpu_runtime_double(checks, cpu_runtime):
    runtime, _, JIT, Wrapper = cpu_runtime
    jit = JIT()
    mod = candidate_module(jit)
    mod.mean_kernel = Wrapper(jit)
    x = torch.arange(6., dtype=torch.float16).reshape(2, 3)
    jit.expected = torch.tensor([[1.], [4.]])
    previous = []
    enter, leave = lambda m: previous.append('enter'), lambda m: previous.append('leave')
    runtime.launch_enter_hook, runtime.launch_exit_hook = enter, leave
    output = checks.checked_candidate_call(mod, mod.mean_dim, x, -1, keepdim=True, dtype=torch.float32)
    torch.testing.assert_close(output, jit.expected)
    assert previous == ['enter', 'leave'] and jit.calls == 1
    assert runtime.launch_enter_hook is enter and runtime.launch_exit_hook is leave
    with pytest.raises(ValueError, match='candidate failed'):
        checks.checked_candidate_call(mod, lambda: (_ for _ in ()).throw(ValueError('candidate failed')))
    assert runtime.launch_enter_hook is enter and runtime.launch_exit_hook is leave


def test_hook_replacement_fails_and_is_restored(checks, cpu_runtime):
    runtime, _, JIT, _ = cpu_runtime
    jit = JIT()
    jit.expected = torch.tensor([1., 4.])
    mod = candidate_module(jit)
    x = torch.arange(6.).reshape(2, 3)
    def changed():
        output = mod.mean_dim(x, -1)
        runtime.launch_exit_hook = None
        return output
    with pytest.raises(AssertionError, match='changed the runtime launch audit'):
        checks.checked_candidate_call(mod, changed)
    assert runtime.launch_enter_hook is runtime.launch_exit_hook is None


@pytest.mark.parametrize('shortcut', [False, True])
def test_real_checked_modules_integration_keeps_oracle_outside_guard(checks, cpu_runtime, shortcut):
    _, _, JIT, _ = cpu_runtime
    jit = JIT()
    mod = candidate_module(jit)
    original = mod.mean_dim
    x = torch.arange(30.).reshape(2, 3, 5)
    jit.expected = checks.reference(x, -1, True, torch.float32)
    if shortcut:
        mod.mean_dim = lambda x, dim, keepdim=False, dtype=None: x.to(dtype or x.dtype).mean(dim, keepdim)
    candidate = mod.mean_dim
    h = SimpleNamespace(load_module=lambda: mod)
    with checks.checked_modules(h):
        if shortcut:
            with pytest.raises(AssertionError, match='non-preparation PyTorch operation'):
                h.load_module().mean_dim(x, -1, True, torch.float32)
        else:
            output = h.load_module().mean_dim(x, -1, True, torch.float32)
            torch.testing.assert_close(output, jit.expected)
            assert jit.calls == 2  # Existing diagnostic plus actual correctness call.
    assert mod.mean_dim is candidate
    assert shortcut or candidate is original


def test_install_guards_initializer_compute(checks):
    x = torch.arange(6.).reshape(2, 3)
    h = SimpleNamespace(load_module=lambda: SimpleNamespace(cached=x.mean(-1)),
                        run_correctness=lambda: None, run_performance=lambda: None)
    checks.install(h)
    with pytest.raises(AssertionError, match='non-preparation PyTorch operation'):
        h.load_module()


@pytest.mark.parametrize('shortcut_phase', [None, 'warmup', 'capture'])
def test_checked_benchmark_guards_actual_warmup_capture_and_replay(checks, cpu_runtime, shortcut_phase):
    _, _, JIT, _ = cpu_runtime
    jit = JIT()
    mod = candidate_module(jit)
    phase = None
    genuine = mod.mean_dim
    def candidate(x, dim):
        if phase == shortcut_phase and shortcut_phase is not None:
            return x.mean(dim=dim)
        return genuine(x, dim)
    mod.mean_dim = candidate
    x, dim = torch.arange(6.).reshape(2, 3), -1
    pristine = x.clone()
    def fn():
        mod.mean_dim(x, dim)
    calls = []
    def benchmark(measured, *, timed_run, **options):
        nonlocal phase
        assert options == {'warmup': 10, 'repetition': 100}
        for phase in ('warmup', 'capture'):
            jit.expected = checks.reference(x, dim)
            calls.append(phase)
            output = measured()
        def replay():
            nonlocal phase
            phase = 'replay'
            jit.expected = checks.reference(x, dim)
            output.copy_(measured())
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h = SimpleNamespace(_TimedRun=SimpleNamespace)
    if shortcut_phase is not None:
        with pytest.raises(AssertionError, match='non-preparation PyTorch operation'):
            checks.checked_benchmark(h, benchmark, fn, warmup=10, repetition=100)
    else:
        ms, metadata = checks.checked_benchmark(h, benchmark, fn, warmup=10, repetition=100)
        assert ms == .125 and metadata['triton_wrapper_dispatch_checked']
        assert calls == ['warmup', 'capture'] and jit.calls == 3
    torch.testing.assert_close(x, pristine)
    assert mod.mean_dim is candidate


def test_original_kernel_cases_sampling_and_timing_helpers_are_unchanged():
    import hashlib
    # Reviewed starting bytes; source archives do not need git history to test.
    originals = {
        'source/triton_mean.py': '81588025a1bbf608001711b3932a3b07abfdf0ad0143313daccadbcfb63d5eac',
        'config.yaml': 'c3f57fb29709e2fb9c7f5c6dce970f8ac0d474ac143be383dc21b74e5dbb26a4',
        'workloads.json': '5706c46f716076ca5866ef7f50ede588be02619c0aecc1536b991204262f8f8a',
        'scripts/task_runner.py': '6b7b6b52db74bed6ecbdd49bda605b4f6a416e24d40df3586ba1559aedb1a632',
    }
    for name, digest in originals.items():
        assert hashlib.sha256((TASK / name).read_bytes()).hexdigest() == digest
