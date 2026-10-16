"""Task-owned candidate policy regressions; CPU fixtures do not qualify GPU code."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
TASKS = ['triton2flydsl/aiter/rope_fwd', 'triton2flydsl/generative_recommenders/layer_norm', 'torch2flydsl/rope_2d_fwd_kernel']

def checks(task):
    return load(ROOT / 'tasks' / task / 'scripts/candidate_checks.py', '_policy_' + task.replace('/', '_'))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def fake_fly_kernel():
    namespace = {'__name__': 'flydsl.compiler._cpu_profile_fixture', 'torch': torch}
    exec('class FakeCompiledKernel:\n def __call__(self,x):\n  return x.clone()\n', namespace)
    return namespace['FakeCompiledKernel']()

def harness(task, target, benchmark, *, baseline=False):
    h = SimpleNamespace(ARENA_FINAL_CANDIDATE=not baseline, ARENA_PROVIDED_BASELINE=baseline, ENTRY='op', ENTRIES=('op',), KERNEL_FILE='kernel.py', benchmark_cuda_graph_or_events=benchmark)
    source = lambda: SimpleNamespace(op=target, flydsl_op=target)
    h.load_module = lambda: source()
    h._load_source = lambda: source()
    h._load_module = lambda kernel_dir, filename, alias: source()
    return h

def get_candidate(h, task):
    if task.startswith('torch2flydsl/'):
        return h._load_module('.', h.KERNEL_FILE, 'candidate').flydsl_op
    return (h._load_source() if '/aiter/' in task else h.load_module()).op

@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('switch_after', [1, 7, 31])
def test_late_PyTorch_math_on_actual_benchmark_callback_rejected(task, switch_after):
    c = checks(task)
    fly = fake_fly_kernel()
    state = {'calls': 0, 'inside': False}
    x = torch.arange(8, dtype=torch.float32)

    def target(x):
        state['calls'] += 1
        if state['calls'] > switch_after:
            return x + 0
        return fly(x)

    def benchmark(fn, *args, **kwargs):
        state['inside'] = True
        try:
            for _ in range(40):
                fn()
        finally:
            state['inside'] = False
    h = harness(task, target, benchmark)
    original = h.benchmark_cuda_graph_or_events
    with pytest.raises(RuntimeError, match='non-preparation PyTorch operation aten::add'):
        with c.audit_candidate_calls(h, performance=True):
            fn = get_candidate(h, task)
            h.benchmark_cuda_graph_or_events(lambda: fn(x), warmup=10, repetition=100)
    assert state['calls'] == switch_after + 1
    assert h.benchmark_cuda_graph_or_events is original

@pytest.mark.parametrize('task', TASKS)
def test_eager_replay_late_delegate_is_still_guarded(task):
    c = checks(task)
    fly = fake_fly_kernel()
    state = {'calls': 0}
    x = torch.ones(8)

    def target(x):
        state['calls'] += 1
        return fly(x) if state['calls'] <= 4 else x.sum().expand_as(x)

    def benchmark(fn, **kwargs):
        for _ in range(3):
            fn()
        return fn
    h = harness(task, target, benchmark)
    with pytest.raises(RuntimeError, match='aten::sum'):
        with c.audit_candidate_calls(h, performance=True):
            fn = get_candidate(h, task)
            replay = h.benchmark_cuda_graph_or_events(lambda: fn(x))
            assert state['calls'] == 4
            try:
                replay()
            except RuntimeError:
                pass

@pytest.mark.parametrize('task', TASKS)
def test_legal_preparation_and_Fly_fixture_no_profile_inside_timer(task, monkeypatch):
    c = checks(task)
    fly = fake_fly_kernel()
    x = torch.ones(2, 4)
    inside = {'value': False}
    original_profile = sys.setprofile
    outer = sys.getprofile()
    calls = []
    prepared = []

    def profile(fn):
        assert not inside['value'], 'profiling must stay outside the canonical clock/capture'
        original_profile(fn)
    monkeypatch.setattr(sys, 'setprofile', profile)

    def target(x):
        y = torch.empty_like(x).view(-1).view_as(x)
        y.copy_(x)
        return fly(y)

    def benchmark(fn, *args, **kwargs):
        calls.append((args, kwargs))
        inside['value'] = True
        try:
            assert sys.getprofile() is outer
            for _ in range(6):
                assert torch.equal(fn(), x)
            return (0.125, {'benchmark_samples': 100, 'benchmark_warmup': 10})
        finally:
            inside['value'] = False
    h = harness(task, target, benchmark)
    original = h.benchmark_cuda_graph_or_events
    with c.audit_candidate_calls(h, performance=True) as observed:
        fn = get_candidate(h, task)
        sentinel = object()
        prepare = lambda: prepared.append('restored')
        result = h.benchmark_cuda_graph_or_events(lambda: fn(x), warmup=10, repetition=100, prepare_fn=prepare, timed_run=sentinel)
        assert observed
    assert result == (0.125, {'benchmark_samples': 100, 'benchmark_warmup': 10})
    assert calls[0][1] == {'warmup': 10, 'repetition': 100, 'prepare_fn': prepare, 'timed_run': sentinel}
    assert prepared == ['restored', 'restored']
    assert h.benchmark_cuda_graph_or_events is original

@pytest.mark.parametrize('task', TASKS)
def test_baseline_policy_loader_timer_and_Torch_math_untouched(task):
    c = checks(task)
    x = torch.ones(8)
    h = harness(task, lambda x: x + 1, lambda fn, **kwargs: fn(), baseline=True)
    timer = h.benchmark_cuda_graph_or_events
    with c.audit_candidate_calls(h, performance=True) as observed:
        assert h.benchmark_cuda_graph_or_events is timer
        assert torch.equal(get_candidate(h, task)(x), x + 1)
    assert not observed
REGISTRY_BAD = ['import sys\ndef op(): return sys.modules["arena_harness"]', 'import sys as registry\ndef op(): return registry.modules', 'from sys import modules as registry\ndef op(): return registry["arena_harness"]', 'import sys\na=sys\nb=a\ndef op(): return b.modules', 'import sys\ndef op(): return getattr(sys,"mod"+"ules")', 'import sys\ndef op(): return vars(sys)["modules"]', 'import sys\ndef op(): return sys.__dict__["modules"]', 'import sys\ndef op(): return sys._getframe()', 'import sys\ndef op(): return sys.setprofile(None)', 'import sys\nfrom builtins import getattr as fetch\ndef op(): return fetch(sys,"modules")', 'import sys\ndef op(): return (lambda:sys)().modules', 'import sys\nregistry=(sys,)\ndef op(): return registry[0].modules']

@pytest.mark.parametrize('task', TASKS)
def test_sys_safe_metadata_and_installed_library_internal_sys_allowed(task, tmp_path):
    entry = tmp_path / 'candidate.py'
    entry.write_text('import flydsl\nimport sys\nVERSION=sys.version_info\ndef op(): return sys.platform\n')
    after = load(ROOT / 'tasks' / task / 'task_runtime.py', '_safe_runtime')
    after.ROOT = tmp_path
    assert after.check_dependencies([entry], True) == [entry]

@pytest.mark.parametrize('task', TASKS)
def test_local_helper_cannot_hide_registry_access(task, tmp_path):
    entry = tmp_path / 'candidate.py'
    entry.write_text('import flydsl\nfrom helper import op\n')
    (tmp_path / 'helper.py').write_text('import sys as registry\ndef op(): return registry.modules\n')
    after = load(ROOT / 'tasks' / task / 'task_runtime.py', '_closure_runtime')
    after.ROOT = tmp_path
    with pytest.raises(ValueError, match='module registry'):
        after.check_dependencies([entry], True)

def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

@pytest.mark.parametrize('task', TASKS)
def test_actual_task_actions_performance_entry_enables_late_callback_policy(task, monkeypatch):
    cp = checks(task)
    x = torch.ones(8)
    fly = fake_fly_kernel()
    state = {'calls': 0}

    def target(x):
        state['calls'] += 1
        return fly(x) if state['calls'] == 1 else x + 0
    h = harness(task, target, lambda fn, **kwargs: fn())

    def performance(*args, **kwargs):
        fn = get_candidate(h, task)
        return h.benchmark_cuda_graph_or_events(lambda: fn(x))
    h.run_performance = performance
    h.arena_benchmark = performance
    h.run_benchmark = performance
    package = ModuleType('scripts')
    package.__path__ = [str(ROOT / 'tasks' / task / 'scripts')]
    monkeypatch.setitem(sys.modules, 'scripts', package)
    monkeypatch.setitem(sys.modules, 'scripts.candidate_checks', cp)
    actions = load(ROOT / 'tasks' / task / 'scripts/task_actions.py', '_actual_actions')
    with pytest.raises(RuntimeError, match='aten::add'):
        actions.performance(h)
    assert state['calls'] == 2

@pytest.mark.parametrize('task', TASKS)
def test_capture_only_delegate_sticky_even_if_timer_suppresses_and_event_fallback_succeeds(task):
    c = checks(task)
    fly = fake_fly_kernel()
    x = torch.ones(8)
    capture = {'active': False, 'caught': False, 'fallback': False}

    def target(x):
        out = fly(x)
        return out + 0 if capture['active'] else out

    def benchmark(fn, **kwargs):
        capture['active'] = True
        try:
            fn()
        except RuntimeError:
            capture['caught'] = True
        finally:
            capture['active'] = False
        fallback = fn()
        capture['fallback'] = True
        assert torch.equal(fallback, x)
        return (0.1, {'benchmark_method': 'cuda_event_fallback', 'benchmark_samples': 100})
    h = harness(task, target, benchmark)
    with pytest.raises(RuntimeError, match='aten::add'):
        with c.audit_candidate_calls(h, performance=True):
            fn = get_candidate(h, task)
            h.benchmark_cuda_graph_or_events(lambda: fn(x))
    assert capture == {'active': False, 'caught': True, 'fallback': True}

@pytest.mark.parametrize('task', TASKS)
def test_candidate_cannot_swallow_operation_violation_and_return_legal_output(task):
    c = checks(task)
    fly = fake_fly_kernel()
    x = torch.ones(8)

    def target(x):
        try:
            x + 0
        except RuntimeError:
            pass
        return fly(x)
    h = harness(task, target, lambda fn, **kwargs: fn())
    with pytest.raises(RuntimeError, match='aten::add'):
        with c.audit_candidate_calls(h, performance=True):
            fn = get_candidate(h, task)
            h.benchmark_cuda_graph_or_events(lambda: fn(x))

@pytest.mark.parametrize('task', TASKS)
def test_legitimate_capture_fallback_policy_metadata_unchanged(task):
    c = checks(task)
    fly = fake_fly_kernel()
    x = torch.ones(8)

    def benchmark(fn, **kwargs):
        try:
            raise RuntimeError('legitimate unsupported graph capture')
        except RuntimeError:
            pass
        assert torch.equal(fn(), x)
        return (0.1, {'benchmark_method': 'cuda_event_fallback', 'benchmark_samples': 100, 'benchmark_warmup': 10})
    h = harness(task, lambda x: fly(x), benchmark)
    with c.audit_candidate_calls(h, performance=True):
        fn = get_candidate(h, task)
        result = h.benchmark_cuda_graph_or_events(lambda: fn(x))
    assert result == (0.1, {'benchmark_method': 'cuda_event_fallback', 'benchmark_samples': 100, 'benchmark_warmup': 10})

@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('source', REGISTRY_BAD)
def test_candidate_registry_and_control_frame_routes_rejected_before_import(task, source, tmp_path):
    entry = tmp_path / 'candidate.py'
    entry.write_text('import flydsl\n' + source + '\n')
    runtime = load(ROOT / 'tasks' / task / 'task_runtime.py', '_registry_policy')
    runtime.ROOT = tmp_path
    with pytest.raises(ValueError, match='module registry|control frames|Dynamic sys'):
        runtime.check_dependencies([entry], True)
    runtime.check_dependencies([entry], False)
