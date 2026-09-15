"""Exercise the full task action plus installed guards on original/new domains.

Only device literals and the candidate implementation are CPU fixtures; the
actual public correctness dispatchers, oracles, guards and manifests execute.
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]


def load(path, name, cpu=False):
    tree = ast.parse(path.read_text())
    if cpu:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'cuda':
                node.value = 'cpu'
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(task, monkeypatch, fault=None):
    root = ROOT / 'tasks/triton2triton/vllm' / ('triton_' + task)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    controls = load(root / '_upstream_controls.py', 'cpu_controls', cpu=True)
    monkeypatch.setitem(sys.modules, '_upstream_controls', controls)
    harness = load(root / 'scripts/task_runner.py', 'cpu_harness', cpu=True)
    calls = []
    if task == 'expert_kernel':
        def public(A, B):
            calls.append((A.dtype, tuple(A.shape)))
            out = (A.double() @ B.double()).to(A.dtype)
            if fault == 'dtype': out = out.to(torch.float32)
            if fault == 'shape': out = out.flatten()
            if fault == 'zero': out.zero_()
            if fault == 'readonly': A.zero_()
            return out
        module = SimpleNamespace(expert_gemm=public)
        checks = load(root / '_arena_checks.py', 'expert_checks')
        harness.load_module = lambda: module
        checks.install(harness)
    else:
        def public(x, lengths, pad_value=-float('inf'), block_t=64, block_d=64):
            calls.append((x.dtype, tuple(x.shape)))
            sequences = torch.split(x, lengths.tolist())
            out = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,
                                                 padding_value=pad_value)
            if fault == 'dtype': out = out.to(torch.float64)
            if fault == 'shape': out = out.flatten()
            if fault == 'zero': out.zero_()
            if fault == 'readonly': x.zero_()
            return out
        module = SimpleNamespace(pack_seq=public)
        contract = load(root / '_arena_contract.py', 'pack_contract')
        replay = load(root / '_arena_replay.py', 'pack_replay', cpu=True)
        monkeypatch.setitem(sys.modules, '_arena_replay', replay)
        harness.load_module = lambda: SimpleNamespace(pack_seq=public)
        replay.install(harness, contract)
    adapter = load(root / '_arena_eval.py', 'task_action')
    adapter.load_harness = lambda: harness
    return adapter, harness, calls


@pytest.mark.parametrize('task,count', [('expert_kernel', 8), ('pack_seq', 9)])
@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_full_correctness_action_accepts_all_original_and_added_domains(task, count, role, monkeypatch):
    adapter, _, calls = fixture(task, monkeypatch)
    result = adapter.evaluate(role, 'correctness')
    assert result['status'] == 'PASS', result
    assert len(result['cases']) == count
    assert all(row['status'] == 'PASS' for row in result['cases'])
    assert sum('performance' in row['checks'] for row in result['cases']) == 5
    assert any(dtype == torch.bfloat16 for dtype, shape in calls)
    if task == 'pack_seq': assert any(len(shape) == 3 for dtype, shape in calls)


@pytest.mark.parametrize('task', ['expert_kernel', 'pack_seq'])
@pytest.mark.parametrize('case_index', [0, 10002])
@pytest.mark.parametrize('fault', ['dtype', 'shape', 'zero', 'readonly'])
def test_installed_original_and_new_control_reject_invalid_candidate(task, case_index, fault, monkeypatch):
    _, harness, calls = fixture(task, monkeypatch, fault)
    ok, error = harness.run_correctness(case_index=case_index)
    assert not ok and error, (task, case_index, fault)
    assert calls
    if fault == 'readonly': assert 'read-only' in error
    if fault in {'dtype', 'shape'}: assert 'shape/dtype/device' in error


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fault', [None, 'timed_dtype', 'replay_dtype', 'readonly'])
def test_expert_actual_timed_and_replay_guard_preserves_dtype_and_inputs(dtype, fault):
    checks = load(ROOT / 'tasks/triton2triton/vllm/triton_expert_kernel/_arena_checks.py', 'timed_expert')
    A = torch.arange(12, dtype=dtype).reshape(3, 4) / 16
    B = torch.arange(20, dtype=dtype).reshape(4, 5) / 32
    saved = (A.clone(), B.clone())
    calls = []
    def public(a, b):
        calls.append(1)
        value = (a.double() @ b.double()).to(dtype)
        if fault == 'timed_dtype' or (fault == 'replay_dtype' and len(calls) == 2):
            value = value.to(torch.float32)
        if fault == 'readonly': a.zero_()
        return value
    mod = SimpleNamespace(expert_gemm=public)
    def fn(): mod.expert_gemm(A, B)
    class Timed:
        def rerun(self): return self.fn()
    def benchmark(fn, timed_run, **options):
        timed_run.fn = fn
        timed_run.outputs = fn()
        return 0.1, {'cpu_fixture': True}
    harness = SimpleNamespace(_TimedRun=Timed)
    if fault is None:
        _, metadata = checks.checked_benchmark(harness, benchmark, fn)
        assert metadata['perturbed_input_replay_checked']
        assert len(calls) == 2
    else:
        with pytest.raises(AssertionError): checks.checked_benchmark(harness, benchmark, fn)
    assert torch.equal(A, saved[0]) and torch.equal(B, saved[1])
    assert mod.expert_gemm is public
