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
            if x.shape[0] == 0:
                if fault == 'empty_dtype': out = out.to(torch.float64)
                if fault == 'empty_shape': out = out.flatten()
                if fault == 'empty_rejected': raise RuntimeError('unsupported empty sequence')
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


@pytest.mark.parametrize('task,count', [('expert_kernel', 8), ('pack_seq', 11)])
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

@pytest.mark.parametrize('case_index', [10003, 10004])
@pytest.mark.parametrize('fault', ['empty_dtype', 'empty_shape', 'empty_rejected'])
def test_pack_all_empty_candidate_is_checked_by_installed_full_action(case_index, fault, monkeypatch):
    adapter, harness, calls = fixture('pack_seq', monkeypatch, fault)
    result = adapter.evaluate('candidate', 'correctness')
    by_index = {row['params']['case_index']: row for row in result['cases']}
    assert result['status'] == 'FAIL'
    assert by_index[case_index]['status'] == 'FAIL', result
    assert all(by_index[i]['status'] == 'PASS' for i in range(5))
    assert all(by_index[i]['status'] == 'PASS' for i in [5, 10000, 10001, 10002])
    assert any(shape[0] == 0 for _, shape in calls)


def test_pack_all_empty_manifest_is_additive_and_unscored():
    import json, subprocess
    root = ROOT / 'tasks/triton2triton/vllm/triton_pack_seq'
    before = json.loads(subprocess.check_output(
        ['git', 'show', '400fcb9d:' + str(root.relative_to(ROOT) / 'workloads.json')], cwd=ROOT))
    after = json.loads((root / 'workloads.json').read_text())
    assert after['cases'][:len(before['cases'])] == before['cases']
    assert after['upstream_controls'][:3] == before['upstream_controls']
    extra = after['cases'][len(before['cases']):]
    assert len(extra) == 2
    assert all(row['checks'] == ['correctness'] for row in extra)
    assert all(set(row['params']['configuration']['lengths']) == {0} for row in extra)
    assert len([row for row in after['cases'] if 'performance' in row['checks']]) == 5
    for name in ['scripts/task_runner.py', '_arena_contract.py', '_arena_replay.py', '_arena_eval.py', 'config.yaml']:
        assert (root / name).read_bytes() == subprocess.check_output(
            ['git', 'show', '400fcb9d:' + str(root.relative_to(ROOT) / name)], cwd=ROOT)

@pytest.mark.parametrize('shape,dtype,lengths', [
    ((0, 65), torch.float16, [0, 0, 0]),
    ((0, 3, 5), torch.bfloat16, [0, 0]),
    ((4, 3, 5), torch.float16, [1, 3]),
])
def test_pack_actual_wrapper_preserves_native_launch_for_empty_and_nonempty(shape, dtype, lengths, monkeypatch):
    # Execute the real public wrapper, intercepting only the GPU launch. The
    # separate real GPU probe verifies zero-grid support; this is CPU logic.
    source = ROOT / 'tasks/triton2triton/vllm/triton_pack_seq/source/triton_pack_seq.py'
    launches = []
    class Kernel:
        def __init__(self, fn): self.fn = fn
        def __getitem__(self, grid):
            def launch(x, out, lens, N, D, Lmax, **kw):
                launches.append((grid, tuple(x.shape), N, D, Lmax, kw))
                # Independent deterministic packing fixture, not the task oracle.
                out.fill_(kw['PAD_VALUE'])
                start = 0
                for b, length in enumerate(lens.tolist()):
                    out[b, :length].copy_(x[start:start+length]); start += length
            return launch
    triton = ModuleType('triton');triton.jit = Kernel
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = ModuleType('triton.language');tl.constexpr = object()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', tl)
    module = load(source, 'real_pack_wrapper')
    x = torch.arange(__import__('math').prod(shape), dtype=dtype).reshape(shape)
    lens = torch.tensor(lengths, dtype=torch.int32)
    output = module.pack_seq(x, lens, pad_value=2.5)
    expected = torch.nn.utils.rnn.pad_sequence(torch.split(x, lengths), batch_first=True, padding_value=2.5)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.shape == (len(lengths), max(lengths), *shape[1:])
    assert output.dtype == dtype and output.device == x.device
    assert len(launches) == 1  # No early return, delegation or bypass.
    grid, flat_shape, N, D, Lmax, kw = launches[0]
    assert grid == (len(lengths), triton.cdiv(max(lengths), 64), triton.cdiv(__import__('math').prod(shape[1:]), 64))
    assert flat_shape == (shape[0], __import__('math').prod(shape[1:]))
    assert kw == {'PAD_VALUE': 2.5, 'BLOCK_T': 64, 'BLOCK_D': 64, 'num_warps': 4, 'num_stages': 2}
