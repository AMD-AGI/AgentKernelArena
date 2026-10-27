"""CPU negative controls for recurrent and SSD task acceptance.

Torch doubles and simulated timing below test protected harness wiring only;
they do not qualify Triton execution or device timing on a GPU.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'
NAMES = ('triton_fla_fused_recurrent', 'triton_ssd_chunk_scan')


def _load(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=NAMES)
def task(request, monkeypatch):
    root = TASKS / request.param
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    adapter = _load(root / '_arena_eval.py', '_signal_adapter', monkeypatch)
    return root, adapter.load_harness(), adapter


@pytest.mark.parametrize('task', ['triton_fla_fused_recurrent'], indirect=True)
def test_recurrence_amplification_follows_linearity_for_every_seed(task):
    _, h, _ = task
    controls = h.semantic_controls
    for seed in h.SEEDS:
        args, kwargs = h.gen_inputs(seed, 'cpu', historical=True)
        original = controls.reference_outputs(*args, **kwargs)
        amplified = controls.reference_outputs(*controls.amplified_inputs(args), **kwargs)
        torch.testing.assert_close(amplified[0], original[0] * 64, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(amplified[1], original[1] * 8, atol=1e-6, rtol=1e-6)
        assert torch.allclose(torch.zeros_like(original[0]), original[0], atol=.05, rtol=.05)
        controls.require_signal(amplified)
        for value in amplified:
            assert not torch.allclose(torch.zeros_like(value), value, atol=.05, rtol=.05)
    with pytest.raises(controls.ContractFailure, match='accepts a zero fill'):
        controls.require_signal((torch.zeros(2), torch.ones(2)))


@pytest.mark.parametrize('task', ['triton_fla_fused_recurrent'], indirect=True)
@pytest.mark.parametrize('zero_member', [None, 0, 1])
def test_same_recurrent_shape_cannot_omit_output_or_history(task, monkeypatch, zero_member):
    _, h, _ = task
    def candidate(*args, **kwargs):
        values = h.semantic_controls.reference_outputs(*args, **kwargs)
        if args[0].shape == (1, 32, 2, 16) and zero_member is not None:
            values[zero_member].zero_()
        return values
    monkeypatch.setattr(h, 'load_module', lambda: SimpleNamespace(
        fused_recurrent_gated_delta_rule_fwd=candidate))
    if zero_member is None:
        for index in range(5):
            assert h.run_signal_controls(case_index=index, device='cpu')['status'] == 'PASS'
    else:
        with pytest.raises(h.semantic_controls.NumericalMismatch, match='numerical mismatch'):
            h.run_signal_controls(case_index=0, device='cpu')


@pytest.mark.parametrize('task', ['triton_ssd_chunk_scan'], indirect=True)
@pytest.mark.parametrize('omit_intra', [False, True])
def test_same_ssd_shapes_reject_omitting_intra_chunk(task, monkeypatch, omit_intra):
    _, h, _ = task
    controls = h.semantic_controls
    def candidate(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx):
        if omit_intra:
            value = controls.inter_chunk_reference(x, dA_cumsum, C, states, seq_idx, dt.shape[-1])
        else:
            value = h.reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, dt.shape[-1])
        out.copy_(value)
    monkeypatch.setattr(h, 'load_module', lambda: SimpleNamespace(chunk_scan_fwd=candidate))
    for index in range(5):
        if omit_intra:
            with pytest.raises(controls.NumericalMismatch, match='numerical mismatch'):
                h.run_signal_controls(case_index=index, device='cpu')
        else:
            assert h.run_signal_controls(case_index=index, device='cpu')['status'] == 'PASS'
    with pytest.raises(controls.ContractFailure, match='accepts omission'):
        controls.require_intra_signal(torch.zeros(2), torch.zeros(2))


def test_official_correctness_action_requires_unscored_signal_control(task, monkeypatch):
    _, h, adapter = task
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    monkeypatch.setattr(h, 'run_correctness', lambda **kwargs: (True, None))
    monkeypatch.setattr(h, 'run_semantic_controls', lambda: [], raising=False)
    seen = []
    def check(*, case_index):
        seen.append(case_index)
        if case_index == 0:
            raise AssertionError('shape-specific omitted computation')
        return {'status': 'PASS', 'scored': False}
    monkeypatch.setattr(h, 'run_signal_controls', check)
    result = adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL'
    assert result['cases'][0]['status'] == 'FAIL'
    assert 'shape-specific omitted computation' in result['cases'][0]['reason']
    assert seen == [row['params']['case_index'] for row in result['cases']]


def _cpu_cuda(monkeypatch):
    for name in ('randn', 'rand', 'arange', 'zeros', 'empty', 'full'):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)


@pytest.mark.parametrize('task', ['triton_fla_fused_recurrent', 'triton_ssd_chunk_scan'], indirect=True)
def test_real_performance_runner_rejects_omitted_work_on_current_draw_or_replay(task, monkeypatch):
    root, h, _ = task
    from src.tools.perf.aka_benchmark import TimedRun
    _cpu_cuda(monkeypatch)
    if root.name == 'triton_fla_fused_recurrent':
        def candidate(*args, **kwargs):
            output, states = h.semantic_controls.reference_outputs(*args, **kwargs)
            return torch.zeros_like(output), states
        mod = SimpleNamespace(fused_recurrent_gated_delta_rule_fwd=candidate)
    else:
        def candidate(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx):
            output = h.semantic_controls.inter_chunk_reference(
                x, dA_cumsum, C, states, seq_idx, dt.shape[-1])
            out.copy_(output)
        mod = SimpleNamespace(chunk_scan_fwd=candidate)
    monkeypatch.setattr(h, 'load_module', lambda: mod)
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=TimedRun))
    def fake_timing(fn, *, warmup, repetition, timed_run):
        assert (warmup, repetition) == (10, 100)
        output = fn()
        timed_run._bind(fn, output)
        return .125, {'benchmark_method': 'cuda_graph', 'cpu_fixture': True}
    monkeypatch.setattr(h, '_benchmark_cuda_graph_or_events', fake_timing)
    rows = h.run_performance()
    assert len(rows) == 5
    assert all(row['execution_time_ms'] < 0 and 'mismatch' in row['error'] for row in rows)
