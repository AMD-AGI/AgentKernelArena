"""CPU stand-ins for the KDA gate's actual measured-buffer replay contract."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_kda_gate"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReusedOutput:
    def __init__(self, honest):
        self.honest = honest
        self.output = None
        self.computations = 0

    def fused_kda_gate(self, g, A, head_k_dim):
        del A, head_k_dim
        if self.output is None:
            self.output = g.detach().clone()
            self.output.fill_(float("nan"))
        if self.honest or not self.output.isfinite().all():
            self.output.copy_(g * 3)
            self.computations += 1
        return self.output


class CachedSeedAnswer:
    """Copies a correct warmup result without computing the measured input."""

    def __init__(self):
        self.output = None
        self.seed_input = None
        self.answer = None
        self.cache_copies = 0
        self.input_refs = []

    def fused_kda_gate(self, g, A, head_k_dim):
        del A, head_k_dim
        value = float(g.item())
        if not any(g is previous for previous, _ in self.input_refs):
            self.input_refs.append((g, value))
        if self.output is None:
            self.output = g.detach().clone()
        if self.seed_input is None or abs(value - self.seed_input) >= 1.0:
            self.seed_input = value
            self.answer = g * 3
        else:
            self.cache_copies += 1
        self.output.copy_(self.answer)
        return self.output


def fake_graph(fn, *, warmup, repetition, timed_run, prepare_fn, max_graph_repeats):
    assert warmup == 10 and repetition == 100 and max_graph_repeats == 1
    for _ in range(warmup):
        prepare_fn()
        fn()
    prepare_fn()
    fn()  # captured output storage
    prepare_fn()
    fn()  # prime
    for _ in range(repetition):
        prepare_fn()
        timed_run.outputs = fn()
        timed_run.after_sample(timed_run.outputs)
    timed_run.bound = True

    def rerun():
        prepare_fn()
        timed_run.outputs = fn()
        return timed_run.outputs

    timed_run.rerun = rerun
    return 0.01, {"benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1,
                  "benchmark_warmup": warmup, "benchmark_samples": repetition}


def run_cpu(monkeypatch, *, honest, timer=fake_graph, kernel_factory=None):
    torch = pytest.importorskip("torch")
    prior_cwd, prior_path = os.getcwd(), sys.path[:]
    prior_modules = {name: value for name, value in sys.modules.items()
                     if name == "scripts" or name.startswith("scripts.")}
    for name in prior_modules:
        sys.modules.pop(name)
    try:
        runner = load(TASK / "scripts/task_runner.py", "kda_gate_replay_runner")
        kernel = kernel_factory() if kernel_factory is not None else ReusedOutput(honest)
        monkeypatch.setattr(runner, "load_module", lambda: kernel)
        monkeypatch.setattr(runner, "gen_inputs", lambda seed, device: (
            (torch.tensor([float(seed)], dtype=torch.float32), torch.zeros(1), 1), {}))

        class CPUExpected:
            def __init__(self, value):
                self.value = value

            def to(self, device):
                assert device == "cuda"  # The production runner still requests GPU.
                return self.value

        monkeypatch.setattr(runner, "reference", lambda g, A, D: CPUExpected(g * 3))
        monkeypatch.setattr(runner, "_benchmark_cuda_graph_or_events", timer)
        monkeypatch.setitem(sys.modules, "_aka_benchmark",
                            SimpleNamespace(TimedRun=lambda: SimpleNamespace(bound=False, outputs=None)))
        return runner.run_performance(), kernel
    finally:
        os.chdir(prior_cwd)
        sys.path[:] = prior_path
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts."):
                sys.modules.pop(name)
        sys.modules.update(prior_modules)


def test_honest_kernel_has_100_checked_one_call_samples_and_unpoisoned_replay(monkeypatch):
    with monkeypatch.context() as patch:
        rows, kernel = run_cpu(patch, honest=True)
    assert len(rows) == 5
    assert all(row["timed_output_correctness"] == "PASS"
               and row["changed_input_replay_unpoisoned"] is True
               and row["input_state_restored"] is True
               and row["measured_samples_checked"] == 100
               and row["benchmark_effective_repeats"] == 1
               and row["captured_output_preparations"] >= 100
               for row in rows)
    assert kernel.computations >= 5 * 100


def test_valid_output_cache_is_rejected_by_unpoisoned_changed_input_replay(monkeypatch):
    torch = pytest.importorskip("torch")
    # Under the old control, poisoning just before the changed-input rerun
    # forced this cache to compute and made the wrong ordinary behavior pass.
    stale = ReusedOutput(honest=False)
    g, A = torch.tensor([2.]), torch.zeros(1)
    stale.fused_kda_gate(g, A, 1)
    g.add_(1)
    stale.output.fill_(float("nan"))
    assert torch.equal(stale.fused_kda_gate(g, A, 1), g * 3)
    with monkeypatch.context() as patch:
        rows, kernel = run_cpu(patch, honest=False)
    assert len(rows) == 5
    assert all("Output numerical mismatch" in row["error"] for row in rows)
    assert kernel.computations >= 5 * 100  # The measured poison required work.


def test_per_seed_cached_answer_copy_fails_on_distinct_measured_input(monkeypatch):
    with monkeypatch.context() as patch:
        rows, kernel = run_cpu(patch, honest=False, kernel_factory=CachedSeedAnswer)
    assert len(rows) == 5
    assert all('Output numerical mismatch' in row['error'] for row in rows)
    assert kernel.cache_copies > 0
    assert len(kernel.input_refs) == 5
    assert all(float(g.item()) == original for g, original in kernel.input_refs)


def test_stream_uses_original_first_sample_then_100_distinct_oracles(monkeypatch):
    torch = pytest.importorskip('torch')
    prior_cwd, prior_path = os.getcwd(), sys.path[:]
    prior_modules = {name: value for name, value in sys.modules.items()
                     if name == 'scripts' or name.startswith('scripts.')}
    for name in prior_modules:
        sys.modules.pop(name)
    try:
        runner = load(TASK / 'scripts/task_runner.py', 'kda_gate_stream_runner')
        g, A = torch.tensor([42.]), torch.zeros(1)
        monkeypatch.setattr(runner, 'reference', lambda g, A, D: (g * 3))
        reset = runner.MeasuredOutputReset()
        stream = runner.measured_input_stream((g, A, 1), {}, 42, 'cpu', reset)
        assert len(stream.variants) == len(stream.expected) == 100
        assert torch.equal(stream.variants[0][0], g)
        assert torch.equal(stream.variants[0][1], A)
        assert len({float(sample_g.item()) for sample_g, _ in stream.variants}) == 100
        for index in range(100):
            stream.prepare()
            assert stream.current_index == index
            stream.observe(g * 3, atol=1e-3, rtol=1e-3)
        assert stream.sample_index == 100
        stream.check_current()
    finally:
        os.chdir(prior_cwd)
        sys.path[:] = prior_path
        for name in list(sys.modules):
            if name == 'scripts' or name.startswith('scripts.'):
                sys.modules.pop(name)
        sys.modules.update(prior_modules)


@pytest.mark.parametrize("method,repeats", [("cuda_graph", 283), ("cuda_event_fallback", 1)])
def test_batching_or_event_path_cannot_claim_checked_graph_samples(method, repeats, monkeypatch):
    def wrong_method(*args, **kwargs):
        elapsed, metadata = fake_graph(*args, **kwargs)
        metadata.update(benchmark_method=method, benchmark_effective_repeats=repeats)
        return elapsed, metadata
    with monkeypatch.context() as patch:
        rows, _ = run_cpu(patch, honest=True, timer=wrong_method)
    assert len(rows) == 5
    assert all("Timed graph did not reset one captured output per sample" in row["error"]
               for row in rows)
