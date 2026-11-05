"""Eviction/Event ordering, observed replay, paired statistics and cache identity."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch

from src.task_protocol import RESULT_PREFIX, TaskProtocolError, parse_command_result, performance_cases
from src.testcases import (TestCaseResult as Case, analyze_benchmark_method_consistency,
                           calculate_average_speedup)

HELPER = Path(__file__).resolve().parents[1] / "src/tools/perf/aka_benchmark.py"


@pytest.fixture
def helper(monkeypatch):
    trace = []

    class Event:
        index = 0

        def __init__(self, **kwargs):
            self.name = "start" if Event.index % 2 == 0 else "end"
            Event.index += 1

        def record(self, stream=None):
            trace.append(self.name)

        def synchronize(self):
            trace.append("wait")

        def elapsed_time(self, end):
            return 0.01

    cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0,
        Event=Event, synchronize=lambda: None,
        Stream=lambda: types.SimpleNamespace(wait_stream=lambda s: None, synchronize=lambda: None),
        current_stream=lambda: object(), stream=lambda s: nullcontext(),
    )
    fake = types.ModuleType("torch")
    fake.__getattr__ = lambda name: getattr(torch, name)
    fake.cuda, fake.device, fake.int32 = cuda, torch.device, torch.int32
    # Exercise the real CPU reduction while replacing GPU allocation and Events.
    fake.zeros = lambda shape, dtype, device: torch.zeros(shape, dtype=dtype)
    fake.empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype)

    def sum_(value, **kwargs):
        trace.append("flush")
        return torch.sum(value, **kwargs)

    fake.sum = sum_
    monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_cold_cache_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.trace = trace
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT", raising=False)
    return module


def replaying_capture(helper, monkeypatch):
    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        assert repeats == 1
        prepare_fn()
        output = fn()
        if output_holder is not None:
            output_holder[:] = [output]

        def replay():
            helper.trace.append("replay")
            fn()

        return types.SimpleNamespace(replay=replay)

    monkeypatch.setattr(helper, "_capture_graph", capture)


def test_sweep_reads_initialized_persistent_buffer_without_overwriting_it(helper):
    eviction = helper.ReadOnlyCacheFlush(16384)
    assert torch.count_nonzero(eviction._cache) == 0
    eviction._cache.fill_(3)
    pointers = eviction._cache.data_ptr(), eviction._sums.data_ptr()
    for _ in range(2):
        eviction()
        assert torch.equal(eviction._sums, torch.tensor([12288], dtype=torch.int32))
        assert torch.all(eviction._cache == 3)
        assert (eviction._cache.data_ptr(), eviction._sums.data_ptr()) == pointers


@pytest.mark.parametrize("size", [0, -1, 1024, True, 16384.0])
def test_invalid_sweep_size_is_rejected(helper, size):
    with pytest.raises(ValueError, match="positive multiple"):
        helper.ReadOnlyCacheFlush(size)


def test_every_sample_and_timed_rerun_evict_after_prepare_before_start(helper, monkeypatch):
    replaying_capture(helper, monkeypatch)
    eviction = helper.ReadOnlyCacheFlush(16384)
    observed = []
    timed = helper.TimedRun()
    timed.after_sample = observed.append

    def operation():
        helper.trace.append("kernel")
        return "captured output"

    samples, metadata = helper.benchmark_cuda_graph_cold_cache_samples(
        operation, cache_flush=eviction, warmup=1, repetition=3,
        prepare_fn=lambda: helper.trace.append("prepare"), timed_run=timed)
    assert samples == [0.01] * 3
    assert observed == ["captured output"] * 3
    assert metadata["benchmark_method"] == "cuda_graph_cold_cache"
    assert metadata["benchmark_cache_flush_bytes"] == 16384
    assert metadata["benchmark_cache_flush_in_timing"] is False
    assert metadata["benchmark_effective_repeats"] == 1
    assert timed.rerun() == "captured output"
    assert timed.rerun_ms() == 0.01
    for index, name in enumerate(helper.trace):
        if name == "start":
            assert helper.trace[index - 2:index] == ["prepare", "flush"]
            assert helper.trace[index:index + 5] == ["start", "replay", "kernel", "end", "wait"]


def test_cold_capture_cannot_fallback_even_without_caller_collector(helper, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("capture failure")

    monkeypatch.setattr(helper, "_capture_graph", fail)
    with pytest.raises(RuntimeError, match="cannot validate"):
        helper.benchmark_cuda_graph_cold_cache_samples(
            lambda: None, cache_flush=helper.ReadOnlyCacheFlush(16384), warmup=0)


def test_forced_events_device_mismatch_and_cache_warming_observer_are_rejected(helper, monkeypatch):
    flush = helper.ReadOnlyCacheFlush(16384)
    timed = helper.TimedRun()
    timed._bind(lambda: 1, 1)
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="Event timing is forbidden"):
        helper.benchmark_cuda_graph_cold_cache_samples(lambda: None, cache_flush=flush, timed_run=timed)
    assert not timed.bound and timed.outputs is None
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT")
    flush.device = torch.device("cuda:1")
    with pytest.raises(ValueError, match="same current device"):
        helper.benchmark_cuda_graph_cold_cache_samples(lambda: None, cache_flush=flush)
    flush.device = torch.device("cuda:0")
    timed.before_sample = lambda calls: None
    with pytest.raises(ValueError, match="before cache eviction"):
        helper.benchmark_cuda_graph_cold_cache_samples(lambda: None, cache_flush=flush, timed_run=timed)


def test_paired_rounds_balance_order_keep_raw_samples_and_check_measured_outputs(helper, monkeypatch):
    checked, resets = [], []

    def baseline():
        return "baseline output"

    def candidate():
        return "candidate output"

    def measure(fn, *, timed_run, repetition, **kwargs):
        output = fn()
        for _ in range(repetition):
            timed_run.after_sample(output)
        timed_run._bind(fn, output, lambda: (output, 0.25))
        return [2.0 if fn is baseline else 1.0] * repetition, {"benchmark_method": "cuda_graph_cold_cache"}

    monkeypatch.setattr(helper, "benchmark_cuda_graph_cold_cache_samples", measure)
    result = helper.benchmark_paired_cuda_graph_cold_cache(
        baseline, candidate, cache_flush=object(), rounds=4, repetition=3, seed=7,
        check_fn=lambda role, output: checked.append((role, output)),
        before_arm=lambda role, index: resets.append((role, index)))
    records = result["measurements"]
    assert sum(row["order"][0] == "baseline" for row in records) == 2
    assert len(resets) == 8
    assert all(output == role + " output" for role, output in checked)
    assert len(checked) == 8 * (3 + 3)  # samples, last output, rerun, timed rerun
    assert all(row["baseline_samples_ms"] == [2.0] * 3 for row in records)
    assert result["summary"]["speedup_geomean"] == pytest.approx(2)
    assert result["summary"]["speedup_ci95"] == pytest.approx([2, 2])


def test_bootstrap_preserves_pairing_and_rejects_invalid_evidence(helper):
    # Baseline/candidate drift together; ratios remain exactly two.
    rows = [{"baseline_ms": t * 2, "candidate_ms": t} for t in (1, 10, 100, 1000)]
    summary = helper.summarize_paired_speedup(rows, seed=9)
    assert summary["speedup_ci95"] == pytest.approx([2, 2])
    assert summary == helper.summarize_paired_speedup(rows, seed=9)
    with pytest.raises(ValueError, match="four"):
        helper.summarize_paired_speedup(rows[:3])
    rows[0]["candidate_ms"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        helper.summarize_paired_speedup(rows)


def cold_metadata():
    return {
        "benchmark_method": "cuda_graph_cold_cache",
        "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
        "benchmark_cache_flush_in_timing": False,
        "benchmark_effective_repeats": 1,
        "benchmark_metric": "gpu_event_graph_span",
        "benchmark_cache_flush_bytes": 1 << 30,
    }


def test_protocol_retains_cache_identity_and_matches_only_identical_cold_policy():
    metadata = cold_metadata()
    row = {"test_case_id": "a", "shape": [1], "status": "PASS",
           "execution_time_ms": 2.0, "benchmark_method": "cuda_graph_cold_cache",
           "metadata": metadata}
    report = {"protocol": "arena-eval-v1", "role": "candidate", "action": "performance",
              "status": "PASS", "cases": [row]}
    parsed = parse_command_result(RESULT_PREFIX + json.dumps(report), role="candidate",
                                  action="performance", returncode=0)
    baseline = performance_cases(parsed)
    candidate = [Case("a", [1], 1.0, metadata.copy())]
    assert analyze_benchmark_method_consistency(baseline, candidate) == (True, [])
    assert calculate_average_speedup(baseline, candidate) == pytest.approx(2)
    candidate[0].metadata["benchmark_cache_flush_bytes"] *= 2
    assert calculate_average_speedup(baseline, candidate) == 0
    assert analyze_benchmark_method_consistency(baseline, candidate)[1][0]["reason"] == "cold_cache_policy_mismatch"
    row["metadata"] = {}
    with pytest.raises(TaskProtocolError, match="cold-cache timing requires"):
        parse_command_result(RESULT_PREFIX + json.dumps(report), role="candidate",
                             action="performance", returncode=0)


@pytest.mark.parametrize("method", ["cuda_graph", "cuda_event_fallback", "cuda_graph_cold_weights"])
def test_cold_cache_is_never_scored_against_warm_or_rotating_weight_methods(method):
    baseline = [Case("a", [1], 2.0, cold_metadata())]
    candidate = [Case("a", [1], 1.0, {"benchmark_method": method})]
    assert not analyze_benchmark_method_consistency(baseline, candidate)[0]
    assert calculate_average_speedup(baseline, candidate) == 0


def test_attention_example_cpu_oracle_uses_pristine_values_and_rejects_input_mutation():
    root = HELPER.parents[3]
    example_path = root / "src/tools/perf/examples/cold_cache_graph/paged_attention.py"
    spec = importlib.util.spec_from_file_location("_cold_attention_example", example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    checks = example.load_module(example.TASK / "_arena_checks.py", "_cold_fixture_checks")
    fixture = object.__new__(example.AttentionFixture)
    fixture.torch, fixture.harness, fixture.checks = torch, None, checks
    # Tiny asymmetric fixture: one query/head, two cache slots, two head lanes.
    # Exercise the actual task's oracle without GPU/BLAS dependencies.
    fixture.args = (
        torch.zeros(1, 1, 2, dtype=torch.float16),
        torch.empty(1, 1, 2, dtype=torch.float16),
        torch.zeros(1, 1, 2, 2, 1, dtype=torch.float16),
        torch.zeros(1, 1, 2, 2, dtype=torch.float16),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32), 1.0,
    )
    fixture.values = dict(enumerate(fixture.args)) | {"filter_by_query_len": False}
    fixture.readonly = [0, 2, 3, 4, 5, 6]
    fixture.initial = {i: fixture.args[i].clone() for i in fixture.readonly}
    fixture.metadata = {i: (t.shape, t.stride(), t.dtype, t.device, t.data_ptr())
                        for i, t in enumerate(fixture.args) if isinstance(t, torch.Tensor)}
    fixture.generator = torch.Generator().manual_seed(23)
    fixture.prepare()
    q = fixture.pristine[0][0, 0].tolist()
    k = fixture.pristine[2][0, 0, :, :, 0].tolist()
    v = fixture.pristine[3][0, 0].tolist()
    logits = [sum(q[d] * k[d][slot] for d in range(2)) for slot in range(2)]
    weights = [math.exp(z - max(logits)) for z in logits]
    wanted = [sum(weights[s] * v[d][s] for s in range(2)) / sum(weights) for d in range(2)]
    torch.testing.assert_close(fixture.expected, torch.tensor([[wanted]], dtype=torch.float16),
                               atol=0.001, rtol=0.001)
    fixture.args[1].copy_(fixture.expected)
    fixture.check("baseline", fixture.args[1])
    fixture.args[0].zero_()
    with pytest.raises(AssertionError, match="read-only input"):
        fixture.check("candidate", fixture.args[1])
