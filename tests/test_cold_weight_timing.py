"""Cold-weight timing: copy count, one graph over all weight copies, method pairing.

These tests run on the CPU with a fake ``torch.cuda``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from src.task_protocol import RESULT_PREFIX, parse_command_result, performance_cases
from src.testcases import (
    TestCaseResult as CaseResult,
    analyze_benchmark_method_consistency,
    calculate_average_speedup,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src/tools/perf/aka_benchmark.py"
METHOD = "cuda_graph_cold_weights"


class _Event:
    def __init__(self, enable_timing=False):
        del enable_timing

    def record(self, stream=None):
        del stream

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        del end
        # Every replay takes 12 ms of device time.
        return 12.0


class _Cuda:
    Event = _Event

    def is_available(self):
        return True

    def synchronize(self):
        pass

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda _s: None, synchronize=lambda: None)

    def current_stream(self):
        return object()

    def stream(self, _stream):
        return nullcontext()


@pytest.fixture
def helper(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location(f"_aka_benchmark_cold_{id(fake)}", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT", raising=False)
    return module


@pytest.mark.parametrize("n,k,expected", [
    (2048, 2048, 128),
    (4096, 2048, 64),
    (6144, 2048, 43),
    (16384, 2048, 16),
    # A 1 GiB weight still gets the minimum of 8 copies.
    (65536, 8192, 8),
])
def test_copy_count_passes_rotate_bytes(helper, n, k, expected):
    weight = n * k * 2
    count = helper.cold_weight_copy_count(weight)
    assert count == expected
    assert count * weight >= 1 << 30
    assert count == 8 or (count - 1) * weight < 1 << 30


@pytest.mark.parametrize("args", [(0,), (-1,), (1 << 20, 0), (1 << 20, 1 << 30, 1)])
def test_copy_count_rejects_sizes_that_describe_no_weight(helper, args):
    with pytest.raises(ValueError):
        helper.cold_weight_copy_count(*args)


def test_one_graph_calls_every_copy_and_reports_per_call(helper, monkeypatch):
    calls = []
    captured = []
    replays = []

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        captured.append(repeats)
        output_holder[:] = [fn()]
        return types.SimpleNamespace(replay=lambda: replays.append(True))

    monkeypatch.setattr(helper, "_capture_graph", capture)
    fns = [lambda i=i: calls.append(i) or i for i in range(8)]
    timed = helper.TimedRun()
    seen = []
    timed.after_sample = seen.append
    samples, metadata = helper.benchmark_cuda_graph_cold_weights_samples(
        fns, warmup=2, repetition=3, prepare_fn=lambda: None, timed_run=timed)
    # Each replay takes 12 ms and calls all 8 weight copies, so a sample is 1.5 ms per call.
    assert samples == [1.5, 1.5, 1.5]
    # The eager pass and the capture each call every copy once, in order.
    assert captured == [1]
    assert calls == list(range(8)) * 2
    assert metadata["benchmark_method"] == METHOD
    assert metadata["benchmark_calls_per_replay"] == 8
    assert metadata["benchmark_cold_weight_copies"] == 8
    assert metadata["benchmark_timed_run_kind"] == "captured_graph"
    assert seen == [list(range(8))] * 3
    assert timed.bound and timed.outputs == list(range(8))
    assert timed.rerun_ms() == 1.5
    # Two warm-up replays, three samples and one rerun.
    assert len(replays) == 6


def test_cold_timing_has_no_event_fallback(helper, monkeypatch):
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="CUDA-graph"):
        helper.benchmark_cuda_graph_cold_weights_samples([lambda: None, lambda: None])


def test_cold_timing_needs_callables(helper):
    with pytest.raises(ValueError):
        helper.benchmark_cuda_graph_cold_weights_samples([])


def test_empty_cold_capture_raises(helper, monkeypatch):
    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        raise helper._EmptyGraphCapture("empty")

    monkeypatch.setattr(helper, "_capture_graph", capture)
    with pytest.raises(RuntimeError, match="cold-weight"):
        helper.benchmark_cuda_graph_cold_weights_samples([lambda: None] * 2)


def _case(method, time_ms):
    return CaseResult("m_1", [1, 2048, 2048], time_ms,
                      {"benchmark_method": method, "benchmark_samples": 100})


def test_cold_method_is_a_valid_protocol_timing():
    row = dict(test_case_id="m_1", shape=[1, 2048, 2048], dtype="bfloat16", status="PASS",
               execution_time_ms=0.004, benchmark_method=METHOD)
    report = {"protocol": "arena-eval-v1", "role": "candidate", "action": "performance",
              "status": "PASS", "cases": [row]}
    result = parse_command_result(RESULT_PREFIX + json.dumps(report), role="candidate",
                                  action="performance", returncode=0)
    (case,) = performance_cases(result)
    assert case.execution_time_ms == 0.004
    assert case.metadata["benchmark_method"] == METHOD


def test_cold_against_cold_is_scored():
    baseline, optimized = [_case(METHOD, 0.006)], [_case(METHOD, 0.003)]
    assert analyze_benchmark_method_consistency(baseline, optimized) == (True, [])
    assert calculate_average_speedup(baseline, optimized) == pytest.approx(2.0)


@pytest.mark.parametrize("pair", [(METHOD, "cuda_graph"), ("cuda_graph", METHOD),
                                  (METHOD, "cuda_event_fallback")])
def test_cold_never_compares_with_another_method(pair):
    consistent, mismatches = analyze_benchmark_method_consistency(
        [_case(pair[0], 0.006)], [_case(pair[1], 0.003)])
    assert not consistent
    assert mismatches[0]["baseline_benchmark_method"] == pair[0]
    assert mismatches[0]["optimized_benchmark_method"] == pair[1]
