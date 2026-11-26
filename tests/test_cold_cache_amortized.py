"""Amortized cold-cache timing: call count, subtraction, poison and per-call output checks."""
from __future__ import annotations

from contextlib import nullcontext
import functools
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from src.tools.arena_eval import evaluate, integrity, values

HELPER = Path(__file__).resolve().parents[1] / "src/tools/perf/aka_benchmark.py"
HELPER_MODULE = "src.tools.arena_eval._aka_benchmark"
OPERATOR_SPAN, REFERENCE_SPAN, SINGLE_SPAN = 0.9, 0.5, 0.02


@pytest.fixture
def helper(monkeypatch):
    """The real helper over a CPU stand-in for torch.cuda whose graphs replay eagerly.

    A captured graph runs its Python once at capture, as real capture does, and
    re-runs it on every replay; each graph's Event span is fixed by its kind, so
    a sample's arithmetic is exact.
    """
    trace, state = [], {"span": None}

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return state["span"]

    cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, Event=Event, synchronize=lambda: None,
        Stream=lambda: types.SimpleNamespace(wait_stream=lambda s: None, synchronize=lambda: None),
        current_stream=lambda: object(), stream=lambda s: nullcontext(),
    )
    fake = types.ModuleType("torch")
    fake.__getattr__ = lambda name: getattr(torch, name)
    fake.cuda, fake.device, fake.int32 = cuda, torch.device, torch.int32
    fake.zeros = lambda shape, dtype, device: torch.zeros(shape, dtype=dtype)
    fake.empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype)

    def sum_(value, **kwargs):
        trace.append("flush")
        return torch.sum(value, **kwargs)

    fake.sum = sum_
    monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_amortized_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        # The single-call cold-cache method always passes its preparation; the
        # amortized operator graph collects its outputs, the reference graph not.
        kind = "single" if prepare_fn is not None else "operator" if output_holder is not None else "reference"
        if prepare_fn is not None:
            prepare_fn()
        trace.append(f"capture-{kind}")
        for _ in range(repeats):
            output = fn()
        if output_holder is not None:
            output_holder[:] = [output]

        def replay():
            trace.append(f"replay-{kind}")
            state["span"] = {"single": SINGLE_SPAN, "operator": OPERATOR_SPAN, "reference": REFERENCE_SPAN}[kind]
            for _ in range(repeats):
                fn()

        return types.SimpleNamespace(replay=replay)

    monkeypatch.setattr(module, "_capture_graph", capture)
    module.trace = trace
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT", raising=False)
    monkeypatch.setitem(sys.modules, HELPER_MODULE, module)
    return module


# --------------------------------------------------------------------------- helper


@pytest.mark.parametrize("estimate_ms, expected", [
    (0.001, 32), (0.0625, 32), (0.1, 20), (0.25, 8), (0.5, 4), (1.0, 4), (40.0, 4)])
def test_calls_per_replay_targets_two_milliseconds_between_four_and_thirty_two(helper, estimate_ms, expected):
    assert helper.amortized_calls_per_replay(estimate_ms) == expected


def test_snapshot_memory_budget_reduces_calls_but_never_below_the_minimum(helper):
    gib = 1 << 30
    assert helper.amortized_calls_per_replay(0.001, per_call_bytes=gib // 8) == 32
    assert helper.amortized_calls_per_replay(0.001, per_call_bytes=gib // 4) == 16
    assert helper.amortized_calls_per_replay(0.001, per_call_bytes=gib) == 4
    assert helper.amortized_calls_per_replay(0.001, per_call_bytes=64 * gib) == 4
    for estimate in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite estimate"):
            helper.amortized_calls_per_replay(estimate)
    with pytest.raises(ValueError, match="per_call_bytes"):
        helper.amortized_calls_per_replay(0.1, per_call_bytes=-1)


def test_graphs_hold_k_cold_calls_and_samples_subtract_the_reference_graph(helper):
    flush = helper.ReadOnlyCacheFlush(16384)
    observed = []
    timed = helper.TimedRun()

    def after(outputs):
        # The operator graph's results are in place whenever a sample is observed.
        observed.append([name for name in helper.trace if name.startswith("replay")][-1])

    timed.after_sample = after
    count = 4

    def operation(index):
        helper.trace.append(f"kernel{index}")
        return f"out{index}"

    samples, metadata = helper.benchmark_cuda_graph_cold_cache_amortized_samples(
        [functools.partial(operation, index) for index in range(count)], cache_flush=flush,
        warmup=2, repetition=4, prepare_fn=lambda i: helper.trace.append(f"prepare{i}"), timed_run=timed)
    assert samples == pytest.approx([(OPERATOR_SPAN - REFERENCE_SPAN) / count] * 4)
    trace = helper.trace
    # Eager warm-up: preparation and the call, cycling through the calls.
    assert trace[:4] == ["prepare0", "kernel0", "prepare1", "kernel1"]
    start = trace.index("capture-operator") + 1
    body = [f"prepare{i} flush kernel{i}".split() for i in range(count)]
    assert trace[start:start + 3 * count] == [step for call in body for step in call]
    start = trace.index("capture-reference") + 1
    assert trace[start:start + 2 * count] == [s for call in body for s in call if not s.startswith("kernel")]
    replays = [name.removeprefix("replay-") for name in trace if name.startswith("replay")]
    # Two priming pairs, then alternating order per sample, then the final reference control.
    assert replays == (["reference", "operator"] * 2 + ["reference", "operator", "operator", "reference"] * 2
                       + ["reference"])
    assert observed == ["replay-operator"] * 4
    assert metadata["benchmark_method"] == "cuda_graph_cold_cache_amortized"
    assert metadata["benchmark_calls_per_replay"] == count
    assert metadata["benchmark_cache_flush_bytes"] == 16384
    assert metadata["benchmark_cache_flush_in_timing"] is False
    assert metadata["benchmark_operator_graph_ms"] == [OPERATOR_SPAN] * 4
    assert metadata["benchmark_reference_graph_ms"] == [REFERENCE_SPAN] * 4
    assert timed.outputs == [f"out{index}" for index in range(count)]
    assert timed.rerun_ms() == pytest.approx((OPERATOR_SPAN - REFERENCE_SPAN) / count)


def test_amortized_timing_fails_closed(helper, monkeypatch):
    flush = helper.ReadOnlyCacheFlush(16384)
    calls = [lambda: None] * 4
    run = functools.partial(helper.benchmark_cuda_graph_cold_cache_amortized_samples,
                            cache_flush=flush, warmup=0, repetition=2)
    for invalid in ([], [None], [lambda: None, 3]):
        with pytest.raises(ValueError, match="one callable per call"):
            run(invalid)
    with pytest.raises(TypeError, match="ReadOnlyCacheFlush"):
        helper.benchmark_cuda_graph_cold_cache_amortized_samples(calls, cache_flush=object())
    timed = helper.TimedRun()
    timed.before_sample = lambda count: None
    with pytest.raises(ValueError, match="before cache eviction"):
        run(calls, timed_run=timed)
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="Event timing is forbidden"):
        run(calls)
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT")

    def fail(*args, **kwargs):
        raise RuntimeError("operation not permitted when stream is capturing")

    with monkeypatch.context() as patch:
        patch.setattr(helper, "_capture_graph", fail)
        with pytest.raises(RuntimeError, match="capture failed"):
            run(calls)
    # An operator graph no slower than its reference is an empty measurement, never a fallback.
    monkeypatch.setattr(helper, "_amortized_replay_ms", lambda graph, stream: graph.replay() or 0.5)
    with pytest.raises(RuntimeError, match="empty or invalid"):
        run(calls)


# ------------------------------------------------------------------------ evaluator


def _contract(mutable=()):
    checked = ("y", "idx") + tuple(mutable)
    return types.SimpleNamespace(mutable=tuple(mutable), checked_tensors=checked, element_order="fixed",
                                 compare={name: {"bound": "exact"} for name in checked}, timing={"warmup": 0,
                                                                                                "repetition": 4})


def _data(seeds=(0, 1), mutable=False):
    inputs, references = [], []
    for seed in seeds:
        x = torch.randn(64, generator=torch.Generator().manual_seed(seed))
        kernel = {"x": x}
        reference = {"y": 2 * x, "idx": (x > 0).to(torch.int32)}
        if mutable:
            kernel["state"] = x + 1
            reference["state"] = kernel["state"] + 1
        inputs.append(types.SimpleNamespace(kernel=kernel))
        references.append(reference)
    return types.SimpleNamespace(inputs=inputs, references=references, baselines=None)


def _honest(x, y, idx, state=None):
    y.copy_(2 * x)
    idx.copy_((x > 0).to(torch.int32))
    if state is not None:
        state.add_(1)


def _amortized(helper, call, *, role="candidate", mutable=False, estimate_ms=0.01, check=None):
    contract = _contract(("state",) if mutable else ())
    data = _data(mutable=mutable)
    working = values.clone_inputs(data.inputs[0].kernel)
    outputs = {"y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)}
    return evaluate._amortized_cold(contract, role, call, data, working, outputs,
                                    helper.ReadOnlyCacheFlush(16384), check, estimate_ms)


def test_every_call_of_an_honest_candidate_is_checked_after_every_sample(helper):
    samples, metadata, failed = _amortized(helper, _honest, mutable=True)
    assert failed == []
    assert samples == pytest.approx([(OPERATOR_SPAN - REFERENCE_SPAN) / 32] * 4)
    assert metadata["benchmark_calls_per_replay"] == 32
    assert metadata["checked_samples"] == 4 and metadata["checked_calls"] == 4 * 32
    assert metadata["reference_control_undetected_calls"] == 0
    # y, idx and the mutable state per call, plus an int32 poison pattern.
    assert metadata["benchmark_call_buffer_bytes"] == 64 * 4 * 3 + 64 * 4


def test_a_candidate_that_works_only_on_the_last_captured_call_fails(helper):
    calls = {"count": 0}

    def last_call_only(x, y, idx):
        # Python runs once per captured call; it counts the calls and computes on every 32nd.
        calls["count"] += 1
        if calls["count"] % 32 == 0:
            _honest(x, y, idx)

    samples, metadata, failed = _amortized(helper, last_call_only)
    assert failed and all(failure["call"] != 31 for failure in failed if "call" in failure)
    assert failed[0]["sample"] == 1 and failed[0]["call"] == 0


def test_a_candidate_that_skips_one_call_fails(helper):
    calls = {"count": 0}

    def skips_call_five(x, y, idx):
        calls["count"] += 1
        if calls["count"] % 32 != 6:
            _honest(x, y, idx)

    _, _, failed = _amortized(helper, skips_call_five)
    assert [failure["call"] for failure in failed] == [5, 5, 5]


def test_checks_that_accept_calls_without_the_operator_fail_a_candidate_only(helper, monkeypatch):
    monkeypatch.setattr(evaluate, "_check", lambda *args: (True, {}))
    _, metadata, failed = _amortized(helper, _honest)
    assert metadata["reference_control_undetected_calls"] == 32
    assert "per-call work cannot be verified" in failed[-1]["control"]
    _, metadata, failed = _amortized(helper, _honest, role="baseline")
    assert metadata["reference_control_undetected_calls"] == 32 and failed == []


def test_the_memory_budget_reduces_calls_per_replay(helper, monkeypatch):
    per_call = 64 * 4 * 2 + 64 * 4
    monkeypatch.setattr(helper, "amortized_calls_per_replay",
                        functools.partial(helper.amortized_calls_per_replay, budget_bytes=5 * per_call))
    _, metadata, failed = _amortized(helper, _honest)
    assert failed == [] and metadata["benchmark_calls_per_replay"] == 5
    monkeypatch.setattr(helper, "amortized_calls_per_replay",
                        functools.partial(helper.amortized_calls_per_replay.func, budget_bytes=per_call))
    _, metadata, _ = _amortized(helper, _honest)
    assert metadata["benchmark_calls_per_replay"] == 4


def test_long_operators_get_fewer_calls(helper):
    _, metadata, failed = _amortized(helper, _honest, estimate_ms=0.4)
    assert failed == [] and metadata["benchmark_calls_per_replay"] == 5


def test_poison_patterns_are_precomputed_per_call_and_match_eager_poison():
    outputs = {"y": torch.empty(16), "idx": torch.empty(16, dtype=torch.int32),
               "q": torch.empty(16, dtype=torch.uint8)}
    first, second = values.poison_pattern(outputs, 1000), values.poison_pattern(outputs, 1001)
    assert set(first) == {"idx", "q"}  # float outputs are NaN-filled, which captures without a generator
    assert not torch.equal(first["idx"], second["idx"]) and not torch.equal(first["q"], second["q"])
    eager = {name: torch.empty_like(tensor) for name, tensor in outputs.items()}
    values.poison(eager, 1000)
    values.apply_poison(outputs, first)
    assert torch.isnan(outputs["y"]).all()
    assert torch.equal(outputs["idx"], eager["idx"]) and torch.equal(outputs["q"], eager["q"])


def test_performance_rows_report_the_amortized_time_and_the_single_call_time(helper, monkeypatch):
    contract = _contract()
    case = types.SimpleNamespace(checks=("correctness", "performance"), identity=lambda: {"test_case_id": "c0"},
                                 case_id="c0")
    contract.cases = [case]
    flush = helper.ReadOnlyCacheFlush(16384)
    monkeypatch.setattr(evaluate, "_device", lambda: "cpu")
    monkeypatch.setattr(evaluate, "_role_calls", lambda contract, role: (_honest, None, False, None))
    monkeypatch.setattr(evaluate, "_cache_flush", lambda nbytes: flush)
    monkeypatch.setattr(evaluate, "_CaseData", lambda *args: _data())
    monkeypatch.setattr(values, "allocate_outputs", lambda contract, case, device: {
        "y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)})
    report = evaluate.performance_action(contract, "candidate")
    assert report["status"] == "PASS", report
    row = report["cases"][0]
    assert row["benchmark_method"] == "cuda_graph_cold_cache_amortized"
    assert row["execution_time_ms"] == pytest.approx((OPERATOR_SPAN - REFERENCE_SPAN) / 32)
    assert row["metadata"]["single_call_cold_ms"] == pytest.approx(SINGLE_SPAN)
    assert row["metadata"]["single_call_cold_method"] == "cuda_graph_cold_cache"
    assert row["metadata"]["benchmark_calls_per_replay"] == 32

    from src.measurement import cold_cache_identity
    cold_cache_identity(row["metadata"], row["benchmark_method"])


def test_tampering_found_during_an_amortized_sample_fails_the_action(helper, monkeypatch):
    contract = _contract()
    contract.cases = [types.SimpleNamespace(checks=("performance",), identity=lambda: {"test_case_id": "c0"})]
    flush = helper.ReadOnlyCacheFlush(16384)
    stages = []

    def check(stage):
        stages.append(stage)
        # The single-call stage's samples pass; the first amortized sample is tampered.
        if stage == "a timing sample" and "timing" in stages:
            raise integrity.TamperError(f"Harness tampering detected after {stage}: [...]")

    monkeypatch.setattr(evaluate, "load_contract", lambda: contract)
    monkeypatch.setattr(evaluate, "_device", lambda: "cpu")
    monkeypatch.setattr(evaluate, "_role_calls", lambda contract, role: (_honest, None, False, check))
    monkeypatch.setattr(evaluate, "_cache_flush", lambda nbytes: flush)
    monkeypatch.setattr(evaluate, "_CaseData", lambda *args: _data())
    monkeypatch.setattr(values, "allocate_outputs", lambda contract, case, device: {
        "y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)})
    report, code = evaluate.run(["candidate", "performance"])
    assert code == 1 and report["failure_kind"] == "harness_tampering"


# ------------------------------------------------------------------ comparison


def _amortized_metadata(calls=32):
    return {"benchmark_method": "cuda_graph_cold_cache_amortized",
            "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
            "benchmark_cache_flush_in_timing": False,
            "benchmark_metric": "gpu_event_graph_span_difference_per_call",
            "benchmark_replay_order": "alternating", "benchmark_calls_per_replay": calls,
            "benchmark_cache_flush_bytes": 1 << 30}


def test_amortized_cases_compare_across_call_counts_but_never_with_single_call_timing():
    import json

    from src.task_protocol import RESULT_PREFIX, TaskProtocolError, parse_command_result, performance_cases
    from src.testcases import (TestCaseResult as Case, analyze_benchmark_method_consistency,
                               calculate_average_speedup)

    row = {"test_case_id": "a", "shape": [1], "status": "PASS", "execution_time_ms": 2.0,
           "benchmark_method": "cuda_graph_cold_cache_amortized", "metadata": _amortized_metadata(32)}
    report = {"protocol": "arena-eval-v1", "role": "baseline", "action": "performance",
              "status": "PASS", "cases": [row]}
    baseline = performance_cases(parse_command_result(RESULT_PREFIX + json.dumps(report), role="baseline",
                                                      action="performance", returncode=0))
    # The candidate is slower per call, so it gets fewer calls per replay: still comparable.
    candidate = [Case("a", [1], 1.0, _amortized_metadata(4))]
    assert analyze_benchmark_method_consistency(baseline, candidate) == (True, [])
    assert calculate_average_speedup(baseline, candidate) == pytest.approx(2)
    single = {"benchmark_method": "cuda_graph_cold_cache",
              "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
              "benchmark_cache_flush_in_timing": False, "benchmark_effective_repeats": 1,
              "benchmark_metric": "gpu_event_graph_span", "benchmark_cache_flush_bytes": 1 << 30}
    assert not analyze_benchmark_method_consistency(baseline, [Case("a", [1], 1.0, single)])[0]
    assert calculate_average_speedup(baseline, [Case("a", [1], 1.0, single)]) == 0
    for broken in ({"benchmark_calls_per_replay": 0}, {"benchmark_metric": "gpu_event_graph_span"},
                   {"benchmark_method": "cuda_graph_cold_cache"}):
        row["metadata"] = {**_amortized_metadata(), **broken}
        with pytest.raises(TaskProtocolError, match="cold-cache"):
            parse_command_result(RESULT_PREFIX + json.dumps(report), role="baseline",
                                 action="performance", returncode=0)
