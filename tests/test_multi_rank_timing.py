"""Multi-rank cold-cache timing: max-over-ranks aggregation, synchronized release and the GPU flow.

CPU only. The pure aggregation functions run directly; the synchronized-release loop runs on
threads that stand in for ranks; the GPU flow runs against a fake ``torch.cuda`` (as in
test_aka_benchmark.py), so no event, graph or device behavior is verified here.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import statistics
import sys
import threading
import time
import types
from pathlib import Path

import pytest

HELPER = Path(__file__).parents[1] / "src/tools/perf/aka_benchmark.py"


def _load(monkeypatch=None, fake_torch=None):
    name = f"_aka_benchmark_rank_test_{id(fake_torch)}"
    if fake_torch is not None:
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(name, HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load()


class ThreadComm:
    """Ranks as threads: a shared barrier, broadcast slot and gather slots."""

    def __init__(self, rank: int, shared: dict):
        self.rank = rank
        self.world_size = shared["world_size"]
        self.shared = shared

    def barrier(self):
        self.shared["barrier"].wait(timeout=20)

    def broadcast_int(self, value):
        if self.rank == 0:
            self.shared["broadcast"] = value
        self.barrier()
        value = self.shared["broadcast"]
        self.barrier()
        return value

    def all_gather_object(self, obj):
        self.shared["slots"][self.rank] = obj
        self.barrier()
        gathered = list(self.shared["slots"])
        self.barrier()
        return gathered


def _comms(world_size: int) -> list[ThreadComm]:
    shared = {"world_size": world_size, "barrier": threading.Barrier(world_size), "broadcast": None,
              "slots": [None] * world_size}
    return [ThreadComm(rank, shared) for rank in range(world_size)]


def _run_ranks(world_size, body):
    """Run body(rank, comm) on one thread per rank; return results or re-raise the first error."""
    comms = _comms(world_size)
    results, errors = [None] * world_size, [None] * world_size

    def target(rank):
        try:
            results[rank] = body(rank, comms[rank])
        except BaseException as exc:  # noqa: BLE001 - the protocol makes every rank fail together
            errors[rank] = exc

    threads = [threading.Thread(target=target, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return results, errors


# ------------------------------------------------------------------------------------------------
# Pure aggregation


def test_a_sample_is_the_maximum_over_ranks(helper):
    assert helper.max_over_ranks([[1.0, 5.0, 2.0], [3.0, 1.0, 2.0], [0.5, 4.0, 2.5]]) == [3.0, 5.0, 2.5]


@pytest.mark.parametrize("samples, error", [
    ([], ValueError), ([[1.0], [1.0, 2.0]], ValueError), ([[], []], ValueError),
    ([[1.0], [0.0]], RuntimeError), ([[1.0], [float("nan")]], RuntimeError), ([[1.0], [-2.0]], RuntimeError),
])
def test_invalid_rank_samples_are_rejected(helper, samples, error):
    with pytest.raises(error):
        helper.max_over_ranks(samples)


def test_invalid_spans_name_their_rank(helper):
    with pytest.raises(RuntimeError, match="rank 1"):
        helper.max_over_ranks([[1.0, 1.0], [1.0, float("inf")]])


def test_release_skew_is_the_spread_of_release_times(helper):
    assert helper.release_skew_us([[1_000, 5_000], [3_000, 5_500], [2_000, 4_900]]) == [2.0, 0.6]


def test_summary_keeps_raw_samples_skew_and_slowest_ranks(helper):
    samples, metadata = helper.summarize_rank_samples(
        [[1.0, 2.0, 3.0, 9.0], [2.0, 1.0, 4.0, 1.0]],
        [[0, 10_000, 20_000, 30_000], [4_000, 11_000, 20_000, 38_000]],
        [[100, 0, 0, 0], [0, 2_000, 0, 50]])
    assert samples == [2.0, 2.0, 4.0, 9.0]
    assert statistics.median(samples) == 3.0
    assert metadata["benchmark_rank_aggregation"] == "max_over_ranks"
    assert metadata["benchmark_world_size"] == 2
    assert metadata["benchmark_rank_samples_ms"] == [[1.0, 2.0, 3.0, 9.0], [2.0, 1.0, 4.0, 1.0]]
    assert metadata["benchmark_release_skew_us"] == [4.0, 1.0, 0.0, 8.0]
    assert metadata["benchmark_release_skew_us_median"] == 2.5
    assert metadata["benchmark_release_skew_us_max"] == 8.0
    assert metadata["benchmark_slowest_rank_counts"] == [2, 2]
    assert metadata["benchmark_release_late_us_max"] == 2.0
    json.dumps(metadata, allow_nan=False)


def test_summary_rejects_mismatched_release_evidence(helper):
    with pytest.raises(ValueError):
        helper.summarize_rank_samples([[1.0, 2.0], [1.0, 2.0]], [[0, 1]])
    with pytest.raises(ValueError):
        helper.summarize_rank_samples([[1.0, 2.0], [1.0, 2.0]], [[0], [1]])


# ------------------------------------------------------------------------------------------------
# Synchronized release (ranks as threads)


def test_every_rank_is_released_at_one_deadline_and_reports_the_same_samples(helper):
    spans = {0: [1.0, 4.0, 2.0], 1: [3.0, 1.0, 5.0]}
    logs = {0: [], 1: []}
    deadlines = {0: [], 1: []}

    def body(rank, comm):
        log = logs[rank]
        original = comm.broadcast_int

        def broadcast(value):
            deadline = original(value)
            deadlines[rank].append(deadline)
            log.append("deadline")
            return deadline

        comm.broadcast_int = broadcast
        values = iter(spans[rank])

        def run_sample():
            log.append("run")
            return next(values)

        return helper.synchronized_release_samples(
            run_sample, comm=comm, repetition=3, prepare=lambda: log.append("prepare"),
            synchronize=lambda: log.append("synchronize"), after_sample=lambda i: log.append(f"check{i}"),
            release_lead_ms=1.0)

    results, errors = _run_ranks(2, body)
    assert errors == [None, None]
    (samples0, metadata0), (samples1, metadata1) = results
    assert samples0 == samples1 == [3.0, 4.0, 5.0]
    assert metadata0 == metadata1
    assert metadata0["benchmark_rank_samples_ms"] == [spans[0], spans[1]]
    assert metadata0["benchmark_release_policy"] == "barrier_then_rank0_monotonic_deadline"
    assert metadata0["benchmark_release_lead_ms"] == 1.0
    assert deadlines[0] == deadlines[1]
    for rank in (0, 1):
        assert logs[rank] == ["prepare", "synchronize", "deadline", "run", "check0",
                              "prepare", "synchronize", "deadline", "run", "check1",
                              "prepare", "synchronize", "deadline", "run", "check2"]
    # Releases happen at or after the shared deadline; on one host the threads are microseconds apart.
    assert all(skew >= 0 for skew in metadata0["benchmark_release_skew_us"])
    assert metadata0["benchmark_release_late_us_max"] >= 0


def test_the_release_waits_for_the_deadline(helper):
    clock = {"now": 1_000_000}
    seen = []

    def tick():
        clock["now"] += 250_000  # 0.25 ms per read
        return clock["now"]

    class SoloComm:
        rank, world_size = 0, 1

        def barrier(self):
            pass

        def broadcast_int(self, value):
            seen.append(value)
            return value

        def all_gather_object(self, obj):
            return [obj]

    samples, metadata = helper.synchronized_release_samples(
        lambda: 1.5, comm=SoloComm(), repetition=1, release_lead_ms=2.0, clock=tick)
    assert samples == [1.5]
    # The deadline was 2 ms after rank 0's read; the release is the first read at or after it.
    assert seen == [1_250_000 + 2_000_000]
    assert 0 <= metadata["benchmark_release_late_us_max"] < 250


@pytest.mark.parametrize("kwargs", [{"repetition": 0}, {"repetition": 2, "release_lead_ms": 0},
                                    {"repetition": 2, "release_lead_ms": float("nan")}])
def test_release_parameters_are_validated(helper, kwargs):
    with pytest.raises(ValueError):
        helper.synchronized_release_samples(lambda: 1.0, comm=_comms(1)[0], **kwargs)


# ------------------------------------------------------------------------------------------------
# The GPU flow against a fake torch.cuda


class _FakeEvent:
    def __init__(self, cuda, enable_timing=False):
        self.cuda = cuda

    def record(self, stream=None):
        self.cuda.log().append("event")

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return self.cuda.next_span()


class _FakeCuda:
    def __init__(self, spans_by_thread):
        self.spans = spans_by_thread
        self.logs: dict = {}

    def log(self):
        return self.logs.setdefault(threading.current_thread().name, [])

    def next_span(self):
        return self.spans[threading.current_thread().name].pop(0)

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def synchronize(self):
        self.log().append("synchronize")

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda _stream: None, synchronize=lambda: None)

    def current_stream(self):
        return object()

    def Event(self, enable_timing=False):
        return _FakeEvent(self, enable_timing)

    @contextmanager
    def stream(self, _stream):
        yield


def _fake_helper(monkeypatch, spans_by_thread):
    fake = types.ModuleType("torch")
    fake.cuda = _FakeCuda(spans_by_thread)
    module = _load(monkeypatch, fake)

    class Flush(module.ReadOnlyCacheFlush):
        def __init__(self):
            self.nbytes = 1 << 30
            self.device = types.SimpleNamespace(index=0)

        def __call__(self):
            fake.cuda.log().append("flush")

    return module, fake.cuda, Flush


def test_gpu_flow_warms_up_captures_primes_and_times_one_call_per_replay(monkeypatch):
    monkeypatch.setattr(threading.current_thread(), "name", "solo")
    helper, cuda, Flush = _fake_helper(monkeypatch, {"solo": [0.5] + [0.2, 0.4, 0.3]})
    captured = []

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        captured.append(repeats)
        prepare_fn()
        cuda.log().append("capture")
        return types.SimpleNamespace(replay=lambda: cuda.log().append("replay"))

    monkeypatch.setattr(helper, "_capture_graph", capture)
    comm = _comms(1)[0]
    checked = []
    samples, metadata = helper.benchmark_distributed_cold_cache_samples(
        lambda: cuda.log().append("call"), cache_flush=Flush(), comm=comm, warmup=2, repetition=3,
        prepare_fn=lambda: cuda.log().append("prepare"), after_sample=checked.append)
    assert samples == [0.2, 0.4, 0.3] and checked == [0, 1, 2] and captured == [1]
    log = cuda.log()
    assert log[:9] == ["synchronize", "prepare", "flush", "synchronize", "call", "synchronize",
                       "prepare", "flush", "synchronize"]
    assert log.count("call") == 2  # warm-up only; every timed call is a replay
    assert log.count("replay") == 4  # one priming replay, three samples
    sample = log[-6:]
    assert sample == ["prepare", "flush", "synchronize", "event", "replay", "event"]
    assert metadata["benchmark_method"] == "cuda_graph_cold_cache"
    assert metadata["benchmark_effective_repeats"] == 1
    assert metadata["benchmark_metric"] == "gpu_event_graph_span"
    assert metadata["benchmark_cache_flush_bytes"] == 1 << 30
    assert metadata["benchmark_rank_aggregation"] == "max_over_ranks"
    assert metadata["benchmark_rank_samples_ms"] == [[0.2, 0.4, 0.3]]

    from src.measurement import cold_cache_identity
    from src.task_protocol import parse_command_result

    assert cold_cache_identity(metadata)
    row = {"test_case_id": "c", "status": "PASS", "execution_time_ms": statistics.median(samples),
           "benchmark_method": metadata["benchmark_method"], "metadata": metadata}
    report = {"protocol": "arena-eval-v1", "role": "candidate", "action": "performance", "status": "PASS",
              "cases": [row]}
    parsed = parse_command_result("ARENA_EVAL_RESULT=" + json.dumps(report), role="candidate",
                                  action="performance", returncode=0)
    assert parsed.passed


def test_a_capture_failure_on_one_rank_fails_every_rank(monkeypatch):
    helper, cuda, Flush = _fake_helper(monkeypatch, {"rank0": [], "rank1": []})

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        if threading.current_thread().name == "rank1":
            raise RuntimeError("operation not permitted when stream is capturing")
        return types.SimpleNamespace(replay=lambda: None)

    monkeypatch.setattr(helper, "_capture_graph", capture)

    def body(rank, comm):
        threading.current_thread().name = f"rank{rank}"
        return helper.benchmark_distributed_cold_cache_samples(
            lambda: None, cache_flush=Flush(), comm=comm, warmup=0, repetition=2)

    _, errors = _run_ranks(2, body)
    for error in errors:
        assert isinstance(error, RuntimeError)
        assert "CUDA-graph capture failed: rank 1: RuntimeError: operation not permitted" in str(error)


def test_an_empty_replay_on_any_rank_fails_every_rank(monkeypatch):
    helper, cuda, Flush = _fake_helper(monkeypatch, {"rank0": [0.5, 0.3, 0.3], "rank1": [0.5, 0.3, 1.0e-6]})
    monkeypatch.setattr(helper, "_capture_graph",
                        lambda fn, repeats, stream, prepare_fn=None, output_holder=None:
                        types.SimpleNamespace(replay=lambda: None))

    def body(rank, comm):
        threading.current_thread().name = f"rank{rank}"
        return helper.benchmark_distributed_cold_cache_samples(
            lambda: None, cache_flush=Flush(), comm=comm, warmup=0, repetition=2)

    _, errors = _run_ranks(2, body)
    assert all(isinstance(e, RuntimeError) and "empty or invalid" in str(e) for e in errors)


def test_forced_event_timing_is_rejected(monkeypatch):
    helper, cuda, Flush = _fake_helper(monkeypatch, {})
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="Event timing is forbidden"):
        helper.benchmark_distributed_cold_cache_samples(lambda: None, cache_flush=Flush(), comm=_comms(1)[0])


def test_the_lead_keeps_ranks_together_on_one_host(helper):
    """Real threads and the real monotonic clock: every rank releases within the lead window."""
    def body(rank, comm):
        return helper.synchronized_release_samples(lambda: time.sleep(0.0005) or 0.5, comm=comm, repetition=20,
                                                   release_lead_ms=2.0)

    results, errors = _run_ranks(4, body)
    assert errors == [None] * 4
    metadata = results[0][1]
    assert len(metadata["benchmark_release_skew_us"]) == 20
    assert metadata["benchmark_world_size"] == 4
