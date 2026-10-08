"""Exercise the RMSNorm adapter with the actual materialized timing helper.

The fake CUDA backend records stream/event boundaries and captures deferred
kernel writes. It does not replace the canonical capture/sampling functions.
"""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import types

import pytest

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace

ROOT = Path(__file__).parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm"
HELPER = ROOT / "src/tools/perf/aka_benchmark.py"
META = json.loads((TASK / "ut/meta.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, cuda, name, value, shape=(64, 8192), stride=(8192, 1)):
        self.cuda, self.name, self.value = cuda, name, value
        self.shape, self._stride, self.dtype = shape, stride, "torch.bfloat16"

    def clone(self):
        return Tensor(self.cuda, self.name + "_clone", self.value, self.shape, self._stride)

    def stride(self):
        return self._stride

    def untyped_storage(self):
        return types.SimpleNamespace(data_ptr=lambda: id(self))

    def copy_(self, other):
        self.cuda.log.append(("copy", self.name))
        assert not self.cuda.in_interval and self.cuda.capturing is None
        self.value = other.value

    def fill_(self, value):
        self.cuda.log.append(("poison", self.name))
        assert math.isnan(value)
        assert not self.cuda.in_interval and self.cuda.capturing is None
        self.value = value

    def equal(self, other):
        self.cuda.log.append(("input_check", self.name))
        assert not self.cuda.in_interval and self.cuda.capturing is None
        return self.value == other.value


class Cuda:
    def __init__(self, *, defect=None):
        self.log, self.graphs = [], []
        self.capturing = None
        self.in_interval = False
        self.stream_now = object()
        self.event_count = 0
        self.defect = defect

    def is_available(self):
        return True

    def is_current_stream_capturing(self):
        return self.capturing is not None

    def synchronize(self):
        assert not self.in_interval

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)

    def current_stream(self):
        return self.stream_now

    @contextmanager
    def stream(self, stream):
        prior, self.stream_now = self.stream_now, stream
        try:
            yield
        finally:
            self.stream_now = prior

    def CUDAGraph(self):
        cuda = self

        class Graph:
            def __init__(self):
                self.ops = []
                self.replays = 0
                self.index = len(cuda.graphs)
                self.outputs = None

            def replay(self):
                assert cuda.capturing is None
                assert len(self.ops) == 1
                self.replays += 1
                cuda.log.append(("replay", self.index, self.replays))
                for op in self.ops:
                    op(self.replays)

        graph = Graph()
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph):
        if self.defect == "capture_error":
            raise RuntimeError("simulated capture failure")
        assert not self.in_interval
        self.capturing = graph
        self.log.append(("capture_start", graph.index))
        try:
            yield
        finally:
            self.log.append(("capture_end", graph.index))
            self.capturing = None

    def Event(self, enable_timing):
        assert enable_timing
        cuda = self
        index = self.event_count
        self.event_count += 1

        class Event:
            def record(self, stream):
                assert stream is cuda.stream_now
                start = index % 2 == 0
                assert cuda.in_interval != start
                cuda.in_interval = start
                cuda.log.append(("start" if start else "end", index // 2))

            def synchronize(self):
                assert not cuda.in_interval

            def elapsed_time(self, end):
                if cuda.defect == "invalid_time":
                    return 0.0
                # Deliberately unsorted: aggregation must not reorder raw data.
                return (7 - (index // 2) % 7) / 10

        return Event()


def environment(monkeypatch, tmp_path, *, defect=None):
    workspace = tmp_path / "task"
    shutil.copytree(TASK, workspace, symlinks=True)
    materialize_perf_helpers_in_workspace(workspace)
    assert (workspace / "scripts/_aka_benchmark.py").read_bytes() == HELPER.read_bytes()
    cuda = Cuda(defect=defect)
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    helper = load("_rmsnorm_real_helper", workspace / "scripts/_aka_benchmark.py")
    monkeypatch.setitem(sys.modules, "_aka_benchmark", helper)
    bench = load("_rmsnorm_bench", workspace / "scripts/_bench.py")
    cases = load("_rmsnorm_cases", workspace / "ut/cases.py")
    runner = load("_rmsnorm_runner", workspace / "scripts/task_runner.py")
    spec = META["workload"]["cases"][1]
    args = {name: Tensor(cuda, name, value, tuple(spec[name + "_shape"]),
                         tuple(spec[name + "_stride"]))
            for name, value in (("x", 0.75), ("residual", -0.25), ("weight", 0.0625))}
    args.update(eps=1e-6, verify_inputs=False)

    def baseline(x, residual, weight, eps):
        value = x.value + residual.value
        return (Tensor(cuda, "reference_normed", value * (1 + weight.value)),
                Tensor(cuda, "reference_sum", value))

    generation = 0

    def candidate(x, residual, weight, eps):
        nonlocal generation
        generation += 1
        normed = Tensor(cuda, f"normed_{generation}", float("nan"))
        summed = Tensor(cuda, f"pre_norm_sum_{generation}", float("nan"))
        graph = cuda.capturing

        def kernel(replay=0):
            cuda.log.append(("kernel", generation))
            assert (x.value, residual.value, weight.value) == (0.75, -0.25, 0.0625)
            if graph is not None:
                # Every graph launch must consume both poisoned buffers.
                assert math.isnan(normed.value) and math.isnan(summed.value)
            omit = defect if graph is not None else None
            if omit == "last_sample_normed":
                omit = "normed" if graph.index == 1 and replay == 101 else None
            value = x.value + residual.value
            if omit != "normed":
                normed.value = value * (1 + weight.value)
            if omit != "pre_norm_sum":
                summed.value = value
            if omit == "mutate_input":
                x.value += 1

        if graph is not None:
            graph.ops.append(kernel)
            graph.outputs = (normed, summed)
        else:
            kernel()
        return normed, summed

    cases._RESOLVED = candidate

    def correct(out, ref, tol):
        assert tol == 0.02
        assert not cuda.in_interval and cuda.capturing is None
        cuda.log.append(("output_check", out.name))
        return out.value == ref.value, 0.0 if out.value == ref.value else float("inf")

    harness = types.SimpleNamespace(correct=correct)
    case = {"args": args, "sig": spec["sig"], "regime": spec["regime"], "m": spec["m"]}
    return bench, runner, torch, cases, harness, case, spec, baseline, helper


def measure(env):
    bench, _, torch, cases, harness, case, spec, baseline, _ = env
    return bench.measure_case(torch, cases, harness, case, spec, baseline, 0.02)


def test_real_helper_orders_resets_poison_replay_events_and_retains_samples(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    row = measure(env)
    cuda = env[2].cuda
    assert row["benchmark_method"] == "cuda_graph"
    assert row["benchmark_warmup"] == 10
    assert row["benchmark_samples"] == 100
    assert row["benchmark_effective_repeats"] == 1
    assert row["samples_ms"] == [(7 - index % 7) / 10 for index in range(3, 103)]
    assert row["samples_ms"] != sorted(row["samples_ms"])
    assert row["validation"]["checked_invocations"] == 114
    assert [graph.replays for graph in cuda.graphs] == [2, 102]
    assert [len(graph.ops) for graph in cuda.graphs] == [1, 1]
    starts = [index for index, entry in enumerate(cuda.log) if entry[0] == "start"]
    assert len(starts) == 103  # estimate prime/sample, final prime, 100 samples
    for index in starts:
        before = cuda.log[index - 5:index]
        assert before[:3] == [("copy", name) for name in ("x", "residual", "weight")]
        assert before[3][0] == before[4][0] == "poison"
        assert before[3][1].startswith("normed_")
        assert before[4][1].startswith("pre_norm_sum_")
        assert [entry[0] for entry in cuda.log[index:index + 4]] == ["start", "replay", "kernel", "end"]
    # The last checked outputs belong to the final graph, not a separate eager call.
    assert cuda.log[-2:] == [("output_check", tensor.name) for tensor in cuda.graphs[-1].outputs]
    captures = [index for index, entry in enumerate(cuda.log) if entry[0] == "capture_start"]
    assert len([entry for entry in cuda.log[:captures[0]] if entry[0] == "kernel"]) == 10


@pytest.mark.parametrize("defect", ["normed", "pre_norm_sum", "last_sample_normed", "mutate_input",
                                    "capture_error", "invalid_time"])
def test_actual_graph_failures_never_become_event_samples(monkeypatch, tmp_path, defect):
    env = environment(monkeypatch, tmp_path, defect=defect)
    # Fail if the real helper ever attempts an eager fallback for this adapter.
    monkeypatch.setattr(env[-1], "benchmark_cuda_event_samples",
                        lambda *a, **kw: pytest.fail("event fallback attempted"))
    with pytest.raises(RuntimeError):
        measure(env)


def test_forced_events_are_rejected(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="timing is disabled"):
        measure(env)
    assert env[2].cuda.graphs == []


def raw_report(row):
    rows = []
    for spec in META["workload"]["cases"]:
        other = copy.deepcopy(row)
        other["sig"] = f"{spec['sig']}|{spec['regime']}"
        other["params"] = {key: spec[key] for key in row["params"]}
        rows.append(other)
    return {"timer": "cuda_graph", "warmup": 10, "iters": 100, "cases": rows}


def test_report_propagates_all_samples_and_per_case_method(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    raw = raw_report(measure(env))
    rows = env[1]._benchmark_cases(raw)
    assert {row["test_case_id"] for row in rows} == {
        "prefill_m8192_live_shape|prefill", "decode_m64_live_shape|decode"}
    for actual, original in zip(rows, raw["cases"]):
        assert actual["samples_ms"] == original["samples_ms"]
        assert actual["benchmark_method"] == "cuda_graph"
        assert actual["validation"] == original["validation"]


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(timer="cuda_event"),
    lambda raw: raw.update(iters=99),
    lambda raw: raw["cases"].pop(),
    lambda raw: raw["cases"].__setitem__(1, copy.deepcopy(raw["cases"][0])),
    lambda raw: raw["cases"][0].pop("benchmark_method"),
    lambda raw: raw["cases"][0].update(benchmark_method="cuda_event_fallback"),
    lambda raw: raw["cases"][0].update(benchmark_fallback_reason=None),
    lambda raw: raw["cases"][0].update(benchmark_effective_repeats=2),
    lambda raw: raw["cases"][0]["samples_ms"].pop(),
    lambda raw: raw["cases"][0]["samples_ms"].__setitem__(0, float("nan")),
    lambda raw: raw["cases"][0].update(mean_ms=5.0),
    lambda raw: raw["cases"][0]["params"].update(x_shape=[1, 8192]),
    lambda raw: raw["cases"][0]["params"].update(dtype="torch.float32"),
    lambda raw: raw["cases"][0]["params"].update(regime="decode"),
    lambda raw: raw["cases"][0]["params"].update(eps=1e-5),
    lambda raw: raw["cases"][0]["validation"].update(outputs=["normed"]),
    lambda raw: raw["cases"][0]["validation"].update(checked_invocations=113),
    lambda raw: raw["cases"][0]["validation"].update(measured_graph_validated=False),
])
def test_report_rejects_changed_incomplete_or_ambiguous_cases(monkeypatch, tmp_path, mutation):
    env = environment(monkeypatch, tmp_path)
    raw = raw_report(measure(env))
    mutation(raw)
    with pytest.raises(ValueError):
        env[1]._benchmark_cases(raw)


def test_missing_materialized_helper_cannot_publish_performance(monkeypatch, tmp_path):
    runner = load("_rmsnorm_unmaterialized_runner", TASK / "scripts/task_runner.py")
    monkeypatch.setattr(runner, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("worker launched"))
    assert runner.run_performance({}, 1) == []
    report = json.loads((tmp_path / "performance_report.json").read_text())
    assert report["status"] == "fail" and report["test_cases"] == []


def test_baseline_hash_verified_before_loading(monkeypatch, tmp_path):
    bench = load("_rmsnorm_hash_bench", TASK / "scripts/_bench.py")
    baseline = tmp_path / "baseline.py.orig"
    baseline.write_text("raise RuntimeError('must not execute')\n")
    meta = {"source_provenance": {"baseline_ref": baseline.name, "source_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="hash mismatch"):
        bench.frozen_baseline(tmp_path, meta)
    baseline.write_text("def gemma_fused_add_rmsnorm(*args): return args\n")
    meta["source_provenance"]["source_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert bench.frozen_baseline(tmp_path, meta)(1, 2) == (1, 2)


def test_actual_input_geometry_is_checked_before_timing(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    env[5]["args"]["x"].shape = (1, 8192)
    with pytest.raises(RuntimeError, match="timed inputs differ"):
        measure(env)
    assert env[2].cuda.graphs == []


def test_failure_after_first_case_does_not_publish_partial_raw_report(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    bench, _, torch, cases, harness, case, spec, baseline, helper = env
    out = tmp_path / "raw.json"
    out.write_text('{"stale": true}')
    ut = tmp_path / "ut"
    ut.mkdir()
    (ut / "meta.json").write_text(json.dumps(META))
    fake_cases = types.SimpleNamespace(timing_cases=lambda h, m: [
        {"sig": item["sig"], "regime": item["regime"], "m": item["m"]}
        for item in META["workload"]["cases"]])
    monkeypatch.setattr(bench, "_load", lambda name, path: fake_cases if path.name == "cases.py" else harness)
    monkeypatch.setattr(bench, "frozen_baseline", lambda u, m: baseline)
    calls = []

    def fail_second(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("second case failed")
        return {"sig": "prefill_m8192_live_shape|prefill", "mean_ms": 1.0}

    monkeypatch.setattr(bench, "measure_case", fail_second)
    monkeypatch.setattr(sys, "argv", ["_bench.py", "--ut", str(ut), "--out", str(out)])
    with pytest.raises(RuntimeError, match="second case failed"):
        bench.main()
    assert not out.exists()
