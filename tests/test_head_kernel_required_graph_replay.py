"""Exercise run_correctness's required graph gate with deterministic fake CUDA.

Tensor arithmetic and the earlier random-parity check are substituted to isolate
capture/replay control flow. No GPU execution or numerical validation is claimed.
"""
from contextlib import contextmanager, nullcontext
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from head_kernel_test_utils import task_directory

SUITE = Path(__file__).resolve().parents[1] / "tasks/head_kernels"
HARNESS_PATHS = sorted(
    path for path in SUITE.rglob("ut/harness_lib.py")
    # Native GLM GEMM tasks have a separate graph runner, covered in their task tests.
    if {"run_correctness", "check_graph_replay"}.issubset({
        node.name for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    })
)


def load_harness(path):
    spec = importlib.util.spec_from_file_location("required_graph_harness_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, failure=None):
        self.failure = failure
        self.active_graph = None
        self.capture_count = 0
        self.replay_count = 0

    def is_available(self):
        return self.failure != "no_device"

    def synchronize(self):
        pass

    def Stream(self):
        return SimpleNamespace(wait_stream=lambda _: None)

    current_stream = Stream

    def stream(self, stream):
        return nullcontext()

    def CUDAGraph(self):
        cuda = self

        class Graph:
            operation = None

            def replay(self):
                cuda.replay_count += 1
                if cuda.failure == "replay_exception":
                    raise RuntimeError("injected replay fault")
                if cuda.failure != "stale_output":
                    self.operation()

        return Graph()

    @contextmanager
    def graph(self, graph):
        self.capture_count += 1
        if self.failure == "capture_exception":
            raise RuntimeError("injected capture failure")
        self.active_graph = graph
        try:
            yield
        finally:
            self.active_graph = None


def run_pipeline(harness, monkeypatch, failure=None, regime=None):
    cuda = FakeCuda(failure)
    monkeypatch.setattr(harness, "_torch", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(harness, "correct", lambda actual, expected, tol:
                        (actual == expected, float(abs(actual - expected))))
    parity_calls = []

    def passed_parity(*args, **kwargs):
        parity_calls.append(kwargs)
        return True, [{"case": "parity prerequisite", "correct": True}]

    monkeypatch.setattr(harness, "check_random_vs_baseline", passed_parity)
    state = {"input": None, "output": None}
    replay_cases = [
        {"sig": "large", "value": 7, "ref": 14},
        {"sig": "small", "value": 3, "ref": 6},
    ]

    def fill(case):
        state["input"] = case["value"]

    def compute():
        state["output"] = 2 * state["input"]
        if failure == "wrong_output" and cuda.replay_count:
            state["output"] += 100

    def run():
        if cuda.active_graph is not None:
            cuda.active_graph.operation = compute
        compute()

    result = harness.run_correctness(
        regime or {"cuda_graph": True, "compile": "eager"},
        eager_cases=[{"args": 7, "ref": 14, "sig": "eager prerequisite"}],
        current_call=lambda value: 2 * value,
        baseline_call=lambda value: 2 * value,
        random_shapes=[],
        tol=0.02,
        replay={"fill": fill, "run": run, "read_out": lambda: state["output"],
                "cases": replay_cases, "capture_idx": 0},
    )
    assert parity_calls, "the real run_correctness pipeline did not reach parity"
    assert result[1]["eager"][0]["correct"] is True
    assert result[1]["random"][0]["correct"] is True
    return result, cuda


@pytest.mark.parametrize("path", HARNESS_PATHS, ids=lambda path: path.parent.parent.name)
@pytest.mark.parametrize("failure", [
    "capture_exception", "no_device", "replay_exception", "stale_output", "wrong_output",
])
def test_required_graph_failure_cannot_pass_correctness(path, failure, monkeypatch):
    (ok, report), cuda = run_pipeline(load_harness(path), monkeypatch, failure)
    assert ok is False
    assert any(entry["correct"] is False for entry in report["graph_replay"])
    assert not any("skipped:" in entry["note"] for entry in report["graph_replay"])
    if failure in {"stale_output", "wrong_output", "replay_exception"}:
        assert cuda.replay_count == 2


@pytest.mark.parametrize("path", HARNESS_PATHS, ids=lambda path: path.parent.parent.name)
def test_valid_capture_replays_both_changed_inputs(path, monkeypatch):
    (ok, report), cuda = run_pipeline(load_harness(path), monkeypatch)
    assert ok is True
    assert [entry["case"] for entry in report["graph_replay"]] == ["large", "small"]
    assert all(entry["correct"] is True for entry in report["graph_replay"])
    assert cuda.capture_count == 1
    assert cuda.replay_count == 2


@pytest.mark.parametrize("path", HARNESS_PATHS, ids=lambda path: path.parent.parent.name)
def test_required_graph_empty_cases_fails_without_capture(path, monkeypatch):
    harness = load_harness(path)
    cuda = FakeCuda()
    monkeypatch.setattr(harness, "_torch", lambda: SimpleNamespace(cuda=cuda))
    def unexpected_call(*args):
        pytest.fail("an empty required replay cannot execute a graph")
    ok, report = harness.check_graph_replay(
        unexpected_call, unexpected_call, unexpected_call, [], 0.02)
    assert ok is False
    assert report[0]["correct"] is False
    assert "no CUDA / no cases" in report[0]["note"]
    assert cuda.capture_count == cuda.replay_count == 0


@pytest.mark.parametrize("task", [
    "glm-5.3-flash__elementwise_copy_cluster", "glm-5.3-flash__fused_moe_kernel",
])
def test_actual_eager_glm_contract_never_requires_graph_capture(task, monkeypatch):
    folder = task_directory(task) / "ut"
    regime = json.loads((folder / "meta.json").read_text())["regime"]
    harness = load_harness(folder / "harness_lib.py")
    assert not harness.deployment_graph_mode(regime)
    (ok, report), cuda = run_pipeline(harness, monkeypatch, "capture_exception", regime)
    assert ok is True
    assert "graph_replay" not in report
    assert cuda.capture_count == cuda.replay_count == 0
