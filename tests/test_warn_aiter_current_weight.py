"""CPU regressions for unscored current-weight checks in rotating GEMM tasks."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "gemm_a16w16_nt_n128_k6144", "gemm_a16w16_nt_n2048_k2048",
    "gemm_a16w16_nt_n256_k6144", "gemm_a16w16_nt_n2624_k6144",
    "gemm_a16w16_nt_n28672_k512", "gemm_a16w16_nt_n3072_k6144",
    "gemm_a16w16_nt_n32_k6144", "gemm_a16w16_nt_n3584_k512",
    "gemm_a16w16_nt_n6144_k1536", "gemm_a16w16_nt_n6144_k16384",
    "gemm_a16w16_nt_n6144_k2048", "gemm_a16w16_nt_n6144_k4096",
    "gemm_a16w16_nt_n6144_k6144", "gemm_a16w16_nt_n7168_k512",
)
MODULES = ("task_contract", "task_inputs", "task_compare", "task_initialize",
           "task_reference", "task_baseline", "task_measure")


@contextmanager
def task_modules(name, monkeypatch):
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "tasks/Aiter-task" / name / "scripts"))
        for module in MODULES:
            patch.delitem(sys.modules, module, raising=False)
        measure = importlib.import_module("task_measure")
        yield measure
        for module in MODULES:
            sys.modules.pop(module, None)


def _run(measure, monkeypatch, kernel):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])
    original_build = measure.task_inputs.build_case_inputs
    initial = {}

    def small_inputs(case):
        del case
        inputs = {"a": torch.tensor([[1., 2., -1.], [2., -1., 3.]], dtype=torch.bfloat16),
                  "b": torch.tensor([[2., 0., 1.], [-1., 3., 2.],
                                     [1., -1., -1.], [0., 2., 0.]], dtype=torch.bfloat16)}
        initial.update({key: value.clone() for key, value in inputs.items()})
        return inputs

    monkeypatch.setattr(measure.task_inputs, "build_case_inputs", small_inputs)
    try:
        result = measure.check_case({"m": 2}, role="candidate", launch=kernel)
    finally:
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", original_build)
    return result, initial


class Honest:
    def __init__(self, reference):
        self.reference = reference
        self.calls = []

    def __call__(self, a, b):
        self.calls.append((b.data_ptr(), b._version))
        return self.reference.run(a=a, b=b)


class StaleShapeWeight(Honest):
    def __call__(self, a, b):
        if not hasattr(self, "cached"):
            self.cached = b.clone()
        self.calls.append((b.data_ptr(), b._version))
        return self.reference.run(a=a, b=self.cached)


class StaleNewStorageWeight(Honest):
    def __call__(self, a, b):
        if not hasattr(self, "first_ptr"):
            self.first_ptr = b.data_ptr()
        if b.data_ptr() == self.first_ptr:
            self.cached = b.clone()  # Handles in-place updates, but not a new tensor.
        self.calls.append((b.data_ptr(), b._version))
        return self.reference.run(a=a, b=self.cached)


class MutatesWeightOnReplay(Honest):
    def __call__(self, a, b):
        result = super().__call__(a, b)
        if len(self.calls) == 2:
            b.add_(1)
        return result


@pytest.mark.parametrize("name", TASKS)
def test_current_weight_replays_reject_stale_values_and_accept_honest_launch(name, monkeypatch):
    pytest.importorskip("torch")
    with task_modules(name, monkeypatch) as measure:
        for cls, expected in ((Honest, "PASS"), (StaleShapeWeight, "FAIL"),
                              (StaleNewStorageWeight, "FAIL")):
            kernel = cls(measure.task_reference)
            with monkeypatch.context() as patch:
                result, _ = _run(measure, patch, kernel)
            assert result["status"] == expected, (name, cls.__name__, result)
            if expected == "PASS":
                replay = result["metadata"]["current_weight_replays"]
                assert replay["status"] == "PASS"
                checks = replay["metadata"]
                assert [row["mode"] for row in checks] == ["in_place", "new_storage"]
                assert len(kernel.calls) == 3
                assert kernel.calls[0][0] == kernel.calls[1][0]
                assert kernel.calls[1][0] != kernel.calls[2][0]
            else:
                assert result["failure_kind"] == "numerical_mismatch"
                mode = "in_place" if cls is StaleShapeWeight else "new_storage"
                assert f"Current-weight {mode}" in result["reason"]


def test_baseline_uses_the_same_current_weight_checks(monkeypatch):
    torch = pytest.importorskip("torch")
    with task_modules("gemm_a16w16_nt_n6144_k16384", monkeypatch) as measure:
        seen = {}
        def build(case):
            del case
            seen["inputs"] = {"a": torch.tensor([[1., 2., -1.]], dtype=torch.bfloat16),
                              "b": torch.tensor([[2., 0., 1.], [-1., 3., 2.]], dtype=torch.bfloat16)}
            return seen["inputs"]
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", build)
        monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])
        monkeypatch.setattr(measure.task_baseline, "run", measure.task_reference.run)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        result = measure.check_case({"m": 1}, role="baseline")
        assert result["status"] == "PASS"
        assert [row["mode"] for row in result["metadata"]["current_weight_replays"]["metadata"]] == [
            "in_place", "new_storage"]


def test_baseline_numerical_diagnostic_still_runs_weight_controls(monkeypatch):
    torch = pytest.importorskip("torch")
    with task_modules("gemm_a16w16_nt_n256_k6144", monkeypatch) as measure:
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", lambda case: {
            "a": torch.tensor([[1., 2., -1.]], dtype=torch.bfloat16),
            "b": torch.tensor([[2., 0., 1.], [-1., 3., 2.]], dtype=torch.bfloat16),
        })
        monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])
        monkeypatch.setattr(measure.task_baseline, "run", measure.task_reference.run)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        compare = measure.compare_output
        calls = []
        def diagnostic(got, expected):
            calls.append(1)
            if len(calls) == 1:
                return {"status": "FAIL", "failure_kind": "numerical_mismatch",
                        "reason": "existing baseline diagnostic", "metadata": {}}
            return compare(got, expected)
        monkeypatch.setattr(measure, "compare_output", diagnostic)
        result = measure.check_case({"m": 1}, role="baseline")
        assert result["status"] == "FAIL"
        assert result["reason"] == "existing baseline diagnostic"
        assert len(calls) == 3
        assert result["metadata"]["current_weight_replays"]["status"] == "PASS"


def test_weight_replay_failure_restores_original_buffers(monkeypatch):
    torch = pytest.importorskip("torch")
    with task_modules("gemm_a16w16_nt_n6144_k16384", monkeypatch) as measure:
        seen = {}
        original_build = measure.task_inputs.build_case_inputs
        def build(case):
            del case
            seen["inputs"] = {"a": torch.tensor([[1., 2., -1.]], dtype=torch.bfloat16),
                              "b": torch.tensor([[2., 0., 1.], [-1., 3., 2.]], dtype=torch.bfloat16)}
            seen["original"] = measure.input_snapshot(seen["inputs"])
            return seen["inputs"]
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", build)
        monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        try:
            result = measure.check_case({"m": 1}, role="candidate",
                                        launch=StaleShapeWeight(measure.task_reference))
        finally:
            monkeypatch.setattr(measure.task_inputs, "build_case_inputs", original_build)
        assert result["status"] == "FAIL"
        measure.assert_inputs_unchanged(seen["inputs"], seen["original"])


def test_input_mutation_after_correct_replay_is_detected_and_restored(monkeypatch):
    torch = pytest.importorskip("torch")
    with task_modules("gemm_a16w16_nt_n6144_k16384", monkeypatch) as measure:
        seen = {}
        def build(case):
            del case
            seen["inputs"] = {"a": torch.tensor([[1., 2., -1.]], dtype=torch.bfloat16),
                              "b": torch.tensor([[2., 0., 1.], [-1., 3., 2.]], dtype=torch.bfloat16)}
            seen["original"] = measure.input_snapshot(seen["inputs"])
            return seen["inputs"]
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", build)
        monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        with pytest.raises(RuntimeError, match="protected input tensor: b"):
            measure.check_case({"m": 1}, role="candidate",
                               launch=MutatesWeightOnReplay(measure.task_reference))
        measure.assert_inputs_unchanged(seen["inputs"], seen["original"])
