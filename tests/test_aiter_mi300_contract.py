"""CPU controls for the MI300X task declarations and measured-path contract."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

from src.task_protocol import CaseManifest, baseline_correctness_accepted, parse_command_result
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = tuple(sorted((ROOT / "tasks/Aiter-task-mi300").glob("*/config.yaml")))
MODULES = ("task_contract", "task_inputs", "task_compare", "task_initialize",
           "task_reference", "task_baseline", "task_measure", "task_validation", "evaluate")


@contextmanager
def task_modules(config, monkeypatch):
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(config.parent / "scripts"))
        for name in MODULES:
            patch.delitem(sys.modules, name, raising=False)
        try:
            yield importlib.import_module("task_contract")
        finally:
            for name in MODULES:
                sys.modules.pop(name, None)


def parsed(report):
    return parse_command_result("ARENA_EVAL_RESULT=" + json.dumps(report),
                                role=report["role"], action=report["action"],
                                returncode=0 if report["status"] == "PASS" else 1)


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
def test_new_platform_requires_complete_correct_baseline(config, monkeypatch):
    spec = load_task_spec(config, task_id=str(config.parent.relative_to(ROOT / "tasks")))
    with task_modules(config, monkeypatch) as contract:
        rows = contract.case_manifest(contract.load_workload())
        assert len(rows) == 13
        manifest = CaseManifest.from_result(parsed({
            "protocol": "arena-eval-v1", "role": "task", "action": "validate-task",
            "status": "PASS", "cases": rows, "metadata": {"candidate_state": "implemented"}}))
        failures = deepcopy(rows)
        for row in failures:
            row.pop("checks")
            row.update(status="FAIL", failure_kind="numerical_mismatch", reason="injected finite mismatch")
        failed = parsed({"protocol": "arena-eval-v1", "role": "baseline", "action": "correctness",
                         "status": "FAIL", "failure_kind": "numerical_mismatch",
                         "reason": "injected finite mismatch", "cases": failures})
        assert not baseline_correctness_accepted(
            failed, baseline=spec.baseline, phase="task_validation", manifest=manifest)
        for row in failures:
            row.update(status="PASS")
            row.pop("failure_kind")
            row.pop("reason")
        passed = parsed({"protocol": "arena-eval-v1", "role": "baseline", "action": "correctness",
                         "status": "PASS", "cases": failures})
        assert baseline_correctness_accepted(
            passed, baseline=spec.baseline, phase="task_validation", manifest=manifest)


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
def test_implemented_initial_state_checks_actual_builder(config, tmp_path, monkeypatch):
    with task_modules(config, monkeypatch) as contract:
        runner = importlib.import_module("evaluate")
        declaration = contract.load_config()
        monkeypatch.setenv("ARENA_EVAL_PHASE", "task_validation")
        assert runner.check_initial_state(declaration) == "implemented"
        monkeypatch.setattr(contract, "ROOT", tmp_path)
        with pytest.raises(RuntimeError, match="declared implemented.*missing"):
            runner.check_initial_state(declaration)
        (tmp_path / "kernel.py").write_text('"""An empty generation target."""\n')
        with pytest.raises(RuntimeError, match="declared implemented.*defines no"):
            runner.check_initial_state(declaration)


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
def test_runtime_platform_gate_agrees_with_declared_gfx942(config, monkeypatch):
    torch = pytest.importorskip("torch")
    spec = load_task_spec(config, task_id=str(config.parent.relative_to(ROOT / "tasks")))
    assert spec.to_mapping()["platform_support"]["required_arch"] == "gfx942"
    with task_modules(config, monkeypatch):
        runner = importlib.import_module("evaluate")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.version, "hip", "injected CPU runtime")
        monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
        monkeypatch.setitem(sys.modules, "flydsl", types.ModuleType("flydsl"))
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda _: types.SimpleNamespace(gcnArchName="gfx950:sramecc+"))
        with pytest.raises(RuntimeError, match="requires gfx942; found gfx950"):
            runner.require_runtime({"op_type": "gemm"})
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda _: types.SimpleNamespace(gcnArchName="gfx942:sramecc+"))
        runner.require_runtime({"op_type": "gemm"})


class Kernel:
    def __init__(self, reference, *, wrong=False, mutation=None):
        self.reference, self.wrong, self.mutation = reference, wrong, mutation
        self.phase, self.index = None, None

    def __call__(self, a, b):
        output = self.reference.run(a=a, b=b)
        if self.wrong:
            output = -output
        if self.mutation and (self.phase, self.index) == self.mutation[:2]:
            {"a": a, "b": b}[self.mutation[2]].view(-1)[0].add_(1)
        return output


def simulated_cold_timer(kernel, calls):
    class Flush:
        def __init__(self, nbytes, *, device):
            calls["flush_bytes"], calls["device"] = nbytes, device

        def __call__(self):
            calls["flushes"] = calls.get("flushes", 0) + 1

    def benchmark(fn, *, cache_flush, warmup, repetition, prepare_fn, timed_run):
        calls["warmup"], calls["repetition"] = warmup, repetition

        def invoke(phase, index):
            kernel.phase, kernel.index = phase, index
            prepare_fn()
            cache_flush()
            return fn()

        for index in range(warmup):
            invoke("warmup", index)
        samples = [1.0 + index / 1000 for index in range(repetition)]
        for index in range(repetition):
            output = invoke("sample", index)
            timed_run.after_sample(output)
        unseen = 0

        def rerun_ms():
            nonlocal unseen
            timed_run.outputs = invoke("unseen", unseen)
            unseen += 1
            return 1.0

        timed_run.bound, timed_run.outputs = True, output
        timed_run.rerun_ms = rerun_ms
        return samples, {"benchmark_method": "cuda_graph_cold_cache",
                         "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
                         "benchmark_cache_flush_bytes": calls["flush_bytes"],
                         "benchmark_cache_flush_in_timing": False,
                         "benchmark_metric": "gpu_event_graph_span",
                         "benchmark_effective_repeats": 1}

    return types.SimpleNamespace(
        ReadOnlyCacheFlush=Flush,
        TimedRun=lambda: types.SimpleNamespace(bound=False, outputs=None),
        benchmark_cuda_graph_cold_cache_samples=benchmark)


def run_timed(measure, monkeypatch, *, role="candidate", wrong=False, mutation=None):
    torch = pytest.importorskip("torch")
    kernel = Kernel(measure.task_reference, wrong=wrong, mutation=mutation)
    calls = {}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(measure.task_inputs, "build_case_inputs", lambda _: {
        "a": torch.tensor([[1., -2., 3.]], dtype=torch.bfloat16),
        "b": torch.tensor([[2., 1., -1.], [1., -1., 2.]], dtype=torch.bfloat16)})
    monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: list(range(1, count + 1)))
    monkeypatch.setattr(measure, "choose_checked_samples", lambda repetition, count: list(range(count)))
    monkeypatch.setattr(measure.task_inputs, "call_varying_draws", lambda inputs, seeds: [
        {"a": torch.full_like(inputs["a"], seed)} for seed in seeds])
    if role == "baseline":
        monkeypatch.setattr(measure.task_baseline, "run", kernel)
    monkeypatch.setitem(sys.modules, "_aka_benchmark", simulated_cold_timer(kernel, calls))
    return measure.time_case({"m": 1}, role=role,
                             launch=kernel if role == "candidate" else None), calls


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_cold_measurement_keeps_samples_identity_and_actual_output_checks(config, role, monkeypatch):
    pytest.importorskip("torch")
    with task_modules(config, monkeypatch):
        measure = importlib.import_module("task_measure")
        result, calls = run_timed(measure, monkeypatch, role=role)
        assert result["status"] == "PASS", result.get("reason")
        assert calls["warmup"] == 20 and calls["repetition"] == 100
        assert calls["flush_bytes"] == 1 << 30
        assert calls["flushes"] == 20 + 100 + measure.UNSEEN_DRAWS
        metadata = result["metadata"]
        samples = metadata["samples_ms"]
        assert len(samples) == 100
        assert result["execution_time_ms"] == pytest.approx(sum(samples) / len(samples))
        assert result["benchmark_method"] == metadata["timing"]["benchmark_method"] == "cuda_graph_cold_cache"
        assert metadata["benchmark_cache_flush_bytes"] == 1 << 30
        assert metadata["benchmark_cache_flush_in_timing"] is False
        checked = metadata["timed_output_correctness"]["metadata"]["checked_invocations"]
        assert checked == measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS
        assert metadata["guarded_invocations"] == calls["flushes"]


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_wrong_measured_outputs_fail_for_both_roles(config, role, monkeypatch):
    pytest.importorskip("torch")
    with task_modules(config, monkeypatch):
        result, _ = run_timed(importlib.import_module("task_measure"), monkeypatch, role=role, wrong=True)
        assert result["status"] == "FAIL"
        assert result["failure_kind"] == "numerical_mismatch"


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("mutation", [("warmup", 1, "b"), ("sample", 37, "a"),
                                      ("sample", 99, "a"), ("unseen", 3, "b")])
def test_rotating_input_guards_still_cover_cold_samples_and_unseen_reruns(config, mutation, monkeypatch):
    pytest.importorskip("torch")
    with task_modules(config, monkeypatch):
        with pytest.raises(RuntimeError, match=f"protected input tensor: {mutation[2]}"):
            run_timed(importlib.import_module("task_measure"), monkeypatch, mutation=mutation)


@pytest.mark.parametrize("config", TASKS, ids=lambda p: p.parent.name)
def test_current_weight_replays_reject_stale_weight_and_restore_inputs(config, monkeypatch):
    torch = pytest.importorskip("torch")
    with task_modules(config, monkeypatch):
        measure = importlib.import_module("task_measure")
        original = {"a": torch.tensor([[1., -2., 3.]], dtype=torch.bfloat16),
                    "b": torch.tensor([[2., 1., -1.], [1., -1., 2.]], dtype=torch.bfloat16)}
        inputs = {name: value.clone() for name, value in original.items()}
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(measure.task_inputs, "build_case_inputs", lambda _: inputs)
        monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: [1, 2][:count])

        class StaleWeight:
            def __call__(self, a, b):
                if not hasattr(self, "weight"):
                    self.weight = b.clone()
                return measure.task_reference.run(a=a, b=self.weight)

        honest = measure.check_case({"m": 1}, role="candidate", launch=Kernel(measure.task_reference))
        assert honest["status"] == "PASS"
        replays = honest["metadata"]["current_weight_replays"]["metadata"]
        assert [row["mode"] for row in replays] == ["in_place", "new_storage"]
        stale = measure.check_case({"m": 1}, role="candidate", launch=StaleWeight())
        assert stale["status"] == "FAIL" and stale["failure_kind"] == "numerical_mismatch"
        for name, expected in original.items():
            assert torch.equal(inputs[name], expected)
