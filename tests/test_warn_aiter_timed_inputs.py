"""CPU controls for every invocation's timed GEMM input immutability."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import types

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
def task_measure(name, monkeypatch):
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "tasks/Aiter-task" / name / "scripts"))
        for module in MODULES:
            patch.delitem(sys.modules, module, raising=False)
        try:
            yield importlib.import_module("task_measure")
        finally:
            for module in MODULES:
                sys.modules.pop(module, None)


class CorrectThenMutate:
    def __init__(self, reference, target=None, operand="a"):
        self.reference = reference
        self.target = target
        self.operand = operand
        self.phase = None
        self.index = None
        self.cost = 1.0

    def __call__(self, a, b):
        output = self.reference.run(a=a, b=b)
        if (self.phase, self.index) == self.target:
            {"a": a, "b": b}[self.operand].view(-1)[0].add_(1)
        return output


def simulated_timer(kernel):
    """Prepare outside each sample, observe only after it, including unseen reruns."""
    def benchmark(fn, *, warmup, repetition, target_ms, prepare_fn, timed_run):
        del target_ms
        for index in range(warmup + 3):  # warmup, estimate/capture, prime
            kernel.phase, kernel.index = "warmup", index
            prepare_fn()
            fn()
        for index in range(repetition):
            kernel.phase, kernel.index = "sample", index
            prepare_fn()
            output = fn()
            timed_run.after_sample(output)
        unseen_index = 0
        def rerun_ms():
            nonlocal unseen_index
            kernel.phase, kernel.index = "unseen", unseen_index
            unseen_index += 1
            prepare_fn()
            timed_run.outputs = fn()
            return kernel.cost
        timed_run.bound = True
        timed_run.outputs = output
        timed_run.rerun_ms = rerun_ms
        return kernel.cost, {"benchmark_method": "cuda_graph",
                             "benchmark_effective_repeats": 1}
    return types.SimpleNamespace(
        TimedRun=lambda: types.SimpleNamespace(bound=False, outputs=None),
        benchmark_cuda_graph_or_events=benchmark)


def run_case(measure, monkeypatch, target=None, operand="a", role="candidate"):
    torch = pytest.importorskip("torch")
    kernel = CorrectThenMutate(measure.task_reference, target, operand)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(measure.task_inputs, "build_case_inputs", lambda case: {
        "a": torch.tensor([[1., -2., 3.]], dtype=torch.bfloat16),
        "b": torch.tensor([[2., 1., -1.], [1., -1., 2.]], dtype=torch.bfloat16),
    })
    monkeypatch.setattr(measure, "fresh_draw_seeds", lambda count: list(range(1, count + 1)))
    monkeypatch.setattr(measure, "choose_checked_samples",
                        lambda repetition, count: list(range(count)))
    monkeypatch.setattr(measure.task_inputs, "call_varying_draws",
                        lambda inputs, seeds: [{"a": torch.full_like(inputs["a"], seed)}
                                               for seed in seeds])
    if role == "baseline":
        monkeypatch.setattr(measure.task_baseline, "run", kernel)
    monkeypatch.setitem(sys.modules, "_aka_benchmark", simulated_timer(kernel))
    return measure.time_case({"m": 1}, role=role,
                             launch=kernel if role == "candidate" else None)


@pytest.mark.parametrize("name", TASKS)
def test_middle_measured_activation_mutation_is_caught_before_next_draw(name, monkeypatch):
    with task_measure(name, monkeypatch) as measure:
        with monkeypatch.context() as patch:
            with pytest.raises(RuntimeError, match="protected input tensor: a"):
                run_case(measure, patch, ("sample", 37))


@pytest.mark.parametrize("phase,index,operand", [
    ("warmup", 4, "a"), ("warmup", 21, "b"),
    ("sample", 99, "a"), ("unseen", 1, "a"),
    ("unseen", 3, "b"),
])
def test_warmup_capture_final_sample_and_unseen_mutations_fail(phase, index, operand, monkeypatch):
    with task_measure("gemm_a16w16_nt_n2624_k6144", monkeypatch) as measure:
        with monkeypatch.context() as patch:
            with pytest.raises(RuntimeError, match=f"protected input tensor: {operand}"):
                run_case(measure, patch, (phase, index), operand)


def test_honest_reported_and_unseen_paths_have_one_guard_per_invocation(monkeypatch):
    with task_measure("gemm_a16w16_nt_n2624_k6144", monkeypatch) as measure:
        with monkeypatch.context() as patch:
            result = run_case(measure, patch)
        assert result["status"] == "PASS", result.get("reason")
        assert result["metadata"]["guarded_invocations"] == (
            measure.task_inputs.BENCH_WARMUP + 3
            + measure.task_inputs.BENCH_REPETITION + measure.UNSEEN_DRAWS)
        assert result["metadata"]["timed_output_correctness"]["metadata"]["checked_invocations"] == (
            measure.CHECKED_SAMPLES + measure.UNSEEN_DRAWS)


def test_baseline_uses_identical_input_guard(monkeypatch):
    with task_measure("gemm_a16w16_nt_n2624_k6144", monkeypatch) as measure:
        with monkeypatch.context() as patch:
            with pytest.raises(RuntimeError, match="protected input tensor: a"):
                run_case(measure, patch, ("sample", 37), role="baseline")
