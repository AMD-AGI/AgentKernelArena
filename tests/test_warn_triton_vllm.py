"""CPU checks for measured samples and exact bound replay in vLLM tasks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MOE_TASKS = ("triton_fused_moe", "triton_fused_moe_gptq_awq", "triton_moe_mmk")
RUNNER_TASKS = ("triton_fla_chunk_fwd_o", "triton_ssd_bmm", "triton_kda_gate", "triton_kda_gla_fwd_o")


def load(path):
    spec = importlib.util.spec_from_file_location("_checked_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTimedRun:
    def __init__(self):
        self.after_sample = None
        self.outputs = None
        self.bound = False
        self.replay = None

    def rerun(self):
        self.outputs = self.replay()
        return self.outputs


@pytest.mark.parametrize("name", MOE_TASKS)
@pytest.mark.parametrize("wrong_measured,wrong_replay", ((False, False), (True, False), (False, True)))
def test_moe_sample_and_replay_are_independent(monkeypatch, name, wrong_measured, wrong_replay):
    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = FakeTimedRun
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)
    checks = load(ROOT / "tasks/triton2triton/vllm" / name / "_contract_checks.py")
    inputs = {"A": torch.tensor([2.0]), "B": torch.tensor([3.0])}
    def reference(saved):
        return saved["A"] + saved["B"]
    def fn():
        return inputs["A"] + inputs["B"]
    def benchmark(fn, *, timed_run, repetition, warmup):
        assert repetition == 3 and warmup == 2
        output = fn()
        for index in range(repetition):
            sample = output + 1 if wrong_measured and index == 1 else output
            timed_run.after_sample(sample)
        timed_run.outputs = output
        timed_run.bound = True
        timed_run.replay = lambda: output + 1 if wrong_replay else fn()
        return .125, {"benchmark_method": "cuda_graph"}
    call = lambda: checks.checked_benchmark(
        benchmark, fn, inputs=inputs, reference=reference,
        check=checks.compare_output,
        perturb=lambda saved: {"A": -saved["A"], "B": saved["B"]},
        warmup=2, repetition=3)
    if wrong_measured or wrong_replay:
        with pytest.raises(AssertionError):
            call()
    else:
        ms, metadata = call()
        assert ms == .125 and metadata["benchmark_measured_samples_checked"] == 3
        assert metadata["benchmark_replay_checked"]
    torch.testing.assert_close(inputs["A"], torch.tensor([2.0]))


@pytest.mark.parametrize("name", RUNNER_TASKS)
def test_runner_checker_rejects_wrong_sample_and_missing_samples(name):
    checks = load(ROOT / "tasks/triton2triton/vllm" / name / "scripts/contract_checks.py")
    source = torch.tensor([2.0])
    readonly = checks.InputSnapshot({"source": source})
    timed = SimpleNamespace(bound=True, outputs=torch.tensor([5.0]))
    checks.observe_measured_samples(timed, readonly, lambda: torch.tensor([5.0]), atol=0, rtol=0)
    with pytest.raises(checks.ContractFailure, match="not all checked"):
        checks.validate_timed(timed, readonly, lambda: torch.tensor([5.0]),
                              lambda: source.add_(1), atol=0, rtol=0, expected_samples=1)
    with pytest.raises(checks.NumericalMismatch):
        timed.after_sample(torch.tensor([6.0]))
    assert timed.sample_checks == 0
    timed.after_sample(torch.tensor([5.0]))
    assert timed.sample_checks == 1


@pytest.mark.parametrize("name", RUNNER_TASKS)
def test_runner_sample_observer_rejects_mutated_input(name):
    checks = load(ROOT / "tasks/triton2triton/vllm" / name / "scripts/contract_checks.py")
    source = torch.tensor([2.0])
    readonly = checks.InputSnapshot({"source": source})
    timed = SimpleNamespace(bound=True, outputs=torch.tensor([5.0]))
    checks.observe_measured_samples(timed, readonly, lambda: torch.tensor([5.0]), atol=0, rtol=0)
    source.add_(1)
    if name in ("triton_fla_chunk_fwd_o", "triton_kda_gla_fwd_o"):
        with pytest.raises(checks.ContractFailure, match="Read-only input changed"):
            timed.after_sample(torch.tensor([5.0]))
        assert timed.sample_checks == 0
    else:
        timed.after_sample(torch.tensor([5.0]))
        assert timed.sample_checks == 1
    with pytest.raises(checks.ContractFailure,
                       match="not all checked" if timed.sample_checks == 0 else "Read-only input changed"):
        checks.validate_timed(timed, readonly, lambda: torch.tensor([5.0]),
                              lambda: source.add_(1), atol=0, rtol=0, expected_samples=1)
