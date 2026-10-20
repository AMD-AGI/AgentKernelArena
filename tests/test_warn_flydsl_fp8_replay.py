"""FP8 graph replay must exercise every read-only operand independently."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/flydsl2flydsl/fp8_gemm_8wave_kernel"


@pytest.fixture
def harness(monkeypatch):
    helper = types.ModuleType("_aka_benchmark")
    helper.TimedRun = type("TimedRun", (), {})
    helper.benchmark_cuda_graph_or_events = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "_aka_benchmark", helper)
    monkeypatch.syspath_prepend(str(TASK))
    for name in ("scripts", "scripts.replay_checks", "task_runtime", "test_kernel_harness"):
        sys.modules.pop(name, None)
    try:
        module = importlib.import_module("test_kernel_harness")
        yield module, importlib.import_module("scripts.replay_checks")
    finally:
        for name in ("scripts", "scripts.replay_checks", "task_runtime", "test_kernel_harness"):
            sys.modules.pop(name, None)


class _GraphRun:
    bound = True

    def __init__(self, operands, *, cached=None):
        self.operands = operands
        self.cached = cached
        self.original = {name: value.clone() for name, value in operands.items()}
        self.outputs = self._compute()

    def _compute(self):
        values = {name: (self.original[name] if name == self.cached else value)
                  for name, value in self.operands.items()}
        return ((values["A"] @ values["B_T"].T)
                * values["A_scale"][:, None] * values["B_scale"][None, :])

    def rerun(self):
        self.outputs.copy_(self._compute())
        return self.outputs


def _case(harness, *, cached=None, bad_measured=False):
    h, checks = harness
    operands = {
        "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "B_T": torch.tensor([[2.0, 1.0], [1.0, 3.0]]),
        "A_scale": torch.tensor([1.0, 1.5]),
        "B_scale": torch.tensor([0.75, 1.25]),
    }
    a, b, sa, sb = (operands[name] for name in ("A", "B_T", "A_scale", "B_scale"))
    inputs = (a, b, b, sa, sb)  # Current declared layout has unshuffled B.
    originals = tuple(value.clone() for value in inputs)
    timed = _GraphRun(operands, cached=cached)
    expected = timed.outputs.clone()
    if bad_measured:
        timed.outputs.zero_()
    kwargs = dict(
        inputs=inputs, originals=originals, expected=expected,
        perturbations=h._replay_perturbations(None, a, b, b, sa, sb),
        reference=lambda: (a @ b.T) * sa[:, None] * sb[None, :],
        compare=lambda got, ref: checks.allclose_output(
            got, ref, atol=0.0, rtol=0.0),
    )
    return checks, timed, kwargs, originals


def test_each_operand_is_checked_on_replay(harness):
    checks, timed, kwargs, originals = _case(harness)
    result = checks.verify_timed_run(timed, **kwargs)
    assert result["replay_operands_checked"] == ["A", "B_T", "A_scale", "B_scale"]
    assert result["timed_output_checked"] is True
    assert all(torch.equal(value, original)
               for value, original in zip(kwargs["inputs"], originals))


@pytest.mark.parametrize("cached", ["A", "B_T", "A_scale", "B_scale"])
def test_cached_operand_replay_fails_without_success_evidence(harness, cached):
    checks, timed, kwargs, originals = _case(harness, cached=cached)
    with pytest.raises(AssertionError):
        checks.verify_timed_run(timed, **kwargs)
    assert all(torch.equal(value, original)
               for value, original in zip(kwargs["inputs"], originals))


def test_wrong_measured_output_fails_before_correct_replay(harness):
    checks, timed, kwargs, _ = _case(harness, bad_measured=True)
    with pytest.raises(AssertionError):
        checks.verify_timed_run(timed, **kwargs)
