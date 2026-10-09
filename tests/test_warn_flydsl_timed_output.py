"""The Event-timed FlyDSL tasks must report checks of measured outputs."""

import importlib.util
from pathlib import Path

import pytest
import torch


TASKS = (
    "gemm_a8w8_per_token_scale_kernel",
    "gemm_afp4wfp4_kernel",
    "gemm_afp8wfp8_kernel",
    "hgemm_kernel",
    "gemm_a16w8_blockscale_kernel",
    "gemm_a16wfp4_kernel",
    "gemm_a4w4_kernel",
    "gemm_a8w8_bpreshuffle_kernel",
    "batched_gemm_a8w8_kernel",
)
ROOT = Path(__file__).resolve().parents[1]


def _checks(task):
    path = ROOT / "tasks" / "torch2flydsl" / task / "scripts" / "replay_checks.py"
    spec = importlib.util.spec_from_file_location(f"replay_checks_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EventRun:
    bound = True

    def __init__(self, a, b, *, stale=False):
        self.a, self.b = a, b
        self.outputs = a * b
        self.stale = stale
        self.reruns = 0

    def rerun(self):
        self.reruns += 1
        return self.outputs if self.stale else self.a * self.b


@pytest.mark.parametrize("task", TASKS)
def test_measured_output_check_is_reported_and_replay_uses_new_input(task):
    checks = _checks(task)
    a = torch.tensor([2.0, -3.0])
    b = torch.tensor([4.0, 5.0])
    originals = (a.clone(), b.clone())
    timed = _EventRun(a, b)
    result = checks.verify_timed_run(
        timed, inputs=(a, b), originals=originals, expected=a * b,
        perturb=lambda: a.add_(1), reference=lambda: a * b,
        compare=lambda got, expected: checks.allclose_output(
            got, expected, atol=0.0, rtol=0.0),
    )
    assert result["timed_output_checked"] is True
    assert result["timed_output_correctness"] == "PASS"
    assert result["replay_correctness"] == "PASS"
    assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])


@pytest.mark.parametrize("task", TASKS)
def test_stale_replay_is_rejected_and_inputs_restored(task):
    checks = _checks(task)
    a = torch.tensor([2.0, -3.0])
    b = torch.tensor([4.0, 5.0])
    originals = (a.clone(), b.clone())
    with pytest.raises(AssertionError):
        checks.verify_timed_run(
            _EventRun(a, b, stale=True), inputs=(a, b), originals=originals,
            expected=a * b, perturb=lambda: a.add_(1), reference=lambda: a * b,
            compare=lambda got, expected: checks.allclose_output(
                got, expected, atol=0.0, rtol=0.0),
        )
    assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])


@pytest.mark.parametrize("task", TASKS)
def test_incorrect_measured_output_fails_before_a_correct_rerun(task):
    checks = _checks(task)
    a = torch.tensor([2.0, -3.0])
    b = torch.tensor([4.0, 5.0])
    originals = (a.clone(), b.clone())
    timed = _EventRun(a, b)
    timed.outputs.zero_()
    with pytest.raises(AssertionError):
        checks.verify_timed_run(
            timed, inputs=(a, b), originals=originals, expected=a * b,
            perturb=lambda: a.add_(1), reference=lambda: a * b,
            compare=lambda got, expected: checks.allclose_output(
                got, expected, atol=0.0, rtol=0.0),
        )
    assert timed.reruns == 0
    assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])
