"""Every reported 8-wave FP8 sample must satisfy the measured-output oracle."""
import importlib.util
from pathlib import Path

import pytest
import torch


def _checks():
    path = (Path(__file__).resolve().parents[1] /
            "tasks/flydsl2flydsl/fp8_gemm_8wave_kernel/scripts/replay_checks.py")
    spec = importlib.util.spec_from_file_location("fp8_sample_checks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _observe(samples, wrong_index=None, mutate_index=None):
    checks = _checks()
    inputs = (torch.tensor([2.0]),)
    originals = (inputs[0].clone(),)
    expected = torch.tensor([6.0])
    timed = type("Timed", (), {})()
    checked = checks.observe_measured_samples(
        timed, inputs=inputs, originals=originals, expected=expected,
        compare=lambda actual, ref: torch.testing.assert_close(
            actual, ref, rtol=0, atol=0),
    )
    for index in range(samples):
        if index == mutate_index:
            inputs[0].add_(1)
        timed.after_sample(torch.tensor([-99.0]) if index == wrong_index
                           else expected.clone())
    return checks, checked


def test_wrong_middle_sample_rejected_even_if_final_sample_correct():
    with pytest.raises(AssertionError):
        _observe(100, wrong_index=42)


def test_mutated_input_rejected_during_measured_sample():
    with pytest.raises(AssertionError, match="read-only"):
        _observe(100, mutate_index=42)


def test_all_samples_and_benchmark_count_required():
    checks, checked = _observe(100)
    assert checks.require_sample_count(checked, {"benchmark_samples": 100}, 100) == {
        "validated_sample_count": 100}
    with pytest.raises(AssertionError, match="not all checked"):
        checks.require_sample_count(checked, {"benchmark_samples": 99}, 100)
