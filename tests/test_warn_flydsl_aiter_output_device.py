"""Retain output provenance when AITER timed samples are copied for checking."""

import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gemm_a16w16_nt_n6144_k1536", "gemm_a16w16_nt_n7168_k512")
MODULES = ("task_measure", "task_baseline", "task_compare", "task_inputs", "task_reference")


@pytest.mark.parametrize("task", TASKS)
def test_timed_output_source_device_survives_host_copy(task, monkeypatch):
    scripts = ROOT / "tasks" / "Aiter-task" / task / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    for name in MODULES:
        sys.modules.pop(name, None)
    try:
        measure = importlib.import_module("task_measure")
        value = torch.tensor([[2.0, 3.0]])
        kept = measure.host_copy(value)
        assert kept.source_device == value.device
        assert torch.equal(kept.value, value)

        monkeypatch.setattr(measure.task_inputs, "load_draw",
                            lambda inputs, draw: inputs["a"].copy_(draw["a"]))
        monkeypatch.setattr(measure.task_inputs, "call_kwargs", lambda inputs: inputs)
        monkeypatch.setattr(measure.task_reference, "run", lambda a: a * 2)
        monkeypatch.setattr(measure, "compare_output", lambda got, expected: {
            "status": "PASS" if torch.equal(got, expected) else "FAIL",
            "failure_kind": "numerical_mismatch", "reason": "incorrect value"})
        draw = {"a": torch.tensor([[1.0, 1.5]])}
        inputs = {"a": torch.zeros_like(draw["a"])}
        good = measure.verify_timed_outputs(inputs, [(draw, kept)])
        assert good["status"] == "PASS"

        forged_device = measure.MeasuredOutput(kept.value, torch.device("cuda:0"))
        bad = measure.verify_timed_outputs(inputs, [(draw, forged_device)])
        assert bad["status"] == "FAIL"
        assert bad["failure_kind"] == "output_contract"
    finally:
        for name in MODULES:
            sys.modules.pop(name, None)
