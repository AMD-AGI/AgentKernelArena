"""CPU regression for the unscored second M-tile tail control."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_moe_mmk"


@contextmanager
def runner():
    path = TASK / "scripts/task_runner.py"
    previous_dir = Path.cwd()
    previous_path = list(sys.path)
    previous_checks = sys.modules.get("_contract_checks")
    had_checks = "_contract_checks" in sys.modules
    sys.modules.pop("_contract_checks", None)
    spec = importlib.util.spec_from_file_location("moe_mmk_second_tail_runner", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        os.chdir(previous_dir)
        sys.path[:] = previous_path
        if had_checks:
            sys.modules["_contract_checks"] = previous_checks
        else:
            sys.modules.pop("_contract_checks", None)


def test_original_scored_rows_and_existing_controls_are_retained():
    manifest = json.loads((TASK / "workloads.json").read_text())
    with runner() as task:
        assert task.TEST_SHAPES == [
            (32, 64, 32), (64, 128, 64), (128, 256, 128),
            (256, 512, 256), (512, 1024, 512)]
        assert task.WARMUP_ITERATIONS == 10
        assert task.BENCHMARK_ITERATIONS == 100
        assert task.CONTROL_CASES == (
            "rectangular_tail", "m_tail", "n_tail", "k_below64",
            "k_above64", "m_second_tail")
        assert [row["test_case_id"] for row in manifest["cases"] if
                "performance" in row["checks"]] == [f"perf{i}" for i in range(1, 6)]
        assert [row["params"]["control"] for row in manifest["cases"] if
                "control" in row["params"]] == list(task.CONTROL_CASES)
        new = manifest["cases"][-1]
        assert new["params"]["configuration"] == [65, 64, 64]
        assert new["checks"] == ["correctness"]


def test_second_tile_answer_is_nontrivial_and_rejects_dropped_final_row():
    with runner() as task:
        inputs = task.control_inputs("m_second_tail", "cpu")
        assert (inputs["A"].shape, inputs["B"].shape) == ((65, 64), (64, 64))
        assert inputs["A"].dtype == inputs["B"].dtype == torch.float16
        expected = task.reference(inputs)
        assert expected.shape == (65, 64)
        assert expected[64].abs().max() > 1
        task.compare_output(expected.clone(), expected)
        omitted = expected.clone()
        omitted[64].zero_()
        with pytest.raises(task.compare_output.__globals__["NumericalMismatch"]):
            task.compare_output(omitted, expected)


@pytest.mark.parametrize("mode", ["honest", "zero_second_tail", "copy_first_tile",
                                    "mutate_input"])
def test_public_correctness_route_checks_second_tile_and_readonly(monkeypatch, mode):
    with runner() as task:
        original_control_inputs = task.control_inputs
        monkeypatch.setattr(task, "control_inputs",
                            lambda name, _device: original_control_inputs(name, "cpu"))
        seen = []

        def candidate(a, b):
            seen.append((a.shape, b.shape))
            output = (a.float() @ b.float()).to(a.dtype)
            if mode == "zero_second_tail":
                output[64].zero_()
            elif mode == "copy_first_tile":
                output[64].copy_(output[0])
            elif mode == "mutate_input":
                a[64].add_(1)
            return output

        monkeypatch.setattr(task, "load_module",
                            lambda: SimpleNamespace(moe_matmul=candidate))
        ok, error = task.run_correctness(control="m_second_tail")
        assert seen == [(torch.Size([65, 64]), torch.Size([64, 64]))]
        assert ok is (mode == "honest")
        if mode == "mutate_input":
            assert "Read-only input changed" in str(error)
        elif mode != "honest":
            assert "Numerical mismatch" in str(error)
