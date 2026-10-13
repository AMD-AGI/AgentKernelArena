"""CPU-only controls for the task-local nonaligned-K known answers."""
from contextlib import contextmanager
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_ssd_bmm"


@contextmanager
def isolated_task_modules():
    """Load task-local scripts only during a test, then restore process imports."""
    previous_path = list(sys.path)
    previous_cwd = os.getcwd()
    previous_scripts = {name: value for name, value in sys.modules.items()
                        if name == "scripts" or name.startswith("scripts.")}
    for name in previous_scripts:
        sys.modules.pop(name)
    try:
        sys.path.insert(0, str(TASK))
        controls = importlib.import_module("scripts.semantic_controls")
        checks = importlib.import_module("scripts.contract_checks")
        spec = importlib.util.spec_from_file_location("ssd_task_runner_controls",
                                                    TASK / "scripts/task_runner.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        yield controls, checks, runner.reference_bmm
    finally:
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts."):
                sys.modules.pop(name)
        sys.modules.update(previous_scripts)
        sys.path[:] = previous_path
        os.chdir(previous_cwd)


@pytest.fixture
def task():
    with isolated_task_modules() as loaded:
        yield SimpleNamespace(semantic_controls=loaded[0], ContractFailure=loaded[1].ContractFailure,
                              NumericalMismatch=loaded[1].NumericalMismatch,
                              reference_bmm=loaded[2])


TAIL_NAMES = {"nonaligned_k_35_tail_causal_False", "nonaligned_k_65_tail_causal_True"}


def tail_cases(task):
    return [c for c in task.semantic_controls.control_cases("cpu") if c["name"] in TAIL_NAMES]


def correct_wrapper(task, **kwargs):
    args = dict(kwargs)
    output_dtype = args.pop("output_dtype")
    result = task.reference_bmm(**args)
    return result.to(output_dtype) if output_dtype is not None else result


def test_two_nonaligned_known_answers_are_independent_and_complete(task):
    cases = tail_cases(task)
    assert {c["name"] for c in cases} == TAIL_NAMES
    assert {c["kwargs"]["a"].shape[-1] for c in cases} == {35, 65}
    for case in cases:
        a, b = case["kwargs"]["a"], case["kwargs"]["b"]
        assert torch.count_nonzero(a[..., -1]) == 10
        assert torch.count_nonzero(b[..., -1]) == 10
        got = correct_wrapper(task, **case["kwargs"])
        torch.testing.assert_close(got, case["expected"], atol=0, rtol=0)
        assert torch.count_nonzero(case["expected"][:, :, 3:, :]) == 0
        assert torch.count_nonzero(case["expected"][:, :, :, 3:]) == 0
    assert TAIL_NAMES <= {r["control"] for r in task.semantic_controls.run_controls(
        SimpleNamespace(bmm_chunk_fwd=lambda **kwargs: correct_wrapper(task, **kwargs)), "cpu")}
    assert TAIL_NAMES <= {r["control"] for r in task.semantic_controls.reference_controls(
        SimpleNamespace(reference_bmm=task.reference_bmm))}


@pytest.mark.parametrize("k,truncate", [(35, 32), (65, 64)])
def test_dropping_final_k_lane_fails_actual_public_control(task, k, truncate):
    case = next(c for c in tail_cases(task) if c["kwargs"]["a"].shape[-1] == k)

    def bad_wrapper(**kwargs):
        args = dict(kwargs)
        args["a"] = args["a"][..., :truncate]
        args["b"] = args["b"][..., :truncate]
        return correct_wrapper(task, **args)

    original = task.semantic_controls.control_cases
    try:
        task.semantic_controls.control_cases = lambda device: iter([case])
        with pytest.raises(task.NumericalMismatch, match="Output numerical mismatch"):
            task.semantic_controls.run_controls(SimpleNamespace(bmm_chunk_fwd=bad_wrapper), "cpu")
    finally:
        task.semantic_controls.control_cases = original


def test_reference_dropping_tail_fails_independent_answer(task):
    case = next(c for c in tail_cases(task) if c["kwargs"]["a"].shape[-1] == 65)

    def truncated_reference(**kwargs):
        args = dict(kwargs)
        args["a"] = args["a"][..., :64]
        args["b"] = args["b"][..., :64]
        return task.reference_bmm(**args)

    original = task.semantic_controls.control_cases
    try:
        task.semantic_controls.control_cases = lambda device: iter([case])
        with pytest.raises(task.NumericalMismatch, match="Output numerical mismatch"):
            task.semantic_controls.reference_controls(SimpleNamespace(reference_bmm=truncated_reference))
    finally:
        task.semantic_controls.control_cases = original


def test_tail_control_rejects_input_mutation_and_bad_padding(task):
    case = next(c for c in tail_cases(task) if c["kwargs"]["a"].shape[-1] == 35)
    original = task.semantic_controls.control_cases
    try:
        task.semantic_controls.control_cases = lambda device: iter([case])

        def mutating_wrapper(**kwargs):
            got = correct_wrapper(task, **kwargs)
            kwargs["a"][..., -1].add_(1)
            return got

        with pytest.raises(task.ContractFailure, match="Read-only input changed"):
            task.semantic_controls.run_controls(SimpleNamespace(bmm_chunk_fwd=mutating_wrapper), "cpu")

        # The previous fault changed its input. Use a fresh case, and prove
        # the honest wrapper passes it before testing the padded output fault.
        case = next(c for c in original("cpu") if c["kwargs"]["a"].shape[-1] == 35)
        task.semantic_controls.control_cases = lambda device: iter([case])
        task.semantic_controls.run_controls(
            SimpleNamespace(bmm_chunk_fwd=lambda **kwargs: correct_wrapper(task, **kwargs)), "cpu")

        def bad_padding_wrapper(**kwargs):
            got = correct_wrapper(task, **kwargs)
            got[:, :, 31, 31] = 100
            return got

        with pytest.raises(task.NumericalMismatch, match="Output numerical mismatch"):
            task.semantic_controls.run_controls(SimpleNamespace(bmm_chunk_fwd=bad_padding_wrapper), "cpu")
    finally:
        task.semantic_controls.control_cases = original


def test_reference_control_rejects_even_unused_input_mutation(task):
    case = next(c for c in tail_cases(task) if c["kwargs"]["a"].shape[-1] == 35)
    original = task.semantic_controls.control_cases
    try:
        task.semantic_controls.control_cases = lambda device: iter([case])

        def mutating_reference(**kwargs):
            got = task.reference_bmm(**kwargs)
            kwargs["a"][..., 1].add_(1)
            return got

        with pytest.raises(task.ContractFailure, match="Read-only input changed"):
            task.semantic_controls.reference_controls(
                SimpleNamespace(reference_bmm=mutating_reference))
    finally:
        task.semantic_controls.control_cases = original
