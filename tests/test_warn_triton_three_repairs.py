"""CPU regressions for three task-owned numerical and measured-output repairs."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "tasks/triton2triton/vllm"
NAMES = ("triton_fla_chunk_fwd_o", "triton_kda_gla_fwd_o")
_measure_times = None


def _contract(name):
    path = VLLM / name / "scripts/contract_checks.py"
    spec = importlib.util.spec_from_file_location(f"contract_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_scored_seeds_reject_zero_and_retain_historical_draw(name):
    code = r'''
import torch
from scripts import task_runner
for seed in task_runner.SEEDS:
    args, kwargs = task_runner.gen_inputs(seed, 'cpu')
    old_args, old_kwargs = task_runner.gen_inputs(seed, 'cpu', historical=True)
    assert kwargs == old_kwargs
    assert len(args) == len(old_args)
    gate_index = 4 if 'fla' in task_runner.TASK_NAME else 2
    for index, (new, old) in enumerate(zip(args, old_args)):
        if isinstance(new, torch.Tensor):
            assert new.shape == old.shape
            if index == gate_index:
                assert torch.equal(new, old)
            else:
                ratio = 3 if 'fla' in task_runner.TASK_NAME else 5
                assert torch.allclose(new, old * ratio, atol=1e-6, rtol=1e-6)
    expected = task_runner.reference(*args, **kwargs)
    task_runner.require_scored_signal(expected)
    assert not torch.allclose(torch.zeros_like(expected), expected, atol=.05, rtol=.05)
    old_expected = task_runner.reference(*old_args, **old_kwargs)
    assert old_expected.shape == expected.shape
assert task_runner.WARMUP_ITERATIONS == 10
assert task_runner.BENCHMARK_ITERATIONS == 100
print('five scored zero controls and historical draws verified')
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=VLLM / name,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", NAMES)
def test_independent_known_answers_include_batch_head_width_and_partial_chunk(name):
    code = "import scripts.task_runner as h; print(h.run_reference_controls())"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=VLLM / name,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batched_three_heads_mixed_width_partial_chunk" in result.stdout


@pytest.mark.parametrize("name", NAMES)
def test_wrong_measured_sample_and_wrong_replay_are_separate_failures(name):
    contract = _contract(name)
    operand = torch.tensor([2.0])
    readonly = contract.InputSnapshot({"operand": operand})
    reference = lambda: operand * 3

    # A correct rerun cannot rescue one wrong output from the 100 reported samples.
    timed = SimpleNamespace(bound=True, outputs=reference().clone(),
                            rerun=lambda: reference().clone())
    contract.observe_measured_samples(timed, readonly, reference, atol=.05, rtol=.05)
    for _ in range(99):
        timed.after_sample(reference().clone())
    with pytest.raises(contract.NumericalMismatch):
        timed.after_sample(torch.zeros_like(reference()))
    assert timed.sample_checks == 99
    with pytest.raises(contract.ContractFailure, match="Reported samples"):
        contract.validate_timed(timed, readonly, reference, lambda: operand.mul_(8),
                                atol=.05, rtol=.05, expected_samples=100)

    # Conversely, all measured outputs can be right while the bound rerun fails.
    timed = SimpleNamespace(bound=True, outputs=reference().clone(),
                            rerun=lambda: torch.zeros_like(reference()))
    contract.observe_measured_samples(timed, readonly, reference, atol=.05, rtol=.05)
    for _ in range(100):
        timed.after_sample(reference().clone())
    with pytest.raises(contract.NumericalMismatch):
        contract.validate_timed(timed, readonly, reference, lambda: operand.mul_(8),
                                atol=.05, rtol=.05, expected_samples=100)
    assert torch.equal(operand, torch.tensor([2.0]))


@pytest.mark.parametrize("name", NAMES)
def test_measured_callback_rejects_input_mutation(name):
    contract = _contract(name)
    operand = torch.tensor([2.0])
    readonly = contract.InputSnapshot({"operand": operand})
    timed = SimpleNamespace()
    contract.observe_measured_samples(timed, readonly, lambda: torch.tensor([6.0]),
                                      atol=.05, rtol=.05)
    operand.add_(1)
    with pytest.raises(contract.ContractFailure, match="Read-only input changed"):
        timed.after_sample(torch.tensor([6.0]))
    assert timed.sample_checks == 0
