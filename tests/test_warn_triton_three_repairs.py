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
FLASH = ROOT / "tasks/instruction2triton/rocmbench/test_flashattention_fwd"
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


def test_flash_width_tail_control_detects_wrong_batch_head_routing():
    spec = importlib.util.spec_from_file_location("flash_reference", FLASH / "_arena_reference.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)

    def candidate(wrong_head):
        class Kernel:
            def __getitem__(self, grid):
                def launch(q, k, v, scale, L, m, output):
                    scores = (q @ k.transpose(-1, -2)) * scale
                    future = torch.ones(scores.shape[-2:], dtype=torch.bool).triu_(1)
                    scores.masked_fill_(future, float("-inf"))
                    value = torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v
                    output.copy_(value.roll(1, dims=1) if wrong_head else value)
                    stats = (q.float() @ k.float().transpose(-1, -2)) * scale
                    stats.masked_fill_(future, float("-inf"))
                    row_max = stats.max(-1).values
                    m.copy_(row_max.flatten(0, 1))
                    L.copy_(torch.exp(stats - row_max[..., None]).sum(-1).flatten(0, 1))
                return launch

        def attention(q, k, v, scale):
            output = torch.empty_like(q)
            L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]))
            m = torch.empty_like(L)
            module.flash_fwd_kernel[(1,)](q, k, v, scale, L, m, output)
            return output

        module = SimpleNamespace(flash_fwd_kernel=Kernel(), attention=attention)
        return module

    checks.check_width_tail_controls(candidate(False), "cpu")
    with pytest.raises(checks.NumericalMismatch):
        checks.check_width_tail_controls(candidate(True), "cpu")


def test_flash_kernel_uses_operand_specific_bases_and_tail_masks():
    source = (FLASH / "test_flashattention_fwd.py").read_text()
    assert "stride_qh_2d" not in source
    for operand, batch_stride, head_stride in (
        ("Q", "stride_qz", "stride_qh"),
        ("K", "stride_kz", "stride_kh"),
        ("V", "stride_vz", "stride_vh"),
        ("Out", "stride_oz", "stride_oh"),
    ):
        assert f"base={operand} + batch * {batch_stride} + head * {head_stride}" in source
    assert "mask=offs_m < N_CTX" in source
    assert "key_offsets[None, :] < N_CTX" in source
    assert "BLOCK = 64 if Lk == 128 else 128" in source


@pytest.mark.parametrize("fault", (
    "none", "middle_side_missing", "replay_side_missing",
    "middle_max_missing", "replay_max_missing",
    "middle_output_missing", "replay_output_missing", "input_mutation",
))
@pytest.mark.parametrize("method", ("cuda_event_fallback", "cuda_graph"))
def test_flash_adapter_checks_actual_side_buffers_and_restores_on_failure(monkeypatch, fault, method):
    ref_spec = importlib.util.spec_from_file_location("_arena_reference", FLASH / "_arena_reference.py")
    reference = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)
    eval_spec = importlib.util.spec_from_file_location("flash_eval_cpu", FLASH / "_arena_eval.py")
    adapter = importlib.util.module_from_spec(eval_spec)
    eval_spec.loader.exec_module(adapter)

    module = SimpleNamespace()
    phase = {"kind": "initial", "count": 0}

    class Kernel:
        def __getitem__(self, grid):
            def launch(q, k, v, scale, l, m, out):
                scores_half = (q @ k.transpose(-1, -2)) * scale
                mask = torch.ones(scores_half.shape[-2:], dtype=torch.bool).triu_(1)
                scores_half.masked_fill_(mask, float("-inf"))
                out.copy_(torch.softmax(scores_half.float(), dim=-1).to(q.dtype) @ v)
                scores = (q.float() @ k.float().transpose(-1, -2)) * scale
                scores.masked_fill_(mask, float("-inf"))
                max_rows = scores.max(dim=-1).values
                m.copy_(max_rows.reshape_as(m))
                l.copy_(torch.exp(scores - max_rows[..., None]).sum(dim=-1).reshape_as(l))
                phase["count"] += 1
                if ((fault == "middle_side_missing" and phase["kind"] == "measured" and phase["count"] == 50)
                        or (fault == "replay_side_missing" and phase["kind"] == "replay")):
                    l.zero_()
                if ((fault == "middle_max_missing" and phase["kind"] == "measured" and phase["count"] == 50)
                        or (fault == "replay_max_missing" and phase["kind"] == "replay")):
                    m.zero_()
                if ((fault == "middle_output_missing" and phase["kind"] == "measured" and phase["count"] == 50)
                        or (fault == "replay_output_missing" and phase["kind"] == "replay")):
                    out.zero_()
                if fault == "input_mutation" and phase["kind"] == "measured" and phase["count"] == 50:
                    q.add_(1)
            return launch

    module.flash_fwd_kernel = Kernel()
    original_kernel = module.flash_fwd_kernel

    def attention(q, k, v, scale):
        out = torch.empty_like(q)
        l = torch.empty((q.shape[0] * q.shape[1], q.shape[2]))
        m = torch.empty_like(l)
        module.flash_fwd_kernel[(1,)](q, k, v, scale, l, m, out)
        return out

    module.attention = attention

    class TimedRun:
        def __init__(self):
            self.bound = False
            self.outputs = None
            self.after_sample = None

        def rerun(self):
            phase["kind"] = "replay"
            return self._rerun()

    def samples(fn, *, warmup, repetition, max_graph_repeats, timed_run, **_kwargs):
        assert warmup == 10 and repetition == 100 and max_graph_repeats == 1
        phase.update(kind="measured", count=0)
        if method == "cuda_graph":
            # Capture Python launches once, then replay into the same O/L/m
            # buffers without touching the task-local launch observer.
            output = fn()
            side = module.flash_fwd_kernel.latest
            def graph_replay():
                original_kernel[(1,)](q, k, v, sm_scale, side[0], side[1], output)
                return output
            timed_run._rerun = graph_replay
        else:
            timed_run._rerun = fn
        for _ in range(repetition):
            output = graph_replay() if method == "cuda_graph" else fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.bound = True
        return [1.0] * repetition, {"benchmark_method": method}

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = TimedRun
    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    class Base:
        def __init__(self, *, op_callable, config):
            self.op_callable = op_callable
            self.config = config
            self.prepare_fn = None

        def run_benchmark(self, **_kwargs):
            values, info = _measure_times(self.op_callable, self.config)
            assert len(values) == 100
            return {"timing_ms": {"mean": 1.0}, **info}

    row = {"test_case_id": "scored"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    checked_type = adapter.benchmark_type(Base, plugin, module)
    generator = torch.Generator().manual_seed(2048)
    q = torch.randn((1, 2, 16, 16), generator=generator, dtype=torch.float16)
    k = torch.randn((1, 2, 16, 16), generator=generator, dtype=torch.float16)
    v = torch.randn((1, 2, 16, 16), generator=generator, dtype=torch.float16)
    sm_scale = .2
    original = (q.clone(), k.clone(), v.clone())
    checked = checked_type(op_callable=lambda: module.attention(q, k, v, sm_scale),
                           config=SimpleNamespace(warm_up=10, repetition=100))
    if fault == "none":
        checked.run_benchmark()
        assert row["metadata"]["measured_samples_checked"] == 100
        assert row["metadata"]["side_outputs_checked"] == "L,m"
    else:
        with pytest.raises((reference.NumericalMismatch, AssertionError)):
            checked.run_benchmark()
        assert "metadata" not in row
    assert module.flash_fwd_kernel is original_kernel
    assert all(torch.equal(tensor, saved) for tensor, saved in zip((q, k, v), original))
