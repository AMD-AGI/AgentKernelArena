"""Small real-GPU checks for protected graph benchmark semantics.

The ordinary CPU test job collects this module but skips it.  The dedicated
self-hosted ROCm workflow runs it to exercise actual graph capture and replay.
"""

from __future__ import annotations

import pytest
import torch

from src.tools.perf.aka_benchmark import benchmark_cuda_graph_or_events
from src.tools.perf.vllm_cuda_graph_block import _TimedRun


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/ROCm GPU"
)


def test_stateful_graph_replay_restores_input_before_each_sample() -> None:
    torch.manual_seed(7)
    pristine = torch.randn(4096, device="cuda")
    state = pristine.clone()

    def prepare() -> None:
        state.copy_(pristine)

    def mutate() -> torch.Tensor:
        return state.mul_(1.5)

    elapsed_ms, metadata = benchmark_cuda_graph_or_events(
        mutate,
        warmup=2,
        repetition=5,
        target_ms=0.05,
        estimate_reps=2,
        max_graph_repeats=8,
        prepare_fn=prepare,
    )

    assert elapsed_ms > 0.0
    assert metadata["benchmark_method"] == "cuda_graph"
    assert metadata["benchmark_effective_repeats"] == 1
    assert torch.equal(state, pristine * 1.5)


def test_timed_run_distinguishes_graph_replay_from_explicit_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(4096, device="cuda")
    output = torch.empty_like(x)

    def write_output() -> torch.Tensor:
        torch.mul(x, 2.0, out=output)
        return output

    timed_run = _TimedRun()
    _, metadata = benchmark_cuda_graph_or_events(
        write_output,
        warmup=2,
        repetition=5,
        target_ms=0.05,
        estimate_reps=2,
        max_graph_repeats=8,
        timed_run=timed_run,
    )

    assert metadata["benchmark_method"] == "cuda_graph"
    assert metadata["benchmark_timed_run_kind"] == "captured_graph"
    assert timed_run.bound
    assert timed_run.outputs is output

    x.fill_(3.0)
    output.fill_(float("nan"))
    assert timed_run.rerun() is output
    assert torch.equal(output, torch.full_like(output, 6.0))

    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    event_timed = _TimedRun()
    _, event_metadata = benchmark_cuda_graph_or_events(
        write_output,
        warmup=0,
        repetition=2,
        timed_run=event_timed,
    )
    assert event_metadata["benchmark_method"] == "cuda_event_fallback"
    assert event_metadata["benchmark_fallback_reason"] == "forced_event_baseline"
    assert event_metadata["benchmark_timed_run_kind"] == "eager_callable"
    assert event_timed.outputs is output
    assert torch.equal(output, torch.full_like(output, 6.0))
    x.fill_(5.)
    output.fill_(float("nan"))
    assert event_timed.rerun() is output
    assert torch.equal(output, torch.full_like(output, 10.0))


def test_event_collector_keeps_measured_allocation_and_prepares_eager_rerun() -> None:
    from src.tools.perf.aka_benchmark import TimedRun
    pristine = torch.arange(4096, device="cuda", dtype=torch.float32)
    state = torch.empty_like(pristine)
    outputs = []

    def prepare():
        state.copy_(pristine)

    def operation():
        state.mul_(2.)
        result = state.clone()
        outputs.append(result)
        return result

    timed = TimedRun()
    elapsed, metadata = benchmark_cuda_graph_or_events(
        operation, warmup=2, repetition=5, prepare_fn=prepare,
        use_cuda_graph=False, timed_run=timed,
    )
    assert elapsed > 0 and metadata["benchmark_timed_run_kind"] == "eager_callable"
    assert len(outputs) == 7
    measured = timed.outputs
    assert measured is outputs[-1]
    assert torch.equal(measured, pristine * 2.)
    pristine.add_(3.)
    state.fill_(float("nan"))
    result = timed.rerun()
    assert len(outputs) == 8
    assert result is outputs[-1] and result is not measured
    assert torch.equal(result, pristine * 2.)
    assert torch.equal(measured, (pristine - 3.) * 2.)
