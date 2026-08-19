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


def test_timed_run_replays_exact_graph_and_forced_event_fails_closed(
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
    assert timed_run.bound
    assert timed_run.outputs is output

    x.fill_(3.0)
    output.fill_(float("nan"))
    assert timed_run.rerun() is output
    assert torch.equal(output, torch.full_like(output, 6.0))

    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="timed_run requires"):
        benchmark_cuda_graph_or_events(
            write_output,
            warmup=0,
            repetition=2,
            timed_run=_TimedRun(),
        )

    _, event_metadata = benchmark_cuda_graph_or_events(
        write_output,
        warmup=0,
        repetition=2,
    )
    assert event_metadata["benchmark_method"] == "cuda_event_fallback"
    assert event_metadata["benchmark_fallback_reason"] == "forced_event_baseline"
