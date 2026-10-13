"""CPU controls for decode's single-call measured graph contract."""

import importlib.util
from pathlib import Path

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/image_kernel/mi300x_sglang_hip_pa_decode"


def load_runner():
    spec = importlib.util.spec_from_file_location("decode_timing_runner", TASK / "scripts/task_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", (
    None, "cached_output", "cached_scratch", "wrong_middle", "mutated_middle_input",
    "cached_changed_replay", "missing_observer", "batched_graph",
))
def test_decode_checks_every_actual_sample_and_one_bound_replay(monkeypatch, fault):
    runner = load_runner()
    query = torch.tensor([2.0, -3.0])
    original_query = query.clone()
    cache = torch.tensor([5.0])
    original_cache = cache.clone()
    output = torch.empty_like(query)
    scratch = torch.empty(32, dtype=torch.uint8)
    case = {
        "params": {"num_seqs": 1, "num_query_heads": 1, "head_size": 2, "ctx_lens": 2},
        "query": query, "key_cache_new": cache, "output": output,
        "workspace_buffer": scratch,
    }
    monkeypatch.setattr(runner, "PERF_CASES", [("decode_case", {})])
    monkeypatch.setattr(runner, "_make_case", lambda **cfg: case)
    monkeypatch.setattr(runner, "_run_torch", lambda c: c["query"] * 3)
    monkeypatch.setattr(runner, "_write_performance_report", lambda rows: None)
    phase = {"sample": None, "replay": False}

    def op(c):
        if phase["sample"] is not None or phase["replay"]:
            assert torch.isnan(c["output"]).all()
            assert torch.all(c["workspace_buffer"] == 0xA5)
        if fault == "cached_output" and phase["sample"] is not None:
            return c["output"]
        if fault == "cached_scratch" and phase["sample"] is not None:
            c["output"].copy_(c["workspace_buffer"].view(torch.float32)[:2])
            return c["output"]
        if fault == "mutated_middle_input" and phase["sample"] == 50:
            c["key_cache_new"].add_(1)
        if fault == "wrong_middle" and phase["sample"] == 50:
            c["output"].copy_(c["query"] * 3 + 100)
        elif fault == "cached_changed_replay" and phase["replay"]:
            c["output"].copy_(original_query * 3)
        else:
            c["output"].copy_(c["query"] * 3)
        return c["output"]

    monkeypatch.setattr(runner, "_run_aiter", op)

    class TimedRun:
        def __init__(self):
            self.outputs = None
            self.after_sample = None
            self._replay = None

        @property
        def bound(self):
            return self._replay is not None

        def _bind(self, replay, outputs):
            self._replay = replay
            self.outputs = outputs

        def rerun(self):
            self.outputs = self._replay()
            return self.outputs

    monkeypatch.setattr(runner, "_TimedRun", TimedRun, raising=False)

    def benchmark(fn, *, timed_run, prepare_fn, max_graph_repeats,
                  warmup=10, repetition=100):
        assert warmup == 10 and repetition == 100 and max_graph_repeats == 1
        for _ in range(warmup):
            prepare_fn()
            fn()
        for index in range(repetition):
            phase["sample"] = index
            prepare_fn()
            actual = fn()
            if fault != "missing_observer":
                timed_run.after_sample(actual)
        phase["sample"] = None

        def replay():
            phase["replay"] = True
            prepare_fn()
            return fn()

        timed_run._bind(replay, actual)
        return 1.25, {
            "benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph",
            "benchmark_effective_repeats": 4 if fault == "batched_graph" else 1,
        }

    monkeypatch.setattr(runner, "_benchmark_cuda_graph_or_events", benchmark)
    if fault is None:
        rows = runner.run_performance()
        assert len(rows) == 1 and rows[0]["execution_time_ms"] == 1.25
        assert rows[0]["benchmark_measured_samples_checked"] == 100
        assert rows[0]["benchmark_replay_checked"] is True
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            runner.run_performance()
    torch.testing.assert_close(query, original_query, rtol=0, atol=0)
    torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)
