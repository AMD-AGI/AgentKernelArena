"""Exercise allocating task runners with a retained-output graph stand-in.

The canonical helper's tests cover GPU-event ordering. These CPU cases verify
that each task poisons the actual captured allocation and checks every result,
including a sample that skips its write and replay after inputs change.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "glm52": "glm_5_2_mxfp4_tp4_ep4_agentx_c8_m48_fused_moe",
    "glm53": "glm53_flash_tp8_c64_8k1k_fp8_mlp_down",
    "minimax": "minimax_m3_tp8_c64_8k1k_bf16_mlp_down",
}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fault", [None, "skip_sample", "stale_replay"])
def test_allocating_runner_checks_poisoned_samples_and_changed_graph(monkeypatch, task, fault):
    monkeypatch.setattr(sys, "path", list(sys.path))
    scripts = ROOT / "tasks/aiter_workloads" / TASKS[task] / "scripts"
    x = torch.ones((2, 2))
    weight = torch.eye(2)
    operands = (x, weight, torch.ones_like(x), torch.ones_like(weight))
    case = {"test_case_id": "tiny", "shape": [2, 2, 2], "dtype": "float32",
            "params": {"fresh_seed": 2, "benchmark_method": "cuda_graph_cold_cache"},
            "checks": ["correctness", "performance"]}
    data = {"cases": [case], "timing": {"warmup": 1, "repetition": 3},
            "input_policy": {"seed": 1, "replay_seed": 2},
            "input_generation": {"seeds": [1, 2]}, "tolerance": {}}
    payload = {"args": (x,), "kwargs": {}}
    checks = []

    def check(actual, expected, *_):
        assert torch.equal(actual, expected), "wrong or unwritten output"
        checks.append(True)
        return {}

    def snapshot(values):
        return tuple((v, v.clone()) for v in values)

    def unchanged(saved):
        assert all(torch.equal(v, old) for v, old in saved)

    def make_case(_, seed):
        values = operands if seed == 1 else tuple(torch.full_like(v, 2) for v in operands)
        return values, values[1]

    # Actual runner control flow and output checks; GPU math/runtime are replaced
    # with a small independent operation so this needs no GPU or AITER import.
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(verify_runtime=lambda: {}))
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(
        reference=lambda *args: args[0]["args"][0] * 2 if task == "glm52" else args[0] * 2,
        check_output=check, compare_output=lambda a, b: {"accepted": torch.equal(a, b)},
        reference_controls=lambda p: {}))
    monkeypatch.setitem(sys.modules, "cases", SimpleNamespace(
        make_case=make_case, snapshot=snapshot,
        check_unchanged=lambda values, saved: unchanged(saved),
        load_case=lambda c: payload, dynamic_inputs=lambda c, seed: torch.full_like(x, 2),
        replace_dynamic=lambda p, fresh: p["args"][0].copy_(fresh),
        snapshot_inputs=lambda p: snapshot(p["args"]), check_inputs_unchanged=unchanged))
    monkeypatch.setitem(sys.modules, "contract", SimpleNamespace(ROOT=scripts.parent, load_contract=lambda: data))
    monkeypatch.setitem(sys.modules, "inputs", SimpleNamespace(
        make_inputs=lambda c, seed: (x, weight) if seed == 1 else (torch.full_like(x, 2), torch.full_like(weight, 2)),
        snapshot_inputs=snapshot, check_inputs_unchanged=unchanged))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    events = []

    class TimedRun:
        def rerun(self):
            self.prepare()
            self.outputs.copy_(self.initial if fault == "stale_replay" else x * 2)
            return self.outputs

    def benchmark(fn, *, cache_flush, warmup, repetition, prepare_fn, timed_run):
        assert warmup == 1 and repetition == 3
        fn()  # warmup allocation must not be confused with the captured output
        output = fn()
        timed_run.outputs, timed_run.initial = output, output.clone()

        def prepare():
            prepare_fn()
            assert torch.isnan(output).all(), "captured output was not poisoned"
            events.append("poison")
            cache_flush()

        timed_run.prepare = prepare
        for index in range(repetition):
            prepare()
            if fault != "skip_sample" or index != 1:
                output.copy_(x * 2)
            events.append("replay")
            timed_run.after_sample(output)
            events.append("checked")
        return [0.2, 0.3, 0.4], {"benchmark_method": "cuda_graph_cold_cache",
                                "benchmark_effective_repeats": 1,
                                "benchmark_cache_flush_bytes": 1 << 30}

    monkeypatch.setitem(sys.modules, "_aka_benchmark", SimpleNamespace(
        TimedRun=TimedRun, ReadOnlyCacheFlush=lambda: lambda: events.append("evict"),
        benchmark_cuda_graph_cold_cache_samples=benchmark))
    spec = importlib.util.spec_from_file_location("cold_task_" + task, scripts / "evaluate.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    operation = lambda *args, **kwargs: args[0] * 2
    monkeypatch.setattr(runner, "candidate_callable" if task == "glm52" else "candidate", lambda: operation)
    if task == "glm52":
        monkeypatch.setattr(runner, "observed_native_modules", lambda: {})

    def run():
        if task == "glm53":
            return runner.execute("candidate", "performance", data)["cases"]
        return runner.run_action("candidate", "performance", data)[0]

    if fault:
        with pytest.raises(AssertionError, match="wrong or unwritten"):
            run()
    else:
        rows = run()
        assert rows[0]["metadata"]["samples_ms"] == [0.2, 0.3, 0.4]
        assert rows[0]["benchmark_method"] == "cuda_graph_cold_cache"
        assert events[:12] == ["poison", "evict", "replay", "checked"] * 3
