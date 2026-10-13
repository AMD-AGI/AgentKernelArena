"""CPU controls for the task-local known answers and measured replay boundary."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("triton_fla_chunk_fwd_o", "triton_kda_gla_fwd_o")


@contextmanager
def task_runner(name):
    task_dir = ROOT / "tasks" / "triton2triton" / "vllm" / name
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    names = ("scripts", "scripts.contract_checks", "scripts.semantic_controls",
             "scripts.task_runner", "_aka_benchmark")
    saved = {name: sys.modules.get(name) for name in names}
    for module_name in names:
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(task_dir))
    spec = importlib.util.spec_from_file_location(
        f"{name}_task_runner_test", task_dir / "scripts" / "task_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)
        for module_name in names:
            sys.modules.pop(module_name, None)
            if saved[module_name] is not None:
                sys.modules[module_name] = saved[module_name]
        sys.path[:] = old_path
        os.chdir(old_cwd)


class FakeTimedRun:
    def __init__(self):
        self.bound = False
        self.sample_checks = 0
        self.outputs = None
        self.after_sample = None

    def bind(self, replay, output):
        self.bound = True
        self._replay = replay
        self.outputs = output

    def rerun(self):
        return self._replay()


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("mode", ("graph", "event"))
@pytest.mark.parametrize("fault, fault_sample", ((None, None),
    ("skip", 0), ("skip", 41), ("skip", 99),
    ("wrong", 0), ("wrong", 41), ("wrong", 99),
    ("multicall_metadata", None)))
def test_one_wrapper_call_and_every_sample_output(name, mode, fault,
                                                   fault_sample, monkeypatch):
    with task_runner(name) as runner:
        original_gen = runner.gen_inputs
        monkeypatch.setattr(runner, "gen_inputs", lambda seed, device:
                            original_gen(seed, "cpu"))
        original_to = torch.Tensor.to
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs:
                            self if args == ("cuda",) and not kwargs else
                            original_to(self, *args, **kwargs))
        sys.modules["_aka_benchmark"] = types.SimpleNamespace(TimedRun=FakeTimedRun)

        class Module:
            def __init__(self):
                self.args = None
                self.kwargs = None
                self.output = None

            def invoke(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                self.output = runner.reference(*args, **kwargs).clone()
                return self.output

            chunk_fwd_o = invoke
            kda_gla_fwd_o = invoke

            def replay(self):
                self.output.copy_(runner.reference(*self.args, **self.kwargs))
                return self.output

        mod = Module()
        monkeypatch.setattr(runner, "load_module", lambda: mod)

        def graph_benchmark(fn, *, warmup, repetition, max_graph_repeats,
                            prepare_fn, timed_run):
            assert (warmup, repetition, max_graph_repeats) == (10, 100, 1)
            assert prepare_fn is not None
            for _ in range(warmup):
                prepare_fn()
                fn()
            if mode == "graph":
                prepare_fn()
                captured = fn()
                # Capture storage may be uninitialized. Preparation must not
                # demand finite data before the first correct replay.
                captured.fill_(float("nan"))
                for sample in range(repetition):
                    prepare_fn()
                    if sample == fault_sample and fault == "skip":
                        pass
                    elif sample == fault_sample and fault == "wrong":
                        captured.zero_()
                    else:
                        mod.replay()
                    timed_run.after_sample(captured)
                timed_run.bind(mod.replay, captured)
            else:
                current = None
                for sample in range(repetition):
                    prepare_fn()
                    if sample == fault_sample and fault == "skip":
                        # A missing eager call returns the prior poisoned
                        # allocation, including at the first reported sample.
                        current = mod.output
                    else:
                        current = fn()
                    if sample == fault_sample and fault == "wrong":
                        current.zero_()
                    timed_run.after_sample(current)
                timed_run.bind(lambda: fn(), current)
            return 1.0, {"benchmark_method":
                         "cuda_graph" if mode == "graph" else "cuda_event_fallback",
                         "benchmark_effective_repeats":
                         2 if fault == "multicall_metadata" else 1,
                         "benchmark_samples": repetition,
                         "benchmark_warmup": warmup}

        monkeypatch.setattr(runner, "_benchmark_cuda_graph_or_events",
                            graph_benchmark)
        rows = runner.run_performance()
        assert len(rows) == len(runner.SEEDS) == 5
        if fault is None:
            assert all("error" not in row for row in rows), rows
            assert all(row["measured_samples_checked"] == 100 for row in rows)
            assert all(row["benchmark_effective_repeats"] == 1 for row in rows)
            assert all(row["replay_correctness"] == "PASS" for row in rows)
        else:
            assert all(row["execution_time_ms"] == -1.0 for row in rows)
            assert all("error" in row for row in rows)
            expected_error = ("Measured replay" if fault == "multicall_metadata"
                              else "nonfinite" if fault == "skip"
                              else "numerical mismatch")
            assert all(expected_error.lower() in row["error"].lower()
                       for row in rows), rows


@pytest.mark.parametrize("gated", (False, True))
@pytest.mark.parametrize("cutoff", (32, 64))
def test_wide_key_independent_answers_reject_truncation(gated, cutoff):
    with task_runner("triton_fla_chunk_fwd_o") as runner:
        cases = list(runner.semantic_controls.control_cases("cpu"))
        wide = next(case for case in cases if case["name"] ==
                    ("gated_multi_tile_key_width" if gated else
                     "ungated_multi_tile_key_width"))
        assert wide["kwargs"]["q"].shape[-1] == 80
        runner.semantic_controls._check(
            runner.reference(**wide["kwargs"]), wide["expected"],
            wide["atol"], wide["rtol"])

        class TruncatedKey:
            def chunk_fwd_o(self, *, q, k, v, h, g, scale, chunk_size):
                q = q.clone(); q[..., cutoff:] = 0
                k = k.clone(); k[..., cutoff:] = 0
                h = h.clone(); h[..., cutoff:] = 0
                return runner.reference(q, k, v, h, g, scale, chunk_size)

        class CompleteKey:
            def chunk_fwd_o(self, *, q, k, v, h, g, scale, chunk_size):
                return runner.reference(q, k, v, h, g, scale, chunk_size)

        assert len(runner.semantic_controls.run_controls(CompleteKey(),
                                                        device="cpu")) == 7
        with pytest.raises(runner.semantic_controls.ContractFailure):
            runner.semantic_controls._check(
                TruncatedKey().chunk_fwd_o(**wide["kwargs"]),
                wide["expected"], wide["atol"], wide["rtol"])

        # The preexisting five controls cannot detect this shortcut.
        for case in cases[:5]:
            got = TruncatedKey().chunk_fwd_o(**case["kwargs"])
            runner.semantic_controls._check(
                got, case["expected"], case["atol"], case["rtol"])
        with pytest.raises(runner.semantic_controls.ContractFailure):
            runner.semantic_controls.run_controls(TruncatedKey(), device="cpu")
