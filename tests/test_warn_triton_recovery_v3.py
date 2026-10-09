"""CPU controls for output-state replay and declared partial-tile branches."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"
REVERSE_TASKS = (
    "instruction2triton/rocmbench/test_reverse_range",
    "triton2triton/rocmbench/easy/test_reverse_range",
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", REVERSE_TASKS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_reverse_interior_transition_rejects_correct_first_element(task, dtype):
    reference = load(TASKS / task / "_arena_reference.py", "reverse_transition_reference")
    source = torch.arange(516, dtype=torch.float32).remainder_(23).to(dtype)
    output = torch.flip(source[1:513], [0]).clone()
    original_source, original_output = source.clone(), output.clone()
    context = {"data_perf": source, "res_perf_buffer": output}

    def early_return():
        if output[0] != source[512]:
            output.copy_(torch.flip(source[1:513], [0]))
        return output

    with pytest.raises(reference.NumericalMismatch):
        reference.check_interior_transitions(context, early_return)
    assert torch.equal(source, original_source)
    assert torch.equal(output, original_output)


@pytest.mark.parametrize("task", REVERSE_TASKS)
def test_reverse_transition_restores_after_candidate_exception(task):
    reference = load(TASKS / task / "_arena_reference.py", "reverse_restore_reference")
    source = torch.arange(516, dtype=torch.float32)
    output = torch.flip(source[1:513], [0]).clone()
    original_source, original_output = source.clone(), output.clone()

    def raise_after_write():
        output.zero_()
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        reference.check_interior_transitions(
            {"data_perf": source, "res_perf_buffer": output}, raise_after_write)
    assert torch.equal(source, original_source)
    assert torch.equal(output, original_output)


@pytest.mark.parametrize("wrong_branch", (None, "partial_m", "partial_n", "partial_k"))
def test_block_pointer_controls_check_each_partial_dimension(wrong_branch):
    task = TASKS / "triton2triton/rocmbench/hard/test_block_pointer_matmul"
    reference = load(task / "_arena_reference.py", "block_partial_reference")
    seen = []

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def launch(**kwargs):
                a, b, out = kwargs["a_ptr"], kwargs["b_ptr"], kwargs["c_ptr"]
                bm, bn, bk = (kwargs[x] for x in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
                branch = ("partial_m" if bm < kwargs["M"] else
                          "partial_n" if bn < kwargs["N"] else "partial_k")
                seen.append(branch)
                if branch != wrong_branch:
                    out[:bm, :bn] = a[:bm, :bk] @ b[:bk, :bn]

            return launch

    context = {"a": torch.empty((1, 1))}
    module = SimpleNamespace(matmul_no_scf_with_advance_kernel=FakeKernel())
    if wrong_branch is None:
        reference.check_partial_tile_controls(context, module)
        assert seen == ["partial_m", "partial_n", "partial_k"]
    else:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_partial_tile_controls(context, module)
        assert seen[-1] == wrong_branch


@pytest.mark.parametrize("task", (*REVERSE_TASKS, "instruction2triton/rocmbench/gemm"))
@pytest.mark.parametrize("skip_middle", (False, True))
@pytest.mark.parametrize("reported_repeats", (1, 2))
def test_graph_samples_poison_output_and_require_one_call(
    monkeypatch, task, skip_middle, reported_repeats
):
    reference = load(TASKS / task / "_arena_reference.py", "_arena_reference")
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)
    adapter = load(TASKS / task / "_arena_eval.py", "output_state_adapter")

    class TimedRun:
        def __init__(self):
            self.bound, self.outputs, self.after_sample, self.replay = False, None, None, None

        def rerun(self):
            self.outputs = self.replay()
            return self.outputs

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = TimedRun
    skipping = [False]
    preparation_count = [0]

    def samples(fn, *, warmup, repetition, max_graph_repeats, prepare_fn, timed_run, **kwargs):
        assert warmup == 10 and repetition == 3 and max_graph_repeats == 1
        assert prepare_fn is not None
        for index in range(repetition):
            prepare_fn()
            preparation_count[0] += 1
            skipping[0] = skip_middle and index == 1
            result = fn()
            timed_run.after_sample(result)
        skipping[0] = False
        timed_run.outputs, timed_run.bound = result, True
        timed_run.replay = lambda: (prepare_fn(), fn())[1]
        return [0.5] * repetition, {"benchmark_method": "cuda_graph",
                                   "benchmark_effective_repeats": reported_repeats}

    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    def _measure_times(*args, **kwargs):
        raise AssertionError("task did not bind canonical timer")

    class Base:
        def __init__(self, *, op_callable, config):
            self.op_callable, self.config, self.prepare_fn = op_callable, config, None

        def run_benchmark(self, *, baseline_callable=None):
            assert baseline_callable is None
            values, metadata = globals()["_measure_times"](
                self.op_callable, self.config, prepare_fn=self.prepare_fn)
            return {"timing_ms": {"mean": sum(values) / len(values)}, **metadata}

    monkeypatch.setitem(Base.run_benchmark.__globals__, "_measure_times", _measure_times)

    if task.endswith("gemm"):
        a = torch.tensor([[0.25, 0.5], [0.75, 0.125]])
        b = torch.tensor([[0.5, 0.25], [0.75, 0.5]])
        c = torch.empty((2, 2))
        current_scale_a8_b8 = False
        a_fp32_ref, b_fp32_ref = a, b
        a_scale, b_scale = None, None

        def operation():
            if not skipping[0]:
                c.copy_(a @ b)
            return c

    else:
        data_perf = torch.arange(1.0, 517.0)
        res_perf_buffer = torch.empty(512)

        def operation():
            if not skipping[0]:
                res_perf_buffer.copy_(torch.flip(data_perf[1:513], [0]))
            return res_perf_buffer

    row = {"test_case_id": "output-state"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    checked = adapter.benchmark_type(Base, plugin, SimpleNamespace())(
        op_callable=operation, config=SimpleNamespace(warm_up=10, repetition=3))
    if skip_middle:
        with pytest.raises(ValueError, match="nonfinite"):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
    elif reported_repeats != 1:
        with pytest.raises(RuntimeError, match="one prepared invocation"):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
    else:
        checked.run_benchmark()
        assert row["metadata"]["measured_samples_checked"] == 3
        assert preparation_count == [3]
