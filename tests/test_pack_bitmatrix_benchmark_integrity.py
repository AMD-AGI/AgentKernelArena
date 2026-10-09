"""The pack task must verify every invocation charged to a graph replay."""

import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_pack_bitmatrix"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous_directory = Path.cwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_directory)
    return module


class _TimedRun:
    def _bind(self, replay, output):
        self._replay = replay
        self.outputs = output

    def rerun(self):
        self.outputs = self._replay()
        return self.outputs


def _fake_capture(mode="fresh", *, metadata_change=None, wrong_binding=False):
    """Model four graph calls with real tensor writes and a bound replay."""
    controls = _load(TASK / "_upstream_controls.py", "_pack_integrity_controls")
    checks = _load(TASK / "_arena_checks.py", "_pack_integrity_checks")
    reference = controls.reference_pack_bitmatrix
    state = {"calls": 0, "computes": 0, "capture_outputs": []}
    harness = SimpleNamespace(reference_pack_bitmatrix=reference, _TimedRun=_TimedRun)

    def setup():
        topk_ids = torch.tensor(
            [[0, 32], [1, 33]] if mode == "partial_overlap" else [[0, 33], [1, 34]],
            dtype=torch.int16)
        num_experts = 64
        shared = torch.zeros((2, 2), dtype=torch.uint32)
        pool = torch.zeros((8, 2), dtype=torch.uint32)
        interleaved_pool = torch.zeros((2, 8), dtype=torch.uint32)
        overlapping_pool = torch.zeros((2, 5), dtype=torch.uint32)

        def candidate(ids, experts):
            index = state["calls"]
            state["calls"] += 1
            if mode == "alias":
                if index == 3:
                    state["computes"] += 1
                    shared.copy_(reference(ids, experts))
                return shared
            state["computes"] += 1
            if mode == "pool":
                output = pool[2 * index:2 * index + 2]
                output.copy_(reference(ids, experts))
                return output
            if mode == "interleaved_pool":
                output = interleaved_pool[:, index::4]
                output.copy_(reference(ids, experts))
                return output
            if mode == "partial_overlap":
                output = overlapping_pool[:, index:index + 2]
                output.copy_(reference(ids, experts))
                return output
            return reference(ids, experts)

        mod = SimpleNamespace(pack_topk_to_bitmatrix=candidate)

        def fn():
            return mod.pack_topk_to_bitmatrix(topk_ids, num_experts)

        def benchmark(measured, *, timed_run, **options):
            del options
            outputs = [measured() for _ in range(4)]
            state["capture_outputs"] = outputs

            def replay():
                if mode == "alias":
                    shared.copy_(reference(topk_ids, num_experts))
                else:
                    targets = outputs[-1:] if mode == "stale_replay" else outputs
                    for output in targets:
                        output.copy_(reference(topk_ids, num_experts))
                return outputs[-1]

            timed_run._bind(replay, outputs[-1].clone() if wrong_binding else outputs[-1])
            metadata = {"benchmark_method": "cuda_graph",
                        "benchmark_timed_run_kind": "captured_graph",
                        "benchmark_effective_repeats": 4,
                        "benchmark_warmup": 10,
                        "benchmark_samples": 100}
            if metadata_change:
                metadata.update(metadata_change)
            return 0.01, metadata

        return topk_ids, mod, candidate, fn, benchmark

    return checks, harness, state, setup()


@pytest.mark.parametrize("mode", ["fresh", "pool", "interleaved_pool"])
def test_all_four_captured_outputs_and_perturbed_replay_pass(mode):
    checks, harness, state, (ids, mod, candidate, fn, benchmark) = _fake_capture(mode)
    pristine = ids.clone()
    _, metadata = checks.checked_benchmark(harness, benchmark, fn)
    assert metadata["timed_invocation_outputs_checked"] == 4
    assert metadata["timed_invocation_outputs_disjoint"] is True
    assert metadata["timed_output_checked"] is True
    assert metadata["perturbed_input_replay_checked"] is True
    assert state["calls"] == state["computes"] == 4
    if mode in ("pool", "interleaved_pool"):
        outputs = state["capture_outputs"]
        assert len({output.untyped_storage().data_ptr() for output in outputs}) == 1
        if mode == "interleaved_pool":
            spans = [checks.output_span(output) for output in outputs]
            assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]
    assert torch.equal(ids, pristine)
    assert mod.pack_topk_to_bitmatrix is candidate


@pytest.mark.parametrize("mode,reason", [
    ("alias", "overlap"),
    ("partial_overlap", "overlap"),
    ("stale_replay", "independent reference"),
])
def test_selected_compute_alias_and_distinct_stale_outputs_fail_and_restore(mode, reason):
    checks, harness, state, (ids, mod, candidate, fn, benchmark) = _fake_capture(mode)
    pristine = ids.clone()
    with pytest.raises(AssertionError, match=reason):
        checks.checked_benchmark(harness, benchmark, fn)
    if mode == "alias":
        assert state["calls"] == 4 and state["computes"] == 1
        assert len({id(output) for output in state["capture_outputs"]}) == 1
    assert torch.equal(ids, pristine)
    assert mod.pack_topk_to_bitmatrix is candidate


@pytest.mark.parametrize("metadata_change,wrong_binding,reason", [
    ({"benchmark_effective_repeats": None}, False, "repeat count"),
    ({"benchmark_effective_repeats": 5}, False, "not observable"),
    ({"benchmark_timed_run_kind": "eager_callable"}, False, "not observable"),
    (None, True, "final captured invocation"),
])
def test_missing_or_mismatched_graph_binding_fails_closed_and_restores(
    metadata_change, wrong_binding, reason,
):
    checks, harness, _, (ids, mod, candidate, fn, benchmark) = _fake_capture(
        metadata_change=metadata_change, wrong_binding=wrong_binding)
    pristine = ids.clone()
    with pytest.raises(AssertionError, match=reason):
        checks.checked_benchmark(harness, benchmark, fn)
    assert torch.equal(ids, pristine)
    assert mod.pack_topk_to_bitmatrix is candidate


def test_event_fallback_keeps_one_observed_invocation():
    checks, harness, _, (ids, mod, candidate, fn, _) = _fake_capture()
    pristine = ids.clone()

    def benchmark(measured, *, timed_run, **options):
        del options
        output = measured()
        timed_run._bind(measured, output)
        return 0.01, {"benchmark_method": "cuda_event_fallback",
                      "benchmark_timed_run_kind": "eager_callable",
                      "benchmark_effective_repeats": 1,
                      "benchmark_warmup": 10,
                      "benchmark_samples": 100}

    _, metadata = checks.checked_benchmark(harness, benchmark, fn)
    assert metadata["timed_invocation_outputs_checked"] == 1
    assert metadata["perturbed_input_replay_checked"] is True
    assert torch.equal(ids, pristine)
    assert mod.pack_topk_to_bitmatrix is candidate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton and a CUDA/ROCm GPU are required")
def test_actual_gpu_native_performance_and_output_ownership(tmp_path, request):
    pytest.importorskip("triton")
    previous_directory = Path.cwd()
    previous_path = sys.path[:]
    previous_helper = sys.modules.get("_aka_benchmark")

    def restore_import_state():
        os.chdir(previous_directory)
        sys.path[:] = previous_path
        if previous_helper is None:
            sys.modules.pop("_aka_benchmark", None)
        else:
            sys.modules["_aka_benchmark"] = previous_helper

    request.addfinalizer(restore_import_state)
    task = tmp_path / "pack_task"
    shutil.copytree(TASK, task)
    materialize_perf_helpers_in_workspace(task)
    adapter = _load(task / "_arena_eval.py", "_pack_integrity_gpu_adapter")
    result = adapter.evaluate("candidate", "performance")
    assert result["status"] == "PASS", result
    assert [row["test_case_id"] for row in result["cases"]] == [f"perf{i}" for i in range(1, 6)]
    for row in result["cases"]:
        assert row["status"] == "PASS", row
        measurement = row["metadata"]["harness_measurement"]
        assert measurement["benchmark_method"] == "cuda_graph"
        assert measurement["benchmark_warmup"] == 10
        assert measurement["benchmark_samples"] == 100
        assert measurement["timed_invocation_outputs_checked"] == measurement["benchmark_effective_repeats"]
        assert measurement["timed_invocation_outputs_disjoint"] is True

    checks = _load(task / "_arena_checks.py", "_pack_integrity_gpu_checks")
    harness = adapter.load_harness()
    source = _load(task / "source/triton_pack_bitmatrix.py", "_pack_integrity_gpu_source")
    pool_capacity = 32
    pool = torch.empty((32, pool_capacity * 2), dtype=torch.uint32, device="cuda")
    pool_calls = [0]

    def interleaved(ids, experts):
        index = pool_calls[0]
        pool_calls[0] += 1
        assert index < pool_capacity
        output = pool[:, index::pool_capacity]
        output.copy_(source.pack_topk_to_bitmatrix(ids, experts))
        return output

    def setup_interleaved():
        topk_ids = torch.randint(0, 64, (32, 2), device="cuda", dtype=torch.int16)
        num_experts = 64
        mod = SimpleNamespace(pack_topk_to_bitmatrix=interleaved)

        def fn():
            mod.pack_topk_to_bitmatrix(topk_ids, num_experts)

        return fn, topk_ids

    interleaved_fn, interleaved_ids = setup_interleaved()
    pristine_ids = interleaved_ids.clone()
    _, interleaved_metadata = checks.checked_benchmark(
        harness, harness._benchmark_cuda_graph_or_events, interleaved_fn,
        warmup=1, repetition=3, estimate_reps=1, target_ms=0.2, max_graph_repeats=8,
    )
    assert interleaved_metadata["benchmark_method"] == "cuda_graph"
    assert interleaved_metadata["benchmark_effective_repeats"] >= 2
    assert (interleaved_metadata["timed_invocation_outputs_checked"]
            == interleaved_metadata["benchmark_effective_repeats"])
    assert interleaved_metadata["timed_invocation_outputs_disjoint"] is True
    assert interleaved_metadata["perturbed_input_replay_checked"] is True
    assert torch.equal(interleaved_ids, pristine_ids)

    shared = torch.zeros((32, 2), dtype=torch.uint32, device="cuda")

    def aliased(ids, experts):
        shared.copy_(source.pack_topk_to_bitmatrix(ids, experts))
        return shared

    def setup():
        topk_ids = torch.randint(0, 64, (32, 2), device="cuda", dtype=torch.int16)
        num_experts = 64
        mod = SimpleNamespace(pack_topk_to_bitmatrix=aliased)

        def fn():
            mod.pack_topk_to_bitmatrix(topk_ids, num_experts)

        return fn

    with pytest.raises(AssertionError, match="overlap"):
        checks.checked_benchmark(
            harness, harness._benchmark_cuda_graph_or_events, setup(),
            warmup=1, repetition=3, target_ms=0.2, max_graph_repeats=8,
        )
