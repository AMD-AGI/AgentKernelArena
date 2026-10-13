"""Materialized runner/collector compatibility; simulated timing is not GPU evidence."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

from src.perf_helper_materialization import (
    canonical_aka_helper,
    materialize_perf_helpers_in_workspace,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "example_configs/task_validator_deepseek_drafts_mi355x.yaml"
TASKS = [ROOT / "tasks" / task for task in yaml.safe_load(CONFIG.read_text())["tasks"]]


@contextmanager
def materialized_modules(task, tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    shutil.copytree(task, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    helper_path = workspace / "scripts/_aka_benchmark.py"
    assert helper_path.read_text() == canonical_aka_helper(ROOT)
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(workspace))
        patch.syspath_prepend(str(workspace / "scripts"))
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts.") or name == "_aka_benchmark":
                patch.delitem(sys.modules, name)
        try:
            yield (importlib.import_module("scripts.task_runner"),
                   importlib.import_module("_aka_benchmark"))
        finally:
            for name in list(sys.modules):
                if name == "scripts" or name.startswith("scripts.") or name == "_aka_benchmark":
                    sys.modules.pop(name, None)


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
@pytest.mark.parametrize("stale_output", [False, True])
def test_measure_case_uses_materialized_collector_and_rechecks_refilled_inputs(
    task, stale_output, tmp_path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    # A small functional tensor contract exercises the actual task runner,
    # input refill, output poisoning and comparison without production GPU ops.
    definition = {
        "axes": {}, "op_type": "gemm",
        "inputs": {"a": {"shape": [4], "dtype": "float32"}},
        "outputs": {"out": {"shape": [4], "dtype": "float32"}},
    }
    row = {"workload": {"uuid": "collector-compatibility", "axes": {},
                        "inputs": {"a": {"type": "random"}}}}
    policy = json.loads((task / "scripts/workload.json").read_text())["policy"]
    with materialized_modules(task, tmp_path, monkeypatch) as (runner, helper):
        values = runner.make_inputs(definition, row, policy, device="cpu")
        output = torch.empty_like(values["a"])
        observed = []

        def reference(**inputs):
            return inputs["a"] * 2

        def launch(**inputs):
            output.copy_(reference(**inputs))
            return output

        def simulated_benchmark(call, *, timed_run, warmup, repetition, target_ms):
            # Do not replace the collector with a compatible-looking mock: the
            # previous task-local copy lacked the canonical third bind argument.
            assert type(timed_run) is helper.TimedRun
            assert (warmup, repetition, target_ms) == (
                policy["warmup"], policy["repetition"], policy["target_ms"],
            )
            initial_output = call()
            snapshot = initial_output.clone()

            def replay():
                if stale_output:
                    initial_output.copy_(snapshot)
                    return initial_output
                return call()

            timed_run._bind(replay, initial_output, lambda: (replay(), 0.125))
            assert timed_run.bound
            assert timed_run.rerun_ms() == 0.125
            observed.append(timed_run)
            return 0.125, {"benchmark_method": "simulated-device-timing"}

        monkeypatch.setattr(helper, "benchmark_cuda_graph_or_events", simulated_benchmark)
        if stale_output:
            with pytest.raises(AssertionError):
                runner.measure_case(launch, reference, values, definition, row, policy, device="cpu")
        else:
            result = runner.measure_case(
                launch, reference, values, definition, row, policy, device="cpu",
            )
            assert result["metadata"]["timed_output_checked"]
            assert result["metadata"]["refilled_input_replay_validated"]
        assert len(observed) == 1
