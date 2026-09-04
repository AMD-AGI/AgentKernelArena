# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Invariants for the SIKL rewrite tasks.

Each task reimplements one workload-schema operator in FlyDSL and is scored
against that operator's production implementation over every case the schema
declares. The task carries the schema's unfilled ``kernel-forges`` solution slot
so post-processing can fill it from the scored result.

The checks here pin the couplings that fail silently rather than loudly: a
factory symbol the harness looks up but the pipeline never asks for, a workload
whose tolerance was never measured, or a driver that times the candidate
differently from the way the task is scored.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SIKL_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "SIKL-task"
TASKS = sorted(path for path in SIKL_ROOT.glob("*") if (path / "config.yaml").is_file())


def _config(task: Path) -> dict:
    with (task / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _workload(task: Path) -> dict:
    return json.loads((task / "workload.json").read_text())


def test_the_suite_is_not_empty():
    assert TASKS, f"no SIKL tasks found under {SIKL_ROOT}"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_is_driven_by_the_rewrite_pipeline(task):
    config = _config(task)
    rewrite = config["rewrite"]
    assert config["task_type"] == "rewrite_by_flydsl"
    assert config["source_file_path"] == ["kernel.py"]
    assert rewrite["port_target"] == "kernel.py"
    assert Path(rewrite["port_source"]).is_absolute()
    assert rewrite["source_owner"]


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_builder_symbol_agrees_with_kernelforge(task):
    # KernelForge derives the required factory symbol from the task's logical
    # operator and offers no override, while the harness looks up whatever
    # workload.json names. If the two ever disagree the harness finds no factory
    # and reports the baseline as the port's score -- a silent pass.
    protocol = pytest.importorskip("kernelforge.rewrite_by_flydsl.protocol")
    config = _config(task)
    declared = _workload(task)["builder_symbol"]

    assert protocol.builder_symbol(config["rewrite"]["logical_operator"]) == declared
    assert config["target_kernel_functions"] == [declared]


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_stub_defines_no_operator_specific_factory(task):
    # The factory name is per operator, so a stub that hardcoded one could only
    # ever match a single task. Its absence is also how the harness recognizes
    # that no port has landed yet and scores the baseline instead.
    stub = (task / "kernel.py").read_text()
    assert "def build_" not in stub
    assert _workload(task)["builder_symbol"] not in stub


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_workload_declares_cases_and_a_gate_policy(task):
    workload = _workload(task)
    assert workload["cases"], "a task must score at least one workload case"
    for case in workload["cases"]:
        assert case["uuid"], f"{case['case_id']} carries no schema case uuid"
    assert workload["gate_multiplier"] > 1, (
        "a gate at or below the baseline's own error is unpassable"
    )
    assert workload["gate_policy"].strip()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_workload_records_no_measurement(task):
    # Tolerances and baseline timings are derived at run time on the machine
    # that is about to score the candidate. Recording them here would pin
    # numbers that go stale against a different GPU or framework build, and
    # Arena measures its own baseline every run regardless.
    workload = _workload(task)
    for key in ("max_relerr", "tolerance_reason"):
        assert key not in workload, f"{key} is a recorded measurement"
    for case in workload["cases"]:
        assert set(case) == {"case_id", "uuid", "m"}, (
            f"{case['case_id']} carries more than the schema's case identity"
        )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_gate_is_derived_from_the_baseline(task):
    measure = (task / "scripts" / "task_measure.py").read_text()
    assert "task_inputs.derive_gate" in measure
    assert "task_baseline.run" in measure


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_driver_and_harness_share_one_measurement_implementation(task):
    # The pipeline's own keep/revert decision and the task's score have to come
    # from one measurement regime. Sharing constants is not enough -- two loops
    # that agree today drift tomorrow -- so the loops themselves are shared and
    # neither caller may time anything on its own.
    driver = (task / "scripts" / "forge_driver.py").read_text()
    harness = (task / "test_kernel_harness.py").read_text()
    measure = (task / "scripts" / "task_measure.py").read_text()

    assert "benchmark_cuda_graph_or_events" in measure
    for source in (driver, harness):
        assert "import task_measure" in source
        assert "task_measure.time_cases" in source
        assert "benchmark_cuda_graph_or_events" not in source
    for constant in ("BENCH_WARMUP", "BENCH_REPETITION", "BENCH_TARGET_MS"):
        assert f"task_inputs.{constant}" in measure


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_driver_supports_the_dual_path_contract(task):
    driver = (task / "scripts" / "forge_driver.py").read_text()
    for flag in ("--ref-bench-mode", "--bench-mode", "--profile-run"):
        assert flag in driver
    # The contract reads the first match of each aggregate, so a per-case detail
    # line spelled the same way would be read as the verdict for the whole run.
    assert driver.count('"SNR: ') + driver.count("'SNR: ") <= 1
    assert driver.count('"allclose: ') + driver.count("'allclose: ") <= 1


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_perf_helper_is_materialized_not_committed(task):
    # setup_workspace() injects the canonical helper into every workspace; a
    # committed copy would silently pin an older timing methodology.
    assert not (task / "_aka_benchmark.py").exists()
    assert not (task / "scripts" / "_aka_benchmark.py").exists()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_ships_the_unfilled_solution_slot(task):
    # Post-processing fills this slot from the scored result, so it has to
    # travel with the task and still identify the workload-schema slot it came
    # from.
    solution = json.loads((task / "solution.json").read_text())
    assert solution["author"] == "kernel-forges"
    assert solution["definition"] == _workload(task)["definition"]
    assert solution["name"]
    assert solution["spec"]["entry_point"] == ""
    assert solution["sources"] == [{"path": "", "content": ""}]


def _task_inputs(task: Path):
    """Load one task's task_inputs under a unique module name.

    Every task ships its own copy under the same file name, so a plain import
    would bind whichever task ran first for the whole session.
    """
    path = task / "scripts" / "task_inputs.py"
    spec = importlib.util.spec_from_file_location(f"task_inputs_{task.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_a_candidate_reusing_the_framework_is_rejected(task):
    # aiter's tuned dispatch resolves to aiter's own FlyDSL kernels at the
    # small-M cases, so a port that imports them measures the baseline against
    # itself and reports the removal of per-call host dispatch as a speedup.
    task_inputs = _task_inputs(task)
    source = (
        "import torch\n"
        "import flydsl.compiler as flyc\n"
        "from aiter.ops.flydsl.kernels import splitk_hgemm\n"
    )
    with pytest.raises(RuntimeError, match="imports the framework under test"):
        task_inputs.assert_candidate_is_independent(source)


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_a_candidate_computing_with_torch_is_rejected(task):
    # torch's matmul IS the baseline at the larger M cases, so `a @ b.T` would
    # tie it exactly while implementing no kernel at all.
    task_inputs = _task_inputs(task)
    for body in (
        "    return a @ b.transpose(-1, -2)\n",
        "    return torch.matmul(a, b.transpose(-1, -2))\n",
        "    return torch.nn.functional.linear(a, b)\n",
    ):
        source = "import torch\nimport flydsl\n\n\ndef f(a, b):\n" + body
        with pytest.raises(RuntimeError, match="matrix"):
            task_inputs.assert_candidate_is_independent(source)


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_a_flydsl_candidate_is_accepted(task):
    # The ban is on the matrix product, not on torch: a port still needs torch
    # for tensor plumbing, and `@` as a decorator must not be mistaken for one.
    task_inputs = _task_inputs(task)
    source = (
        "import functools\n"
        "import torch\n"
        "import flydsl.compiler as flyc\n"
        "\n"
        "\n"
        "@functools.lru_cache\n"
        "def build_op_module(m, n, k):\n"
        "    def launch(a, b):\n"
        "        return torch.empty((m, n), dtype=a.dtype, device=a.device)\n"
        "\n"
        "    return launch\n"
    )
    task_inputs.assert_candidate_is_independent(source)
