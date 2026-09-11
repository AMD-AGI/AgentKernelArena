# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Invariants for the SIKL rewrite tasks.

Each task reimplements one workload-schema operator in FlyDSL and is scored
against that operator's production implementation over every case the schema
declares. The task carries the schema's unfilled ``kernel-forges`` solution slot
so post-processing can fill it from the scored result.

The checks here pin the couplings that fail silently rather than loudly: a
factory symbol the harness looks up but the pipeline never asks for, a gate
derived from a number nobody measured, or a driver that times the candidate
differently from the way the task is scored.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SIKL_ROOT = ROOT / "tasks" / "SIKL-task"
TASKS = sorted(path for path in SIKL_ROOT.glob("*") if (path / "config.yaml").is_file())

VAR_AXIS = {"gemm": "m", "moe": "num_tokens"}

# Files every task of an op_type carries byte-identical copies of. The suite is
# generated from one template per op_type by tooling that lives with the
# workload-schema bundle, outside this repository, so nothing in this repo can
# check a task against the generator. What it can check is the invariant the
# generator exists to maintain.
SHARED_TEMPLATE_FILES = (
    "kernel.py",
    "test_kernel_harness.py",
    "scripts/task_inputs.py",
    "scripts/task_reference.py",
    "scripts/task_baseline.py",
    "scripts/task_measure.py",
    "scripts/forge_driver.py",
)


def _config(task: Path) -> dict:
    with (task / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _workload(task: Path) -> dict:
    return json.loads((task / "workload.json").read_text())


def _of_type(op_type: str) -> list[Path]:
    return [task for task in TASKS if _workload(task)["op_type"] == op_type]


GEMM_TASKS = _of_type("gemm")
MOE_TASKS = _of_type("moe")


def test_the_suite_covers_both_operator_families():
    assert GEMM_TASKS, f"no gemm tasks found under {SIKL_ROOT}"
    assert MOE_TASKS, f"no moe tasks found under {SIKL_ROOT}"
    assert len(TASKS) == len(GEMM_TASKS) + len(MOE_TASKS)


@pytest.mark.parametrize("relative", SHARED_TEMPLATE_FILES)
@pytest.mark.parametrize("op_type", sorted(VAR_AXIS))
def test_shared_files_are_identical_across_an_op_type(op_type, relative):
    # Arena copies each task directory into its own workspace, so a task cannot
    # import from a sibling and every one carries its own copy of the harness,
    # the driver and the helpers. Editing one task's copy is the drift this
    # catches: it would silently score that operator under a different regime
    # than the rest of its family, and comparing it to the others would be
    # meaningless. Every shape constant lives in workload.json precisely so
    # these copies can stay identical.
    tasks = _of_type(op_type)
    digests: dict[str, list[str]] = {}
    for task in tasks:
        path = task / relative
        assert path.is_file(), f"{task.name} is missing {relative}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests.setdefault(digest, []).append(task.name)
    assert len(digests) == 1, (
        f"{relative} differs across the {op_type} tasks; regenerate the suite "
        f"from its op_type template instead of editing a single task: "
        f"{ {digest[:8]: names for digest, names in digests.items()} }"
    )


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
    operator = config["rewrite"]["logical_operator"]

    assert protocol.builder_symbol(operator) == declared
    assert config["target_kernel_functions"] == [declared]
    # A longer name is folded into a truncation plus a digest, which is legal
    # but unreadable and useless as a KB identity.
    assert len(operator) <= 40, "logical operator would be truncated into a digest"


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
    assert workload["gate_floor"] > 0, (
        "without a floor a near-exact baseline derives a gate no port can clear"
    )
    assert workload["gate_policy"].strip()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_both_gates_are_derived_from_one_policy(task):
    # The SNR gate is the error gate restated for a statistic that sees error
    # concentrated in a few elements rather than spread over the output, so it
    # must not introduce policy constants of its own: a second knob is a second
    # thing to keep consistent, and a hardcoded dB floor is exactly what fails
    # the MoE baseline against its own reference.
    task_inputs = _task_inputs(task)
    assert task_inputs.SNR_MARGIN_DB == pytest.approx(
        10.0 * math.log10(task_inputs.GATE_MULTIPLIER)
    )
    assert task_inputs.SNR_CEILING_DB == pytest.approx(
        -20.0 * math.log10(task_inputs.GATE_FLOOR)
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_a_case_has_to_clear_both_correctness_gates(task):
    # Either gate alone is passable by a wrong candidate: the error gate
    # averages away localized error, and SNR alone says nothing about how far
    # the operator's own implementations sit from the reference. Both are
    # applied per case, in the one place that decides whether a case passed.
    task_inputs = _task_inputs(task)
    gates = task_inputs.derive_gates({"errors": [1e-3], "snrs": [60.0]})
    assert set(gates) == {"error", "snr_db"}

    measure = (task / "scripts" / "task_measure.py").read_text()
    for expression in ('record["error"] <= gates["error"]', 'record["snr"] >= gates["snr_db"]'):
        assert expression in measure, f"passes() does not apply {expression}"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_port_filter_is_looser_than_the_scoring_gate(task):
    # KernelForge drops a port below config.yaml's rewrite.snr_threshold before
    # the optimize loop ever sees it. If that filter were stricter than the gate
    # the task is scored on, PORT would discard candidates Arena would have
    # accepted -- and for the MoE family a 30 dB filter rejects anything merely
    # as accurate as the production implementation.
    task_inputs = _task_inputs(task)
    port_filter = float(_config(task)["rewrite"]["snr_threshold"])
    # The scoring gate is derived from the baseline at run time; the worst it can
    # demand is the ceiling case, and the filter has to stay under that.
    strictest_scoring_gate = task_inputs.SNR_CEILING_DB - task_inputs.SNR_MARGIN_DB
    assert port_filter <= strictest_scoring_gate, (
        f"rewrite.snr_threshold={port_filter} dB can exceed the derived scoring "
        f"gate, whose strictest value is {strictest_scoring_gate:.2f} dB"
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_workload_records_no_measurement(task):
    # Tolerances and baseline timings are derived at run time on the machine
    # that is about to score the candidate. Recording them here would pin
    # numbers that go stale against a different GPU or framework build, and
    # Arena measures its own baseline every run regardless.
    workload = _workload(task)
    for key in ("max_relerr", "tolerance_reason"):
        assert key not in workload, f"{key} is a recorded measurement"
    var_axis = VAR_AXIS[workload["op_type"]]
    for case in workload["cases"]:
        assert set(case) == {"case_id", "uuid", var_axis}, (
            f"{case['case_id']} carries more than the schema's case identity"
        )


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_gate_is_derived_from_the_baseline(task):
    measure = (task / "scripts" / "task_measure.py").read_text()
    assert "task_inputs.derive_gates" in measure
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
    # aiter's tuned dispatch resolves to aiter's own FlyDSL kernels at many of
    # the cases, so a port that imports them measures the baseline against
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
@pytest.mark.parametrize(
    "source",
    (
        "import task_baseline\n\ndef f(**kwargs):\n    return task_baseline.run(**kwargs)\n",
        "from scripts.task_measure import baseline_calls\n",
    ),
)
def test_a_candidate_importing_task_implementation_is_rejected(task, source):
    task_inputs = _task_inputs(task)
    with pytest.raises(RuntimeError, match="protected task"):
        task_inputs.assert_candidate_is_independent(source)


@pytest.mark.parametrize("task", GEMM_TASKS, ids=lambda task: task.name)
def test_a_gemm_candidate_computing_with_torch_is_rejected(task):
    # torch's matmul IS the baseline at the larger M cases of a GEMM, so
    # `a @ b.T` would tie it exactly while implementing no kernel at all. The
    # MoE tasks carry no such rule: a Python loop over experts is orders of
    # magnitude slower than the fused baseline, so it is not a way to tie it.
    task_inputs = _task_inputs(task)
    for body in (
        "    return a @ b.transpose(-1, -2)\n",
        "    return torch.matmul(a, b.transpose(-1, -2))\n",
        "    return torch.nn.functional.linear(a, b)\n",
        "    product = torch.matmul\n    return product(a, b.transpose(-1, -2))\n",
    ):
        source = "import torch\nimport flydsl\n\n\ndef f(a, b):\n" + body
        with pytest.raises(RuntimeError, match="matrix"):
            task_inputs.assert_candidate_is_independent(source)

    source = (
        "from torch import matmul as product\nimport flydsl\n\n\n"
        "def f(a, b):\n    return product(a, b.transpose(-1, -2))\n"
    )
    with pytest.raises(RuntimeError, match="matrix"):
        task_inputs.assert_candidate_is_independent(source)


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_a_flydsl_candidate_is_accepted(task):
    # The ban is on reusing the framework, not on torch: a port still needs
    # torch for tensor plumbing, and `@` as a decorator must not be mistaken for
    # a matrix product.
    task_inputs = _task_inputs(task)
    source = (
        "import functools\n"
        "import torch\n"
        "import flydsl.compiler as flyc\n"
        "\n"
        "\n"
        "@functools.lru_cache\n"
        "def build_op_module(**axes):\n"
        "    def launch(*args):\n"
        "        return torch.empty(0)\n"
        "\n"
        "    return launch\n"
    )
    task_inputs.assert_candidate_is_independent(source)
