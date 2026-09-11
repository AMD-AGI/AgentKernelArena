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
import sys
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
    "scripts/task_initialize.py",
    "scripts/task_compare.py",
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
    # The task type names its target language and the task carries exactly one
    # field beyond Arena's existing ones. Everything else an operator2flydsl
    # provider needs -- where the port lands, who owns the source, what the
    # operator is called -- is already an Arena field, so a second provider
    # reads the same task without knowing anything about KernelForge.
    config = _config(task)
    assert config["task_type"] == "operator2flydsl"
    assert config["source_file_path"] == ["kernel.py"]
    assert config["rewrite_source_file"]
    assert config["kernel_identity"]["source_owner"]
    assert "rewrite" not in config, "the rewrite block is replaced by Arena fields"
    for key in ("snr_threshold", "max_port_attempts"):
        assert key not in config, f"{key} is agent search policy, not task data"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_builder_symbol_agrees_with_kernelforge(task):
    # KernelForge derives the required factory symbol from the task's logical
    # operator and offers no override, while the harness looks up whatever
    # workload.json names. If the two ever disagree the harness finds no factory
    # and reports the baseline as the port's score -- a silent pass.
    protocol = pytest.importorskip("kernelforge.rewrite_by_flydsl.protocol")
    config = _config(task)
    declared = _workload(task)["builder_symbol"]
    operator = config["kernel_identity"]["logical_operator"]

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
    assert workload["gate_policy"].strip()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_numeric_contract_comes_from_the_schema_callbacks(task):
    # The whole point of this suite's second revision: input construction and
    # the correctness verdict are the bundle's own callbacks, shipped as files
    # rather than reimplemented. A task that derives either from the production
    # implementation cannot promise that a candidate it accepts is a candidate
    # the acceptance run accepts, which is how 92 of 273 points came back
    # rejected after passing here.
    for name in ("task_initialize", "task_compare"):
        assert (task / "scripts" / f"{name}.py").is_file(), f"{name}.py is missing"
    workload = _workload(task)
    for key in ("gate_multiplier", "gate_floor", "atol", "rtol", "snr_threshold"):
        assert key not in workload, (
            f"{key} is a tolerance the task would own; the callback owns it"
        )
    task_inputs = _task_inputs(task)
    for name in ("derive_gates", "GATE_MULTIPLIER", "GATE_FLOOR", "SNR_CEILING_DB"):
        assert not hasattr(task_inputs, name), f"{name} is a second correctness policy"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_verdict_is_the_callback_and_nothing_else(task):
    # The verdict must be "did the callback raise", not a metric recomputed from
    # its inputs: a second implementation of the comparison is the drift this
    # revision exists to remove. AssertionError is the candidate's failure and
    # ValueError is the task's, so only the first may be caught.
    inputs = (task / "scripts" / "task_inputs.py").read_text()
    assert "task_compare.run(got, expected)" in inputs
    assert "except AssertionError" in inputs
    assert "except ValueError" not in inputs, (
        "an invalid reference is this task's bug and must not be scored as a "
        "failed candidate"
    )
    measure = (task / "scripts" / "task_measure.py").read_text()
    assert "task_inputs.verdict" in measure


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_inputs_are_built_one_case_at_a_time(task):
    # The bundle initializes one workload point at a time and draws the
    # activation before the weights, so a case's weights depend on its var axis.
    # Building every case from one pass would hand the operator a weight no
    # acceptance run ever uses.
    task_inputs = _task_inputs(task)
    assert hasattr(task_inputs, "build_case_inputs")
    assert not hasattr(task_inputs, "build_inputs"), (
        "a whole-suite builder cannot reproduce the bundle's per-point seeding"
    )
    source = (task / "scripts" / "task_inputs.py").read_text()
    assert "task_initialize.run(inputs, seed=SEED)" in source


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_the_callback_verdict_is_the_pipeline_verdict(task):
    # KernelForge's correctness stage applies `snr_db >= --snr-threshold` and
    # never reads `allclose` whenever the driver prints an SNR, so an `SNR:`
    # line here would hand PORT and OPTIMIZE keep/revert to a threshold the
    # bundle's comparison does not use.
    driver = (task / "scripts" / "forge_driver.py").read_text()
    assert driver.count('"SNR: ') + driver.count("'SNR: ") == 0, (
        "the driver prints an SNR aggregate, which overrides its own verdict"
    )
    assert "snr_threshold" not in _config(task), (
        "a PORT filter the driver's output can never reach is dead configuration"
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
def test_the_production_implementation_is_judged_beside_the_candidate(task):
    # The bar no longer comes from the baseline, but a failure is unreadable
    # without it: only the shipped implementation's own verdict separates a
    # candidate that is wrong from one that misses a bar production also misses.
    measure = (task / "scripts" / "task_measure.py").read_text()
    assert "task_baseline.run" in measure
    assert "baseline_passed" in measure


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
    would bind whichever task ran first for the whole session. Its siblings are
    imported by bare name the way the harness and the driver import them, so the
    task's own scripts/ has to lead sys.path while it loads.
    """
    scripts = str((task / "scripts").resolve())
    path = task / "scripts" / "task_inputs.py"
    spec = importlib.util.spec_from_file_location(f"task_inputs_{task.name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
        for name in ("task_compare", "task_initialize"):
            sys.modules.pop(name, None)
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
