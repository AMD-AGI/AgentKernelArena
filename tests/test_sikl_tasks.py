# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Invariants shared by the SIKL tasks for the GLM-5.2 MXFP4 MoE operator.

Two tasks measure the same operator from opposite directions: one rewrites it in
FlyDSL from scratch (rewrite_by_flydsl), the other optimizes the FlyDSL
implementation aiter ships (image_kernel). Their scores are only comparable
while they keep measuring the same thing the same way, which is what these
tests hold in place.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIKL = ROOT / "tasks/SIKL-task"
GENERATOR = SIKL / "generate_shape_tasks.py"
SHAPES = (8, 16, 32, 64, 128, 256)

REWRITE_TASK = SIKL / "glm52_moe_mxfp4_per1x32_t64"
OPTIMIZE_TASK = SIKL / "glm52_moe_mxfp4_per1x32_t64_flydsl_opt"
TASKS = (REWRITE_TASK, OPTIMIZE_TASK)

REWRITE_FAMILY = tuple(SIKL / f"glm52_moe_mxfp4_per1x32_t{n}" for n in SHAPES)
OPTIMIZE_FAMILY = tuple(SIKL / f"glm52_moe_mxfp4_per1x32_t{n}_flydsl_opt" for n in SHAPES)
ALL_TASKS = REWRITE_FAMILY + OPTIMIZE_FAMILY

# Present in every task of a family and required to be byte-identical: Arena
# copies each task into its own workspace, so a divergent copy would silently
# score one shape differently from the rest.
SHARED_FILES = (
    "test_kernel_harness.py",
    "scripts/forge_driver.py",
    "scripts/task_inputs.py",
    "scripts/task_reference.py",
    "scripts/task_baseline.py",
)

# aiter's MXFP4 quantizer, used by the correctness reference. An agent that
# could edit it would move the operator and its oracle together.
REFERENCE_QUANTIZER = "aiter/ops/quant.py"


def _config(task: Path) -> dict:
    return yaml.safe_load((task / "config.yaml").read_text())


def _inputs_module(task: Path):
    scripts = str(task / "scripts")
    if scripts in sys.path:
        sys.path.remove(scripts)
    sys.path.insert(0, scripts)
    for stale in ("task_inputs",):
        sys.modules.pop(stale, None)
    import task_inputs

    return task_inputs


def test_every_schema_workload_has_both_tasks():
    for task in ALL_TASKS:
        assert task.is_dir(), f"missing task: {task.name}"


@pytest.mark.parametrize(
    "family", (REWRITE_FAMILY, OPTIMIZE_FAMILY), ids=("rewrite", "flydsl_opt")
)
@pytest.mark.parametrize("shared", SHARED_FILES)
def test_a_family_ships_one_implementation(family, shared):
    digests = {(task / shared).read_bytes() for task in family}
    assert len(digests) == 1, f"{shared} differs across the family"


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.name)
def test_workload_matches_the_task_name(task):
    workload = json.loads((task / "workload.json").read_text())
    expected = int(task.name.split("_t")[1].split("_")[0])
    assert workload["num_tokens"] == expected
    assert workload["definition"] == (
        "aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128"
    )
    # A gate without its measured floor and a reason is a number nobody can
    # audit later.
    assert workload["max_relerr"] >= workload["measured"]["baseline_relerr_vs_reference"]
    assert workload["tolerance_reason"].strip()


def test_generated_tasks_are_up_to_date():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_task_ships_a_driver_and_a_harness(task):
    assert (task / "test_kernel_harness.py").is_file()
    assert (task / "scripts" / "forge_driver.py").is_file()
    # The canonical helper is materialized at run time and must never be
    # committed under tasks/.
    assert not (task / "_aka_benchmark.py").exists()
    assert not (task / "scripts" / "_aka_benchmark.py").exists()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_driver_and_harness_measure_the_same_way(task):
    harness = (task / "test_kernel_harness.py").read_text()
    driver = (task / "scripts" / "forge_driver.py").read_text()
    for source in (harness, driver):
        assert "benchmark_cuda_graph_or_events" in source
        assert "task_inputs.BENCH_TARGET_MS" in source


def test_both_tasks_declare_the_same_workload():
    rewrite, optimize = (_inputs_module(task) for task in TASKS)
    for attribute in (
        "NUM_TOKENS", "MODEL_DIM", "INTER_DIM", "NUM_EXPERTS", "TOPK",
        "SEED", "MAX_RELERR", "ACTIVATION", "DOWEIGHT_STAGE1",
        "BENCH_WARMUP", "BENCH_REPETITION", "BENCH_TARGET_MS",
    ):
        assert getattr(rewrite, attribute) == getattr(optimize, attribute), attribute


def test_rewrite_task_is_driven_by_the_rewrite_pipeline():
    config = _config(REWRITE_TASK)
    assert config["task_type"] == "rewrite_by_flydsl"
    rewrite = config["rewrite"]
    assert rewrite["port_target"] == "kernel.py"
    assert config["source_file_path"] == ["kernel.py"]
    assert rewrite["port_source"].endswith("aiter/fused_moe.py")


@pytest.mark.parametrize("task", REWRITE_FAMILY, ids=lambda task: task.name)
def test_rewrite_builder_symbol_agrees_with_kernelforge(task):
    # KernelForge derives the required factory symbol from the task's logical
    # operator and offers no override, while the harness looks up whatever
    # workload.json names. If the two ever disagree the harness finds no factory
    # and reports the aiter baseline as the port's score -- a silent pass. The
    # operator carries the shape, so this also pins one KB identity per shape.
    protocol = pytest.importorskip("kernelforge.rewrite_by_flydsl.protocol")
    config = _config(task)
    workload = json.loads((task / "workload.json").read_text())
    operator = config["rewrite"]["logical_operator"]
    declared = workload["builder_symbol"]

    assert operator.endswith(f"_t{workload['num_tokens']}")
    assert protocol.builder_symbol(operator) == declared
    assert config["target_kernel_functions"] == [declared]


def test_rewrite_stub_defines_no_shape_specific_factory():
    # The stub is byte-identical across shapes while the factory name is not, so
    # a hardcoded factory here could only ever match one shape.
    stub = (REWRITE_TASK / "kernel.py").read_text()
    assert "def build_" not in stub
    for task in REWRITE_FAMILY:
        declared = json.loads((task / "workload.json").read_text())["builder_symbol"]
        assert declared not in stub


@pytest.mark.parametrize("task", OPTIMIZE_FAMILY, ids=lambda task: task.name)
def test_optimize_workload_declares_no_builder_symbol(task):
    # The optimize family edits aiter in place and exposes no factory; carrying
    # an unused symbol would leave a reader ruling it out.
    assert "builder_symbol" not in json.loads((task / "workload.json").read_text())


def test_optimize_task_edits_the_seeded_flydsl_sources_only():
    config = _config(OPTIMIZE_TASK)
    assert config["task_type"] == "image_kernel"
    assert config["image_repo_path"] == "/sgl-workspace/aiter"
    assert config["repo_subdir"] == "aiter"
    assert config["repository_language"] == "flydsl"
    assert config["kernel_identity"]["kernel_kind"] == "flydsl"

    # The forge anchor is passed as --kernel and forge-loop's task statement
    # names it, so it has to hold the compute core, not the dispatcher.
    assert config["source_file_path"][0].endswith("kernels/mixed_moe_gemm_2stage.py")

    scope = [*config["source_file_path"], *config["editable_sources"]]
    assert all(path.startswith("aiter/") for path in scope), scope
    # Same-language task: no HIP sources, and the reference's quantizer stays out.
    assert not [path for path in scope if path.startswith("csrc/")]
    assert not [path for path in scope if path.endswith((".cu", ".cpp", ".h", ".hpp"))]
    assert REFERENCE_QUANTIZER not in [path[len("aiter/"):] for path in scope]


def test_optimize_task_fails_closed_on_the_in_image_aiter():
    # If the import resolves past the workspace copy the agent's edits are
    # invisible and every number the task reports describes the shipped kernel.
    harness = (OPTIMIZE_TASK / "test_kernel_harness.py").read_text()
    driver = (OPTIMIZE_TASK / "scripts" / "forge_driver.py").read_text()
    for source in (harness, driver):
        assert "task_inputs.use_workspace_aiter()" in source
        assert "assert_aiter_is_workspace_copy" in source
        # The shadowing has to be installed before anything imports aiter.
        assert source.index("use_workspace_aiter()") < source.index("import torch")
