# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Invariants shared by the SIKL tasks for the GLM-5.2 MXFP4 MoE operator.

Two tasks measure the same operator from opposite directions: one rewrites it in
FlyDSL from scratch (rewrite_by_flydsl), the other optimizes the FlyDSL
implementation aiter ships (image_kernel). Their scores are only comparable
while they keep measuring the same thing the same way, which is what these
tests hold in place.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REWRITE_TASK = ROOT / "tasks/SIKL-task/glm52_moe_mxfp4_per1x32_t64"
OPTIMIZE_TASK = ROOT / "tasks/SIKL-task/glm52_moe_mxfp4_per1x32_t64_flydsl_opt"
TASKS = (REWRITE_TASK, OPTIMIZE_TASK)

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
