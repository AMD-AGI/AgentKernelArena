# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the forge_rewrite agent and the rewrite_by_flydsl task type."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forge.common import _GITIGNORE, _infer_backend, _resolve_fellow
from agents.forge_rewrite.launch_agent import (
    REWRITE_WORKSPACE_DIR,
    _build_rewrite_command,
    _locate_ported_kernel,
    _prepare_rewrite_workspace,
    _resolve_port_source,
    _rewrite_config,
)

LOGGER = logging.getLogger("test_forge_rewrite")

AGENT_CONFIG = {
    "model": "claude-opus-5",
    "timeout_seconds": 7200,
    "permission_mode": "acceptEdits",
    "snr_threshold": 30.0,
    "max_port_attempts": 3,
    "supervisor_backend": "codex",
    "rewrite_kb": False,
}


def _task_config(**overrides):
    rewrite = {
        "port_source": "/does/not/matter/fused_moe.py",
        "port_source_entry": "fused_moe",
        "port_target": "kernel.py",
        "logical_operator": "glm52_mxfp4_moe_2stage",
    }
    rewrite.update(overrides)
    return {"task_type": "rewrite_by_flydsl", "rewrite": rewrite}


def _workspace(
    tmp_path: Path,
    *,
    source_name: str = "fused_moe.py",
    with_benchmark_helper: bool = True,
) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "forge_driver.py").write_text(
        "from _aka_benchmark import benchmark_cuda_graph_or_events\n"
    )
    (scripts / "task_inputs.py").write_text("# inputs\n")
    (scripts / "task_reference.py").write_text("# reference\n")
    if with_benchmark_helper:
        # Arena materializes the canonical helper beside every importer.
        (scripts / "_aka_benchmark.py").write_text("# canonical helper\n")
    (workspace / "kernel.py").write_text("# stub\n")
    source = tmp_path / source_name
    source.write_text("def fused_moe():\n    pass\n")
    return workspace, source


def test_rewrite_by_flydsl_maps_to_the_flydsl_fellow():
    config = {"task_type": "rewrite_by_flydsl"}
    assert _infer_backend(config) == "flydsl"
    assert _resolve_fellow(config, {}) == "flydsl-fellow"


def test_gitignore_hides_the_rewrite_scratch_from_the_edit_scope_check():
    # _verify_forge_edit_scope deletes untracked files that are not declared
    # editable sources; an unignored scratch dir would lose the ported kernel.
    assert f"{REWRITE_WORKSPACE_DIR}/" in _GITIGNORE
    assert ".forge_rewrite/" in _GITIGNORE


def test_rewrite_config_requires_the_pipeline_fields():
    with pytest.raises(RuntimeError, match="no 'rewrite' mapping"):
        _rewrite_config({"task_type": "rewrite_by_flydsl"}, "config.yaml")

    config = _task_config()
    del config["rewrite"]["port_target"]
    with pytest.raises(RuntimeError, match="port_target"):
        _rewrite_config(config, "config.yaml")


def test_port_source_resolves_absolute_and_workspace_relative(tmp_path):
    workspace, source = _workspace(tmp_path)
    assert _resolve_port_source(str(workspace), str(source)) == source.resolve()

    local = workspace / "operator_entry.py"
    local.write_text("# entry\n")
    assert _resolve_port_source(str(workspace), "operator_entry.py") == local.resolve()

    with pytest.raises(RuntimeError, match="Port source not found"):
        _resolve_port_source(str(workspace), "missing.py")


def test_rewrite_workspace_carries_the_driver_and_its_modules(tmp_path):
    workspace, source = _workspace(tmp_path)
    root, source_copy, driver_copy = _prepare_rewrite_workspace(
        str(workspace), source, "kernel.py", LOGGER
    )

    assert root == workspace / REWRITE_WORKSPACE_DIR
    assert driver_copy.is_file()
    assert source_copy.name == "fused_moe.py"
    assert {path.name for path in root.glob("*.py")} == {
        "forge_driver.py",
        "task_inputs.py",
        "task_reference.py",
        "_aka_benchmark.py",
        "fused_moe.py",
    }
    # KernelForge rejects a driver whose directory shadows the candidate, so the
    # port target must never be copied in.
    assert not (root / "kernel.py").exists()


def test_rewrite_workspace_is_its_own_clean_repository(tmp_path):
    # KernelForge's agent sessions require a git worktree with a resolvable HEAD,
    # and it must be the scratch directory's own repository: git resolves a
    # repository by walking up, so otherwise the framework base commit -- and any
    # apply-back patch -- would be computed against the Arena task's files.
    workspace, source = _workspace(tmp_path)
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit",
         "--quiet", "--allow-empty", "-m", "arena workspace base"],
        cwd=workspace, check=True, capture_output=True,
    )
    arena_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True
    ).stdout.strip()

    root, _, _ = _prepare_rewrite_workspace(str(workspace), source, "kernel.py", LOGGER)

    assert (root / ".git").is_dir()
    scratch_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    assert scratch_head.returncode == 0, "the agent session needs a resolvable HEAD"
    assert scratch_head.stdout.strip() != arena_head

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    ).stdout
    assert status == "", f"the session must start from a clean worktree, got: {status!r}"


def test_rewrite_workspace_is_rebuilt_from_scratch(tmp_path):
    workspace, source = _workspace(tmp_path)
    root, _, _ = _prepare_rewrite_workspace(str(workspace), source, "kernel.py", LOGGER)
    stale = root / "stale_attempt.txt"
    stale.write_text("from a previous run\n")

    root, _, _ = _prepare_rewrite_workspace(str(workspace), source, "kernel.py", LOGGER)
    assert not stale.exists()


def test_rewrite_workspace_requires_the_canonical_benchmark_helper(tmp_path):
    # Without it the driver would time the baseline and the candidate with a
    # second implementation, and the pipeline's speedup could then disagree with
    # the score about what it measures.
    workspace, source = _workspace(tmp_path, with_benchmark_helper=False)
    with pytest.raises(RuntimeError, match="benchmark helper not found"):
        _prepare_rewrite_workspace(str(workspace), source, "kernel.py", LOGGER)


def test_rewrite_workspace_rejects_a_source_named_like_the_target(tmp_path):
    workspace, source = _workspace(tmp_path, source_name="kernel.py")
    with pytest.raises(RuntimeError, match="cannot share a file name"):
        _prepare_rewrite_workspace(str(workspace), source, "kernel.py", LOGGER)


def test_rewrite_command_forwards_the_task_contract(tmp_path):
    workspace, source = _workspace(tmp_path)
    root, source_copy, driver_copy = _prepare_rewrite_workspace(
        str(workspace), source, "kernel.py", LOGGER
    )
    config = _task_config(snr_threshold=42.0, max_port_attempts=5)

    cmd = _build_rewrite_command(
        forge_bin="kernel-agents",
        rewrite_root=root,
        source_copy=source_copy,
        driver_copy=driver_copy,
        result_json=root / "forge_experiments" / "forge_rewrite_result.json",
        rewrite=config["rewrite"],
        agent_config=AGENT_CONFIG,
        gpu_arch="gfx950",
        gpu_type="mi355x",
    )

    assert cmd[:2] == ["kernel-agents", "forge-rewrite-by-flydsl"]
    assert "--no-prepare-driver" in cmd
    assert "--no-rewrite-kb" in cmd
    assert cmd[cmd.index("--source-kernel") + 1] == str(source_copy)
    assert cmd[cmd.index("--flydsl-kernel-name") + 1] == "kernel.py"
    assert cmd[cmd.index("--logical-op-name") + 1] == "glm52_mxfp4_moe_2stage"
    assert cmd[cmd.index("--source-entry") + 1] == "fused_moe"
    assert cmd[cmd.index("--target-functions") + 1] == "fused_moe"
    assert cmd[cmd.index("--gpu-target") + 1] == "gfx950"
    assert cmd[cmd.index("--gpu-type") + 1] == "mi355x"
    # The task's tolerance and attempt budget override the agent defaults.
    assert cmd[cmd.index("--snr-threshold") + 1] == "42.0"
    assert cmd[cmd.index("--max-port-attempts") + 1] == "5"
    # 7200s minus the 900s shutdown margin.
    assert cmd[cmd.index("--max-hours") + 1] == "1.75"
    # apply-back is deliberately not requested: an Arena task has no framework
    # repository to patch.
    assert "--framework" not in cmd


def test_rewrite_command_can_enable_the_recipe_store():
    config = _task_config()
    cmd = _build_rewrite_command(
        forge_bin="kernel-agents",
        rewrite_root=Path("/tmp/ws"),
        source_copy=Path("/tmp/ws/fused_moe.py"),
        driver_copy=Path("/tmp/ws/forge_driver.py"),
        result_json=Path("/tmp/ws/result.json"),
        rewrite=config["rewrite"],
        agent_config={**AGENT_CONFIG, "rewrite_kb": True},
        gpu_arch="gfx950",
        gpu_type="mi355x",
    )
    assert "--rewrite-kb" in cmd
    assert "--no-rewrite-kb" not in cmd


def test_ported_kernel_is_found_by_attempt_path_then_by_search(tmp_path):
    root = tmp_path / REWRITE_WORKSPACE_DIR
    attempt = root / ".forge_rewrite" / "20260828-081250-fc631ed6"
    attempt.mkdir(parents=True)
    ported = attempt / "kernel.py"
    ported.write_text("# ported flydsl kernel\n")

    result = {"temporary_paths": [".forge_rewrite/20260828-081250-fc631ed6"]}
    assert _locate_ported_kernel(root, result, "kernel.py") == ported
    assert _locate_ported_kernel(root, None, "kernel.py") == ported
    assert _locate_ported_kernel(root, {"temporary_paths": []}, "other.py") is None


def test_no_ported_kernel_reports_missing(tmp_path):
    root = tmp_path / REWRITE_WORKSPACE_DIR
    (root / ".forge_rewrite").mkdir(parents=True)
    assert _locate_ported_kernel(root, None, "kernel.py") is None


def _task_inputs_module():
    task_scripts = ROOT / "tasks/SIKL-task/glm52_moe_mxfp4_per1x32_t64/scripts"
    if str(task_scripts) not in sys.path:
        sys.path.insert(0, str(task_scripts))
    import task_inputs

    return task_inputs


def test_candidate_reusing_the_framework_kernels_is_rejected():
    # The first port produced for this task imported aiter's own FlyDSL kernel
    # factories, which made the "rewrite" launch the baseline's kernels and
    # report the removal of per-call host dispatch as a speedup.
    task_inputs = _task_inputs_module()
    source = (
        "import torch\n"
        "import flydsl.compiler as flyc\n"
        "from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1\n"
        "import aiter\n"
    )
    assert task_inputs.banned_candidate_imports(source) == [
        "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage",
        "aiter",
    ]
    with pytest.raises(RuntimeError, match="imports the framework under test"):
        task_inputs.assert_candidate_is_independent(source)


def test_candidate_written_in_flydsl_is_accepted():
    task_inputs = _task_inputs_module()
    source = (
        "import torch\n"
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n"
        "\n"
        "def build_op_module(**axes):\n"
        "    return lambda *args: None\n"
    )
    assert task_inputs.banned_candidate_imports(source) == []
    task_inputs.assert_candidate_is_independent(source)


def test_the_task_measures_the_driver_and_the_score_the_same_way():
    # One set of benchmark parameters, used by both the harness that produces the
    # score and the driver the rewrite pipeline optimizes against.
    task_inputs = _task_inputs_module()
    task_root = ROOT / "tasks/SIKL-task/glm52_moe_mxfp4_per1x32_t64"
    harness = (task_root / "test_kernel_harness.py").read_text()
    driver = (task_root / "scripts/forge_driver.py").read_text()

    assert task_inputs.BENCH_TARGET_MS == 1.0
    for source in (harness, driver):
        assert "benchmark_cuda_graph_or_events" in source
        assert "task_inputs.BENCH_TARGET_MS" in source
