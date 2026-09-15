"""Image-suite tasks use the shared v2 contract, acquisition and prompt path.

CPU/filesystem regression tests only; these do not qualify an image or GPU.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from src.task_materialization import (
    MaterializationError, load_materialization_record, materialize_task_workspace,
)
from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec, resolve_task_path

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
IMAGE_CONFIGS = sorted((ROOT / "tasks/image_kernel").glob("*/config.yaml"))


def _load_image_task(config_path):
    # Reuse the public schema parser; image acquisition is not a task type.
    spec = load_task_spec(config_path, task_id="image_kernel/" + config_path.parent.name)
    config = spec.to_mapping()
    if "kernel_identity" in config:
        assert set(config["kernel_identity"]) == {"logical_operator", "source_owner"}
    assert config["workspace"]["sources"]  # image or explicitly pinned Git; no family dispatch
    # This validates declarations only. PASS below is a protocol fixture, never
    # a claim that the task's validate-task action ran on its actual image.
    workload = resolve_task_path(config_path.parent, config["evaluation"]["workloads"], must_exist=True)
    rows = json.loads(workload.read_text())["cases"]
    envelope = dict(protocol="arena-eval-v1", role="task", action="validate-task",
                    status="PASS", cases=[dict(row, status="PASS") for row in rows])
    result = parse_command_result(RESULT_PREFIX + json.dumps(envelope),
                                  role="task", action="validate-task", returncode=0)
    return spec, CaseManifest.from_result(result)


@pytest.mark.parametrize("config_path", IMAGE_CONFIGS, ids=lambda p: p.parent.name)
def test_image_configs_use_shared_schema_and_task_owned_workloads(config_path):
    spec, manifest = _load_image_task(config_path)
    assert spec.candidate.initial_state == "implemented"
    assert spec.baseline.kind == "initial_candidate"
    assert spec.baseline.correctness_policy == "required"
    assert spec.candidate.editable and manifest.cases
    assert {a.role for a in spec.actions} == {"task", "baseline", "candidate"}
    # README carries operator/layout/dispatch constraints formerly in prompt fields.
    assert (config_path.parent / "README.md").read_text().strip()


def test_mi355x_image_configs_retain_canonical_operator_identity():
    configs = [path for path in IMAGE_CONFIGS if path.parent.name.startswith("mi355x_")]
    assert len(configs) == 16
    for path in configs:
        spec, _ = _load_image_task(path)
        assert set(spec.to_mapping()["kernel_identity"]) == {"logical_operator", "source_owner"}


@pytest.mark.parametrize("field,value", [
    ("task_type", "image_kernel"),
    ("source_file_path", ["old/path.py"]),
    ("target_kernel_functions", ["old_symbol"]),
    ("repository_language", "hip"),
    ("image_repo_path", "/old/image/repository"),
])
def test_image_config_rejects_mixed_legacy_fields(field, value):
    raw = yaml.safe_load(IMAGE_CONFIGS[0].read_text())
    raw[field] = value
    with pytest.raises(TaskConfigError, match="unknown fields"):
        TaskSpec.from_mapping(raw, task_id="image_kernel/fixture")


@pytest.mark.parametrize("field,value", [("kernel_kind", "ck"), ("workload", {"source": "cases.json"})])
def test_image_identity_does_not_reintroduce_backend_or_workload_schema(field, value):
    raw = yaml.safe_load(IMAGE_CONFIGS[0].read_text())
    raw["kernel_identity"] = {"logical_operator": "attention", "source_owner": "aiter", field: value}
    with pytest.raises(TaskConfigError, match="unknown fields"):
        TaskSpec.from_mapping(raw, task_id="image_kernel/fixture")


def _mk_workspace(tmp_path):
    ws = tmp_path / "ws"
    path = ws / "vendor/aiter/csrc/k.cuh"
    path.parent.mkdir(parents=True)
    path.write_text("// kernel\n")
    (ws / "kernel.py").write_text("def compute(): return 1\n")
    return ws


@pytest.mark.parametrize("relative", ["vendor/aiter/csrc/k.cuh", "kernel.py"])
def test_shared_source_resolution_preserves_declared_task_paths(tmp_path, relative):
    workspace = _mk_workspace(tmp_path)
    assert resolve_task_path(workspace, relative, must_exist=True) == workspace / relative


def test_shared_resolution_does_not_guess_image_basename(tmp_path):
    workspace = _mk_workspace(tmp_path)
    with pytest.raises(TaskConfigError):
        resolve_task_path(workspace, "csrc/k.cuh", must_exist=True)


@pytest.mark.parametrize("relative", ["../outside.py", "/outside.py", "missing.py"])
def test_shared_resolution_rejects_missing_or_escaping_source(tmp_path, relative):
    workspace = _mk_workspace(tmp_path)
    with pytest.raises(TaskConfigError):
        resolve_task_path(workspace, relative, must_exist=True)


# --------------------------------------------------------------------------
# 1b. forge --max-hours must track the run's timeout budget (bootstrap patches
#     timeout_seconds but not max_hours), otherwise a long run is capped early.
# --------------------------------------------------------------------------
def test_forge_max_hours_tracks_timeout():
    from agents.forge.launch_agent import _forge_max_hours

    # 32h run -> ~31.75h loop budget (15-min margin under the hard kill)
    assert _forge_max_hours({"timeout_seconds": 115200}) == 31.75
    # default timeout (29700s) -> ~8h, matching the previous static cap
    assert _forge_max_hours({"timeout_seconds": 29700}) == 8.0
    # never negative / never below the floor
    assert _forge_max_hours({"timeout_seconds": 60}) >= 0.1
    assert _forge_max_hours({}) >= 0.1


def _process_is_running(pid: int) -> bool:
    """Treat a reparented zombie as stopped: it cannot edit files or hold the GPU."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        state = stat_path.read_text().split()[2]
    except (FileNotFoundError, IndexError, OSError):
        return False
    return state not in {"Z", "X"}


@pytest.mark.skipif(
    not hasattr(os, "killpg") or not Path("/proc/self/stat").exists(),
    reason="requires Linux process groups and /proc state",
)
def test_terminate_process_group_kills_descendant_after_leader_exits():
    """A child that ignores SIGTERM must not survive an early-exiting leader."""
    from agents.forge.launch_agent import _terminate_process_group

    child_code = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(os.getpid(), flush=True); "
        "time.sleep(60)"
    )
    leader_code = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "child.wait()"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    pgid = os.getpgid(leader.pid)

    try:
        _terminate_process_group(
            leader,
            logging.getLogger("test_process_group_cleanup"),
            term_timeout=0.2,
            kill_timeout=1,
        )

        deadline = time.monotonic() + 1
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(child_pid)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _mk_fake_image_repo(tmp_path):
    source = tmp_path / "image_aiter"
    (source / "csrc").mkdir(parents=True)
    (source / "csrc/k.cuh").write_text("// original kernel\n")
    (source / "aiter/jit/build").mkdir(parents=True)
    (source / "aiter/jit/build/cached.so").write_text("cache\n")
    (source / ".git").mkdir()
    (source / ".git/config").write_text("[core]\n")
    return source


def _mk_image_task(tmp_path, source):
    task = tmp_path / "task"
    (task / "scripts").mkdir(parents=True)
    # Materialization never executes evaluation actions; no fake GPU result.
    (task / "scripts/evaluate.py").write_text("raise SystemExit('acquisition fixture only')\n")
    (task / "README.md").write_text("Keep paged attention layouts and all case-specific numerical gates.\n")
    config = {
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["vendor/aiter/csrc/k.cuh"]},
        "workspace": {"sources": [{"kind": "image", "image_path": str(source),
                                    "destination": "vendor/aiter", "exclude": ["aiter/jit/build"]}]},
        "evaluation": {"runner": ["python3", "scripts/evaluate.py"]},
    }
    path = task / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_task_spec(path, task_id="fixture/image_operator"), path


def test_v2_image_acquisition_preserves_source_and_excludes_git_cache(tmp_path):
    source = _mk_fake_image_repo(tmp_path)
    spec, config = _mk_image_task(tmp_path, source)
    workspace = materialize_task_workspace(spec, config, tmp_path / "workspace")
    assert (workspace / "vendor/aiter/csrc/k.cuh").read_bytes() == (source / "csrc/k.cuh").read_bytes()
    assert not (workspace / "vendor/aiter/.git").exists()
    assert not (workspace / "vendor/aiter/aiter/jit/build").exists()
    assert (source / "aiter/jit/build/cached.so").exists()
    assert (source / ".git/config").exists()
    assert not (config.parent / "vendor").exists()
    acquisition = load_materialization_record(workspace)["sources"][0]
    assert acquisition["input_tree_sha256"] == acquisition["copied_tree_sha256"]
    assert acquisition["declaration"]["destination"] == "vendor/aiter"


def test_v2_image_resume_preserves_candidate_and_frozen_baseline(tmp_path):
    from src.task_session import TaskSession

    source = _mk_fake_image_repo(tmp_path)
    spec, config = _mk_image_task(tmp_path, source)
    workspace = materialize_task_workspace(spec, config, tmp_path / "workspace")
    session = TaskSession.create(spec, workspace, workspace.parent / ".task-sessions" / workspace.name)
    candidate = workspace / "vendor/aiter/csrc/k.cuh"
    candidate.write_text("// edited candidate\n")
    assert materialize_task_workspace(spec, config, workspace) == workspace
    assert candidate.read_text() == "// edited candidate\n"
    assert (session.baseline_workspace / "vendor/aiter/csrc/k.cuh").read_text() == "// original kernel\n"
    assert (source / "csrc/k.cuh").read_text() == "// original kernel\n"


def test_v2_missing_image_is_explicit_materialization_failure(tmp_path):
    spec, config = _mk_image_task(tmp_path, tmp_path / "missing-image")
    with pytest.raises((FileNotFoundError, MaterializationError)):
        materialize_task_workspace(spec, config, tmp_path / "workspace")


def test_prompt_builder_uses_v2_image_contract_and_readme(tmp_path):
    from src.prompt_builder import prompt_builder

    source = _mk_fake_image_repo(tmp_path)
    spec, config = _mk_image_task(tmp_path, source)
    workspace = materialize_task_workspace(spec, config, tmp_path / "workspace")
    prompt = prompt_builder(str(config), workspace,
                            {"target_gpu_model": "MI355X", "_task_id": spec.task_id}, LOG)
    assert "vendor/aiter/csrc/k.cuh" in prompt
    assert "Required final implementation backend: hip" in prompt
    assert "Performance baseline: initial_candidate" in prompt
    assert (workspace / "README.md").read_text().strip() in prompt
    for action in ("compile", "correctness", "performance"):
        assert f"python3 scripts/evaluate.py candidate {action}" in prompt
    assert "separate frozen baseline workspace" in prompt


def test_mi355x_validator_discovers_every_mi355x_image_task(monkeypatch):
    from main import _discover_tasks

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = yaml.safe_load(
        (root / "example_configs/pr69_image_kernel_validator_mi355x.yaml").read_text()
    )
    discovered = _discover_tasks(config["tasks"])
    expected = {
        f"image_kernel/{path.parent.name}"
        for path in (root / "tasks/image_kernel").glob("mi355x_*/config.yaml")
    }

    assert expected
    assert set(discovered) == expected


def test_task_discovery_rejects_unmatched_selectors(monkeypatch):
    from main import _discover_tasks

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)

    with pytest.raises(ValueError, match="matched no task configs"):
        _discover_tasks(["image_kernel/task_that_does_not_exist"])
