"""Local filesystem/process fixtures; no GPU, network, or task code changes."""
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest
import yaml

from src import task_materialization as materialization
from src.preprocessing import get_task_workspace_path, is_task_complete, setup_workspace
from src.task_materialization import (
    MaterializationError, MaterializationTimeout, load_materialization_record,
    materialization_state_directory, verify_original_materialization,
)
from src.task_session import TaskSession
from src.task_spec import TaskConfigError, load_task_spec


LOG = logging.getLogger(__name__)
TASK_ID = "fixture/operators/nested_kernel"


def write(root, relative, text="original"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def task(tmp_path, *, candidate="source/kernel.py", empty=False, sources=None, setup=None, timeout=30):
    root = tmp_path / "task"
    write(root, "scripts/evaluate.py", "pass\n")
    write(root, "references/reference.py", "reference = 1\n")
    write(root, "docs/operator.md", "Compute the declared operator.\n")
    write(root, "inputs/cases.json", "[]\n")
    if not empty and sources is None:
        write(root, candidate, "def compute(): return 1\n")
    config = {
        "schema_version": 2,
        "candidate": {"language": "triton", "initial_state": "unimplemented" if empty else "implemented",
                      "editable": [candidate], "entrypoints": [{"file": candidate, "kind": "function", "symbol": "compute"}]},
        "baseline": {"kind": "provided", "source_files": ["references/reference.py"]},
        "instructions": ["docs/operator.md"],
        "evaluation": {"runner": ["python3", "scripts/evaluate.py"], "workloads": "inputs/cases.json"},
        "workspace": {"timeout_s": timeout, "sources": sources or []},
        "platform_support": {"required_arch": "gfx950", "status": "active"},
    }
    if setup is not None:
        config["workspace"]["setup"] = setup
    path = write(root, "config.yaml", yaml.safe_dump(config))
    return path, config


def run(config, tmp_path, *, task_id=TASK_ID):
    return setup_workspace(str(config), tmp_path / "runs", "stamp", LOG, task_name=task_id)


def expected_workspace(tmp_path):
    return get_task_workspace_path(tmp_path / "runs", TASK_ID, "stamp")


def record_unchecked(workspace):
    return json.loads((materialization_state_directory(workspace) / "materialization.json").read_text())


def create_session(config, workspace):
    return TaskSession.create(load_task_spec(config, task_id=TASK_ID), workspace,
                              workspace.parent / ".task-sessions" / workspace.name)


def test_fresh_workspace_has_external_provenance_and_nested_paths(tmp_path):
    config, original = task(tmp_path)
    workspace = run(config, tmp_path)
    record = load_materialization_record(workspace)
    assert record["task_id"] == TASK_ID
    assert record["original_config_sha256"]
    assert record["task_package_sha256"]
    assert record["initial_workspace_sha256"]
    assert record["original_candidate_sha256"]
    assert record["workspace"] == str(workspace)
    assert (workspace / "source/kernel.py").is_file()
    assert not (workspace / "kernel.py").exists()
    assert not materialization_state_directory(workspace).is_relative_to(workspace)
    assert "references/reference.py" in record["protected_files"]
    assert yaml.safe_load((workspace / "config.yaml").read_text()) == original
    assert yaml.safe_load(config.read_text()) == original


def test_old_task_reports_are_not_copied_or_used_as_completion(tmp_path):
    config, _ = task(tmp_path)
    for name in ("task_result.yaml", "validation_report.yaml", ".validation_complete"):
        write(config.parent, name, "stale output")
    workspace = run(config, tmp_path)
    for name in ("task_result.yaml", "validation_report.yaml", ".validation_complete"):
        assert not (workspace / name).exists()
        assert (config.parent / name).read_text() == "stale output"
    assert not is_task_complete(tmp_path / "runs", TASK_ID, "stamp")


def test_image_copy_keeps_nested_destination_excludes_and_source_unchanged(tmp_path):
    image = tmp_path / "image"
    write(image, "kernels/op.py", "def compute(): return 1\n")
    write(image, "kernels/cache/large.bin", "cache")
    write(image, ".git/config", "image git metadata")
    write(image, "third_party/.git/config", "nested git metadata")
    source = {"kind": "image", "image_path": str(image), "destination": "vendor/backend",
              "exclude": ["kernels/cache"]}
    config, _ = task(tmp_path, candidate="vendor/backend/kernels/op.py", sources=[source])
    before = sorted(str(p.relative_to(config.parent)) for p in config.parent.rglob("*"))
    workspace = run(config, tmp_path)
    assert (workspace / "vendor/backend/kernels/op.py").is_file()
    assert not (workspace / "vendor/backend/kernels/cache").exists()
    assert not (workspace / "vendor/backend/.git").exists()
    assert not (workspace / "vendor/backend/third_party/.git").exists()
    assert (image / "kernels/cache/large.bin").is_file()
    assert before == sorted(str(p.relative_to(config.parent)) for p in config.parent.rglob("*"))
    provenance = load_materialization_record(workspace)["sources"][0]
    assert provenance["declaration"] == source
    assert provenance["input_tree_sha256"] == provenance["copied_tree_sha256"]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def test_local_git_uses_exact_pin_not_head_and_resume_needs_no_network(tmp_path, monkeypatch):
    repository = tmp_path / "upstream"
    repository.mkdir()
    git(repository, "init")
    write(repository, "nested/kernel.py", "def compute(): return 1\n")
    write(repository, ".gitattributes", "*.py filter=unwanted\n")
    git(repository, "add", ".")
    git(repository, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "original")
    pinned = git(repository, "rev-parse", "HEAD")
    write(repository, "nested/kernel.py", "def compute(): return 2\n")
    git(repository, "add", ".")
    git(repository, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "later")
    upstream_head = git(repository, "rev-parse", "HEAD")
    source = {"kind": "git", "url": str(repository), "revision": pinned, "destination": "vendor/library"}
    config, _ = task(tmp_path, candidate="vendor/library/nested/kernel.py", sources=[source])
    global_config = write(tmp_path, "user-gitconfig", '[filter "unwanted"]\n smudge = touch filter-was-run; cat\n')
    with monkeypatch.context() as env:
        env.setenv("GIT_DIR", str(repository / ".git"))
        env.setenv("GIT_WORK_TREE", str(repository))
        env.setenv("GIT_CONFIG_GLOBAL", str(global_config))
        workspace = run(config, tmp_path)
    assert git(repository, "rev-parse", "HEAD") == upstream_head
    assert not (workspace / "vendor/library/filter-was-run").exists()
    assert (workspace / "vendor/library/nested/kernel.py").read_text() == "def compute(): return 1\n"
    assert git(workspace / "vendor/library", "rev-parse", "HEAD") == pinned
    assert load_materialization_record(workspace)["sources"][0]["revision"] == pinned
    assert not (config.parent / "vendor").exists()
    create_session(config, workspace)
    repository.rename(tmp_path / "upstream-unavailable")
    (workspace / "vendor/library/nested/kernel.py").write_text("optimized candidate")
    assert run(config, tmp_path) == workspace
    assert (workspace / "vendor/library/nested/kernel.py").read_text() == "optimized candidate"


def test_setup_argv_is_literal_and_uses_workspace_root(tmp_path):
    code = "import pathlib, sys; pathlib.Path('argument.txt').write_text(sys.argv[1])"
    literal = "literal; $(touch unexpected) `touch also-unexpected`"
    config, _ = task(tmp_path, setup=[["python3", "-c", code, literal]])
    workspace = run(config, tmp_path)
    assert (workspace / "argument.txt").read_text() == literal
    assert not (workspace / "unexpected").exists()
    assert not (workspace / "also-unexpected").exists()
    assert not (config.parent / "argument.txt").exists()


def test_resume_preserves_candidate_and_does_not_run_setup_again(tmp_path):
    code = "from pathlib import Path; p=Path('setup-count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
    config, _ = task(tmp_path, setup=[[sys.executable, "-c", code]])
    workspace = run(config, tmp_path)
    marker = materialization_state_directory(workspace) / "materialization.json"
    marker_before = marker.read_bytes()
    session = create_session(config, workspace)
    (workspace / "source/kernel.py").write_text("optimized candidate")
    assert run(config, tmp_path) == workspace
    assert (workspace / "source/kernel.py").read_text() == "optimized candidate"
    assert (workspace / "setup-count").read_text() == "1"
    assert marker.read_bytes() == marker_before
    assert (session.baseline_workspace / "source/kernel.py").read_text() == "def compute(): return 1\n"


@pytest.mark.parametrize("change", ["config", "package", "protected", "marker"])
def test_resume_rejects_changed_identity_or_protected_inputs_without_overwrite(tmp_path, change):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    candidate = workspace / "source/kernel.py"
    candidate.write_text("optimized candidate")
    if change == "config":
        config.write_text(config.read_text() + "\n# changed original configuration\n")
    elif change == "package":
        write(config.parent, "scripts/evaluate.py", "changed upstream harness")
    elif change == "protected":
        write(workspace, "references/reference.py", "changed reference")
    else:
        marker = materialization_state_directory(workspace) / "materialization.json"
        value = json.loads(marker.read_text())
        value["original_config_sha256"] = "forged"
        marker.write_text(json.dumps(value))
    with pytest.raises(MaterializationError):
        run(config, tmp_path)
    assert candidate.read_text() == "optimized candidate"


def test_existing_workspace_without_marker_is_never_overwritten(tmp_path):
    config, _ = task(tmp_path)
    workspace = expected_workspace(tmp_path)
    write(workspace, "source/kernel.py", "user candidate")
    with pytest.raises(MaterializationError, match="record"):
        run(config, tmp_path)
    assert (workspace / "source/kernel.py").read_text() == "user candidate"
    assert not (workspace / "scripts").exists()


def test_image_change_rejects_resume_without_clobbering_candidate(tmp_path):
    image = tmp_path / "image"
    write(image, "kernel.py", "original")
    config, _ = task(tmp_path, candidate="vendor/kernel.py", sources=[{
        "kind": "image", "image_path": str(image), "destination": "vendor"}])
    workspace = run(config, tmp_path)
    write(workspace, "vendor/kernel.py", "optimized")
    write(image, "kernel.py", "new image bytes")
    with pytest.raises(MaterializationError, match="image source changed"):
        run(config, tmp_path)
    assert (workspace / "vendor/kernel.py").read_text() == "optimized"


def test_source_destination_conflict_preserves_both_inputs(tmp_path):
    image = tmp_path / "image"
    write(image, "kernel.py", "image candidate")
    config, _ = task(tmp_path, sources=[{"kind": "image", "image_path": str(image), "destination": "scripts"}])
    with pytest.raises(MaterializationError, match="refusing to overwrite"):
        run(config, tmp_path)
    workspace = expected_workspace(tmp_path)
    assert (workspace / "scripts/evaluate.py").read_text() == "pass\n"
    assert (config.parent / "scripts/evaluate.py").read_text() == "pass\n"
    assert not (workspace / "scripts/kernel.py").exists()
    assert record_unchecked(workspace)["status"] == "failed"


def test_failed_setup_is_not_success_and_cannot_be_implicitly_restarted(tmp_path):
    config, _ = task(tmp_path, setup=[[sys.executable, "-c", "raise SystemExit(7)"]])
    with pytest.raises(MaterializationError, match="exit code 7"):
        run(config, tmp_path)
    workspace = expected_workspace(tmp_path)
    assert record_unchecked(workspace)["status"] == "failed"
    write(workspace, "user-artifact", "keep me")
    with pytest.raises(MaterializationError, match="not complete"):
        run(config, tmp_path)
    assert (workspace / "user-artifact").read_text() == "keep me"


def test_setup_commands_share_one_total_deadline(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(0.65)"]
    config, _ = task(tmp_path, setup=[command, command], timeout=1)
    started = time.monotonic()
    with pytest.raises(MaterializationTimeout):
        run(config, tmp_path)
    assert time.monotonic() - started < 3
    assert record_unchecked(expected_workspace(tmp_path))["status"] == "failed"


def test_copy_also_uses_the_total_deadline(tmp_path, monkeypatch):
    config, _ = task(tmp_path)
    original = materialization._run
    def expired_copy(argv, cwd, deadline, state, label, env=None):
        if label == "copy":
            deadline.end = time.monotonic() - 1
        return original(argv, cwd, deadline, state, label, env)
    monkeypatch.setattr(materialization, "_run", expired_copy)
    with pytest.raises(MaterializationTimeout):
        run(config, tmp_path)
    assert record_unchecked(expected_workspace(tmp_path))["status"] == "failed"


def test_unimplemented_candidate_can_be_missing_but_references_cannot(tmp_path):
    config, _ = task(tmp_path, empty=True)
    workspace = run(config, tmp_path)
    assert not (workspace / "source/kernel.py").exists()
    other = tmp_path / "other"
    config, _ = task(other, empty=True)
    (config.parent / "references/reference.py").unlink()
    with pytest.raises(TaskConfigError):
        run(config, other)
    assert record_unchecked(expected_workspace(other))["status"] == "failed"


def test_escape_symlink_is_rejected_after_setup(tmp_path):
    outside = write(tmp_path, "outside.txt", "outside")
    code = "from pathlib import Path; p=Path('source/kernel.py'); p.unlink(); p.symlink_to(" + repr(str(outside)) + ")"
    config, _ = task(tmp_path, setup=[[sys.executable, "-c", code]])
    with pytest.raises(TaskConfigError, match="within workspace"):
        run(config, tmp_path)
    assert outside.read_text() == "outside"
    assert record_unchecked(expected_workspace(tmp_path))["status"] == "failed"


def test_internal_absolute_source_symlink_becomes_portable(tmp_path):
    image = tmp_path / "image"
    original = write(image, "actual.py", "original")
    (image / "kernel.py").symlink_to(original)
    config, _ = task(tmp_path, candidate="vendor/kernel.py", sources=[{
        "kind": "image", "image_path": str(image), "destination": "vendor"}])
    workspace = run(config, tmp_path)
    assert os.readlink(workspace / "vendor/kernel.py") == "actual.py"
    assert (workspace / "vendor/kernel.py").resolve() == workspace / "vendor/actual.py"
    create_session(config, workspace)
    write(workspace, "vendor/kernel.py", "optimized through alias")
    assert run(config, tmp_path) == workspace
    assert (workspace / "vendor/actual.py").read_text() == "optimized through alias"


def test_candidate_tree_scope_keeps_nested_files_and_resume_edits(tmp_path):
    config, value = task(tmp_path)
    value["candidate"]["editable"] = [{"path": "source", "scope": "tree"}]
    config.write_text(yaml.safe_dump(value))
    write(config.parent, "source/headers/helper.h", "header")
    workspace = run(config, tmp_path)
    create_session(config, workspace)
    write(workspace, "source/new/helper.py", "new candidate helper")
    assert run(config, tmp_path) == workspace
    assert (workspace / "source/new/helper.py").read_text() == "new candidate helper"


def test_workspace_under_task_package_is_rejected_without_creating_directories(tmp_path):
    config, _ = task(tmp_path)
    run_directory = config.parent / "never-created"
    with pytest.raises(MaterializationError, match="separate directories"):
        setup_workspace(str(config), run_directory, "stamp", LOG, task_name=TASK_ID)
    assert not run_directory.exists()


def test_v2_requires_stable_discovery_id_outside_bundled_tasks(tmp_path):
    config, _ = task(tmp_path)
    with pytest.raises(TaskConfigError, match="stable discovered task_name"):
        setup_workspace(str(config), tmp_path / "runs", "stamp", LOG)
    assert not (tmp_path / "runs").exists()


def test_legacy_workspace_path_still_works(tmp_path):
    root = tmp_path / "legacy"
    config = write(root, "config.yaml", yaml.safe_dump({"task_type": "hip2hip", "source_file_path": ["kernel.hip"]}))
    write(root, "kernel.hip", "// legacy")
    workspace = run(config, tmp_path)
    assert (workspace / "kernel.hip").read_text() == "// legacy"
    assert not materialization_state_directory(workspace).exists()


def test_image_source_cannot_contain_workspace_or_receive_setup_writes(tmp_path):
    image = tmp_path / "image"
    write(image, "kernel.py", "original")
    config, _ = task(tmp_path, candidate="vendor/kernel.py", sources=[{
        "kind": "image", "image_path": str(image), "destination": "vendor"}])
    with pytest.raises(MaterializationError, match="separate directories"):
        setup_workspace(str(config), image / "runs", "stamp", LOG, task_name=TASK_ID)
    assert not (image / "runs").exists()


def test_resume_checks_non_candidate_dependencies_from_image_source(tmp_path):
    image = tmp_path / "image"
    write(image, "kernel.py", "original")
    write(image, "deps/helper.py", "dependency")
    config, _ = task(tmp_path, candidate="vendor/kernel.py", sources=[{
        "kind": "image", "image_path": str(image), "destination": "vendor"}])
    workspace = run(config, tmp_path)
    write(workspace, "vendor/kernel.py", "optimized")
    write(workspace, "vendor/deps/helper.py", "changed dependency")
    with pytest.raises(MaterializationError, match="Protected materialized task input changed"):
        run(config, tmp_path)
    assert (workspace / "vendor/kernel.py").read_text() == "optimized"


def test_copy_fallback_preserves_nested_paths_and_executable_mode(tmp_path, monkeypatch):
    config, _ = task(tmp_path)
    source = config.parent / "source/kernel.py"
    source.chmod(0o755)
    monkeypatch.setattr(materialization.shutil, "which", lambda name: None)
    workspace = run(config, tmp_path)
    assert (workspace / "source/kernel.py").read_bytes() == source.read_bytes()
    assert (workspace / "source/kernel.py").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("change", ["config", "report"])
def test_setup_cannot_change_contract_or_produce_completion_evidence(tmp_path, change):
    name = "config.yaml" if change == "config" else "validation_report.yaml"
    code = "from pathlib import Path; p=Path(" + repr(name) + "); p.write_text(p.read_text()+'\\n# changed' if p.exists() else 'PASS')"
    config, _ = task(tmp_path, setup=[[sys.executable, "-c", code]])
    with pytest.raises(MaterializationError, match="configuration|completion reports"):
        run(config, tmp_path)
    assert record_unchecked(expected_workspace(tmp_path))["status"] == "failed"


def test_resume_checks_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AKA_SCORING_IMAGE_RUNTIME_REF", "runtime@sha256:original")
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    write(workspace, "source/kernel.py", "optimized")
    monkeypatch.setenv("AKA_SCORING_IMAGE_RUNTIME_REF", "runtime@sha256:changed")
    with pytest.raises(MaterializationError, match="runtime identity changed"):
        run(config, tmp_path)
    assert (workspace / "source/kernel.py").read_text() == "optimized"


def test_resumed_candidate_cannot_alias_immutable_reference(tmp_path):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    candidate = workspace / "source/kernel.py"
    candidate.unlink()
    candidate.symlink_to("../references/reference.py")
    with pytest.raises(MaterializationError, match="resolves to protected"):
        run(config, tmp_path)
    assert candidate.is_symlink()
    assert candidate.read_text() == "reference = 1\n"


@pytest.mark.parametrize("wait_for_child", [False, True])
def test_setup_never_leaves_background_writers(tmp_path, wait_for_child):
    child = "import time; time.sleep(5)"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
            "Path('child-pid').write_text(str(p.pid)); " + ("p.wait()" if wait_for_child else ""))
    config, _ = task(tmp_path, setup=[[sys.executable, "-c", code]], timeout=1)
    error = MaterializationTimeout if wait_for_child else MaterializationError
    with pytest.raises(error):
        run(config, tmp_path)
    workspace = expected_workspace(tmp_path)
    assert record_unchecked(workspace)["status"] == "failed"
    pid = int((workspace / "child-pid").read_text())
    # An orphan may briefly remain a zombie awaiting PID 1; it cannot write.
    proc = Path(f"/proc/{pid}/stat")
    try:
        stopped = not proc.exists() or proc.read_text().split(")", 1)[1].split()[0] == "Z"
        assert stopped
    finally:
        if proc.exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_canonical_perf_helpers_are_generated_and_protected_under_same_deadline(tmp_path, monkeypatch):
    config, _ = task(tmp_path)
    write(config.parent, "scripts/evaluate.py", "import _aka_benchmark\n")
    original = materialization._run
    deadlines = []
    labels = []
    def capture_deadline(argv, cwd, deadline, state, label, env=None):
        deadlines.append(deadline)
        labels.append(label)
        return original(argv, cwd, deadline, state, label, env)
    monkeypatch.setattr(materialization, "_run", capture_deadline)
    workspace = run(config, tmp_path)
    helper = workspace / "scripts/_aka_benchmark.py"
    canonical = Path(__file__).resolve().parents[1] / "src/tools/perf/aka_benchmark.py"
    assert helper.read_bytes() == canonical.read_bytes()
    assert "scripts/_aka_benchmark.py" in load_materialization_record(workspace)["protected_files"]
    assert "copy" in labels and "perf-helpers" in labels
    assert all(deadline is deadlines[0] for deadline in deadlines)
    assert not (config.parent / "scripts/_aka_benchmark.py").exists()


def test_missing_completed_marker_rejects_resume_and_preserves_artifacts(tmp_path):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    write(workspace, "source/kernel.py", "optimized")
    (materialization_state_directory(workspace) / "materialization.json").unlink()
    with pytest.raises(MaterializationError, match="record"):
        run(config, tmp_path)
    assert (workspace / "source/kernel.py").read_text() == "optimized"


def test_symlinked_framework_state_cannot_receive_writes(tmp_path):
    config, _ = task(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    user_artifacts = tmp_path / "artifacts"
    write(user_artifacts, "keep", "user data")
    (runs / ".task-materialization").symlink_to(user_artifacts, target_is_directory=True)
    with pytest.raises(MaterializationError, match="framework directory"):
        run(config, tmp_path)
    assert sorted(path.name for path in user_artifacts.iterdir()) == ["keep"]
    assert not expected_workspace(tmp_path).exists()


def test_resume_without_session_requires_original_materialization(tmp_path):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    assert verify_original_materialization(workspace)["original_candidate_sha256"]
    assert run(config, tmp_path) == workspace
    write(workspace, "source/kernel.py", "optimized before session creation")
    with pytest.raises(MaterializationError, match="cannot capture a new baseline"):
        run(config, tmp_path)
    with pytest.raises(MaterializationError, match="cannot capture a new baseline"):
        verify_original_materialization(workspace)
    assert (workspace / "source/kernel.py").read_text() == "optimized before session creation"


def test_new_helper_prevents_baseline_recapture_without_session(tmp_path):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    write(workspace, "source/extra.py", "new helper")
    with pytest.raises(MaterializationError, match="cannot capture a new baseline"):
        run(config, tmp_path)
    assert (workspace / "source/extra.py").read_text() == "new helper"


def test_v2_completion_delegates_to_framework_record_owner(tmp_path, monkeypatch):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    write(workspace, "task_result.yaml", "agent-written result")
    calls = []
    completed = False
    module = types.ModuleType("src.task_run")
    def task_run_is_complete(root, *, expected_task_name, agent_name):
        calls.append((root, expected_task_name, agent_name))
        return completed
    module.task_run_is_complete = task_run_is_complete
    monkeypatch.setitem(sys.modules, "src.task_run", module)
    assert not is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "codex")
    completed = True
    assert is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "codex")
    assert calls == [(workspace, TASK_ID, "codex"), (workspace, TASK_ID, "codex")]


@pytest.mark.parametrize("config_contents", [None, "task_type: hip2hip", "bad yaml: ["])
def test_v2_completion_cannot_downgrade_by_replacing_config(tmp_path, monkeypatch, config_contents):
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    write(workspace, "task_result.yaml", "fake result")
    if config_contents is None:
        (workspace / "config.yaml").unlink()
    else:
        write(workspace, "config.yaml", config_contents)
    module = types.ModuleType("src.task_run")
    module.task_run_is_complete = lambda root, **kwargs: False
    monkeypatch.setitem(sys.modules, "src.task_run", module)
    assert not is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "codex")


def test_legacy_completion_remains_report_exists(tmp_path, monkeypatch):
    workspace = expected_workspace(tmp_path)
    write(workspace, "config.yaml", "task_type: hip2hip")
    write(workspace, "task_result.yaml", "legacy result")
    module = types.ModuleType("src.task_run")
    def unexpected(*args, **kwargs):
        raise AssertionError("Legacy completion must not invoke v2 completion")
    module.task_run_is_complete = unexpected
    monkeypatch.setitem(sys.modules, "src.task_run", module)
    assert is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "codex")


def test_v2_validator_keeps_existing_finalization_contract(tmp_path, monkeypatch):
    from agents.task_validator import report_schema
    config, _ = task(tmp_path)
    workspace = run(config, tmp_path)
    finalized = False
    calls = []
    def validation_report_is_complete(root):
        calls.append(root)
        return finalized
    monkeypatch.setattr(report_schema, "validation_report_is_complete", validation_report_is_complete)
    assert not is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "task_validator")
    finalized = True
    assert is_task_complete(tmp_path / "runs", TASK_ID, "stamp", "task_validator")
    assert calls == [workspace, workspace]
