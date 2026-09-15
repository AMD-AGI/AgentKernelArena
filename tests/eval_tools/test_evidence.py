from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval_tools.evidence import (
    capture_submission_evidence,
    declared_submission_paths,
    load_submission_evidence,
)


def test_declared_submission_paths_prefers_explicit_profile() -> None:
    config = {
        "source_file_path": ["ignored.py"],
        "evaluation_profile": {"submission_paths": ["kernel.py", "helper.py"]},
    }
    assert declared_submission_paths(config) == ("helper.py", "kernel.py")


def test_capture_tracks_existing_and_missing_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("original = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {
            "evaluation_profile": {
                "submission_paths": ["kernel.py", "generated.py"]
            }
        },
        tmp_path / "evidence",
    )

    assert evidence.manifest["entries"][0]["exists"] is False
    assert evidence.manifest["entries"][1]["exists"] is True
    original_fingerprint = evidence.candidate_fingerprint()
    (workspace / "kernel.py").write_text("optimized = 2\n", encoding="utf-8")
    (workspace / "generated.py").write_text("new = True\n", encoding="utf-8")
    assert evidence.candidate_fingerprint() != original_fingerprint
    load_submission_evidence(evidence.storage_dir).verify()


def test_candidate_fingerprint_rejects_symlink_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "kernel.py"
    candidate.write_text("original = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )

    outside = tmp_path / "outside.py"
    outside.write_text("not_the_candidate = True\n", encoding="utf-8")
    candidate.unlink()
    candidate.symlink_to(outside)

    with pytest.raises(ValueError, match="candidate submission path escapes workspace"):
        evidence.candidate_fingerprint()


def test_candidate_fingerprint_does_not_follow_replaced_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("original = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )

    moved_workspace = tmp_path / "moved-workspace"
    workspace.rename(moved_workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kernel.py").write_text("not_the_candidate = True\n", encoding="utf-8")
    workspace.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="candidate submission path escapes workspace"):
        evidence.candidate_fingerprint()
    loaded = load_submission_evidence(evidence.storage_dir)
    with pytest.raises(ValueError, match="candidate submission path escapes workspace"):
        loaded.candidate_fingerprint()


def test_candidate_fingerprint_allows_symlink_within_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "kernel.py"
    candidate.write_text("original = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )
    original_fingerprint = evidence.candidate_fingerprint()

    replacement = workspace / "optimized.py"
    replacement.write_text("optimized = 2\n", encoding="utf-8")
    candidate.unlink()
    candidate.symlink_to(replacement.name)

    assert evidence.candidate_fingerprint() != original_fingerprint


def test_candidate_fingerprint_detects_retargeted_declared_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("a = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("b = 2\n", encoding="utf-8")
    declared = workspace / "kernel.py"
    declared.symlink_to("a.py")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )
    original_fingerprint = evidence.candidate_fingerprint()

    assert evidence.manifest["entries"][0]["workspace_relative_path"] == "kernel.py"
    assert evidence.manifest["entries"][0]["resolved_workspace_relative_path"] == "a.py"
    declared.unlink()
    declared.symlink_to("b.py")

    assert evidence.candidate_fingerprint() != original_fingerprint


def test_capture_resolves_image_repository_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "aiter" / "aiter" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {
            "image_repo_path": "/sgl-workspace/aiter",
            "source_file_path": ["aiter/kernel.py"],
        },
        tmp_path / "evidence",
    )
    assert evidence.manifest["entries"][0]["workspace_relative_path"] == (
        "aiter/aiter/kernel.py"
    )


def test_evidence_detects_tampering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )
    stored = evidence.files_dir / "kernel.py"
    stored.write_text("tampered = True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after capture"):
        evidence.verify()


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    evidence = capture_submission_evidence(
        workspace,
        {"source_file_path": ["kernel.py"]},
        tmp_path / "evidence",
    )
    manifest_path = evidence.storage_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["size"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest changed"):
        evidence.verify()


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/absolute.py"])
def test_submission_paths_reject_escape(path: str) -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        declared_submission_paths(
            {"evaluation_profile": {"submission_paths": [path]}}
        )



def _v2_config(editable=None):
    return {
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": editable or ["source/kernel.hip"]},
        "evaluation": {"runner": ["python3", "checks/driver.py"], "workloads": "inputs/cases.json"},
    }


def _write(workspace, relative, text="original"):
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_v2_explicit_paths_cannot_hide_candidate_or_protected_files(tmp_path):
    workspace = tmp_path / "workspace"
    config = _v2_config()
    config["candidate"]["entrypoints"] = [{"file": "source/kernel.hip", "kind": "function", "symbol": "launch"}]
    config["instructions"] = ["docs/operator.md"]
    config["baseline"] = {"source_files": ["baseline/reference.py"]}
    config["evaluation_profile"] = {"submission_paths": ["extra/device.hsaco"]}
    required = {"source/kernel.hip", "config.yaml", "README.md", "checks/driver.py", "inputs/cases.json",
                "docs/operator.md", "baseline/reference.py", "extra/device.hsaco", "scripts/compare.py"}
    for path in required:
        _write(workspace, path)
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    captured = {e["workspace_relative_path"] for e in evidence.manifest["entries"]}
    assert required <= captured
    initial = evidence.candidate_fingerprint()
    for path in required:
        _write(workspace, path, "changed")
        assert evidence.candidate_fingerprint() != initial, path
        _write(workspace, path)
    evidence.verify()


def test_v2_never_guesses_or_double_prefixes_repository_paths(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "vendor/aiter/kernel.hip", "real")
    _write(workspace, "aiter/vendor/aiter/kernel.hip", "decoy")
    config = _v2_config(["vendor/aiter/kernel.hip"])
    config.update(image_repo_path="/image/aiter", repo_subdir="aiter")
    config["workspace"] = {"sources": [{"kind": "image", "image_path": "/image/aiter", "destination": "vendor/aiter"}]}
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    assert (evidence.files_dir / "vendor/aiter/kernel.hip").read_text() == "real"
    assert not (evidence.files_dir / "aiter/vendor/aiter/kernel.hip").exists()


def test_v2_missing_target_does_not_fall_back_to_baseline_source(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "aiter/source/kernel.hip", "baseline")
    config = _v2_config()
    config.update(image_repo_path="/image/aiter", repo_subdir="aiter")
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    entry = next(e for e in evidence.manifest["entries"] if e["workspace_relative_path"] == "source/kernel.hip")
    assert not entry["exists"]
    original = evidence.candidate_fingerprint()
    _write(workspace, "source/kernel.hip", "candidate")
    assert evidence.candidate_fingerprint() != original


@pytest.mark.parametrize("editable", [
    ["source/kernel.hip"],
    [{"path": "source/kernel.hip", "scope": "file"}],
    [{"path": "source/kernel.hip", "scope": "symbols", "symbols": ["launch"], "allow_new_helpers": True}],
    [{"path": "source", "scope": "tree"}],
])
def test_v2_whole_file_evidence_preserves_colocated_harness(editable, tmp_path):
    workspace = tmp_path / "workspace"
    kernel = _write(workspace, "source/kernel.hip", "kernel and protected harness")
    evidence = capture_submission_evidence(workspace, _v2_config(editable), tmp_path / "evidence")
    assert (evidence.files_dir / "source/kernel.hip").read_text() == kernel.read_text()
    original = evidence.candidate_fingerprint()
    kernel.write_text("kernel and changed harness")
    assert evidence.candidate_fingerprint() != original


def test_tree_scope_reenumerates_nested_additions_and_deletions_on_resume(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/deep/kernel.hip")
    evidence = capture_submission_evidence(workspace, _v2_config([{"path": "source", "scope": "tree"}]), tmp_path / "evidence")
    original = evidence.candidate_fingerprint()
    loaded = load_submission_evidence(evidence.storage_dir)
    added = _write(workspace, "source/new/helpers/device.h", "new helper")
    assert loaded.candidate_fingerprint() != original
    added.unlink()
    added.parent.rmdir()
    added.parent.parent.rmdir()
    assert loaded.candidate_fingerprint() == original
    (workspace / "source/deep/kernel.hip").unlink()
    assert loaded.candidate_fingerprint() != original
    evidence.verify()


def test_initially_missing_tree_tracks_future_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = capture_submission_evidence(workspace, _v2_config([{"path": "generated", "scope": "tree"}]), tmp_path / "evidence")
    original = evidence.candidate_fingerprint()
    kernel = _write(workspace, "generated/nested/kernel.hip")
    populated = evidence.candidate_fingerprint()
    assert populated != original
    kernel.write_text("optimized")
    assert evidence.candidate_fingerprint() != populated


def test_parent_protected_manifest_covers_indirect_helpers_and_data(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    helper = _write(workspace, "utilities/oracle.py")
    _write(workspace, "fixtures/nested/input.bin")
    evidence = capture_submission_evidence(workspace, _v2_config(), tmp_path / "evidence",
        protected_paths=["utilities/oracle.py", "fixtures"])
    original = evidence.candidate_fingerprint()
    helper.write_text("changed oracle")
    assert evidence.candidate_fingerprint() != original
    assert (evidence.files_dir / "fixtures/nested/input.bin").is_file()


@pytest.mark.parametrize("when", ["capture", "candidate"])
def test_tree_scope_rejects_descendant_symlink_escape(tmp_path, when):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    outside = _write(tmp_path, "outside/secret.hip")
    config = _v2_config([{"path": "source", "scope": "tree"}])
    if when == "capture":
        (workspace / "source/alias").symlink_to(outside.parent, target_is_directory=True)
        with pytest.raises(ValueError, match="escapes workspace"):
            capture_submission_evidence(workspace, config, tmp_path / "evidence")
        assert not (tmp_path / "evidence").exists()
    else:
        evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
        (workspace / "source/alias").symlink_to(outside.parent, target_is_directory=True)
        with pytest.raises(ValueError, match="escapes workspace"):
            evidence.candidate_fingerprint()


def test_tree_scope_tracks_contained_directory_symlink_retargeting(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "a/kernel.hip", "same bytes")
    _write(workspace, "b/kernel.hip", "same bytes")
    link = workspace / "source"
    link.symlink_to("a", target_is_directory=True)
    evidence = capture_submission_evidence(workspace, _v2_config([{"path": "source", "scope": "tree"}]), tmp_path / "evidence")
    original = evidence.candidate_fingerprint()
    link.unlink()
    link.symlink_to("b", target_is_directory=True)
    assert evidence.candidate_fingerprint() != original


def test_tree_cycles_and_special_files_are_rejected(tmp_path):
    import os

    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    link = workspace / "source/loop"
    link.symlink_to(".", target_is_directory=True)
    config = _v2_config([{"path": "source", "scope": "tree"}])
    with pytest.raises(ValueError, match="cyclic"):
        capture_submission_evidence(workspace, config, tmp_path / "evidence")
    link.unlink()
    os.mkfifo(workspace / "source/pipe")
    with pytest.raises(ValueError, match="not a regular file"):
        capture_submission_evidence(workspace, config, tmp_path / "evidence")


def test_evidence_cannot_be_stored_in_mutable_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="outside the task workspace"):
        capture_submission_evidence(workspace, _v2_config(), workspace / "evidence")


def test_loading_modified_manifest_cannot_accept_stale_fingerprint(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    evidence = capture_submission_evidence(workspace, _v2_config(), tmp_path / "evidence")
    path = evidence.storage_dir / "manifest.json"
    data = json.loads(path.read_text())
    data["roots"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        load_submission_evidence(evidence.storage_dir)



def test_legacy_schema_two_evidence_remains_loadable(tmp_path):
    import hashlib

    workspace = tmp_path / "workspace"
    kernel = _write(workspace, "kernel.py", "original")
    storage = tmp_path / "evidence"
    _write(storage, "files/kernel.py", "original")
    record = {"declared_path": "kernel.py", "workspace_relative_path": "kernel.py",
              "resolved_workspace_relative_path": "kernel.py", "symlink_target": None,
              "exists": True, "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(), "size": 8}
    body = {"schema_version": 2, "workspace": str(workspace), "entries": [record]}
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {**body, "fingerprint": digest(body)}
    (storage / "manifest.json").write_text(json.dumps(manifest))
    evidence = load_submission_evidence(storage)
    assert evidence.candidate_fingerprint() == digest([record])
    with pytest.raises(RuntimeError, match="schema-3"):
        load_submission_evidence(storage, task_config=_v2_config())
    kernel.write_text("modified")
    assert evidence.candidate_fingerprint() != digest([record])


@pytest.mark.parametrize("value", ["", ".", "../out", "/tmp/out", "bad\x00path"])
def test_v2_declared_paths_reject_invalid_roots(value):
    with pytest.raises(ValueError, match="workspace-relative"):
        declared_submission_paths(_v2_config([value]))


def test_capture_protects_configured_commands_and_nested_tool_inputs(tmp_path):
    workspace = tmp_path / "workspace"
    config = _v2_config()
    config["evaluation"]["candidate"] = {"compile": {"commands": [["python3", "custom/build.py"]]}}
    config["workspace"] = {"setup": [["bash", "custom/setup.sh"]]}
    config["exports"] = [{"format": "solution", "output": "output.json", "command": ["python3", "custom/export.py"]}]
    config["evaluation_tools"] = {"tools": {"rocjitsu": {"options": {
        "capsule": "tool_inputs/capsule.json", "command": ["python3", "custom/check.py", "--input=tool_inputs/data.bin"]}}}}
    paths = {"source/kernel.hip", "custom/build.py", "custom/setup.sh", "custom/export.py",
             "custom/check.py", "tool_inputs/capsule.json", "tool_inputs/data.bin"}
    for path in paths:
        _write(workspace, path)
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    assert all((evidence.files_dir / path).is_file() for path in paths)



def test_shared_task_spec_mapping_drives_profile_capture_and_tool_selection(tmp_path):
    from src.task_spec import TaskSpec
    from src.eval_tools.config import EvalToolsConfig, merge_task_tool_config
    from src.eval_tools.task_profile import resolve_task_profile

    config = _v2_config([{"path": "vendor/kernel.py", "scope": "symbols", "symbols": ["compute"], "allow_new_helpers": True}])
    config["candidate"]["language"] = "triton"
    config["candidate"]["entrypoints"] = [{"file": "vendor/kernel.py", "kind": "function", "symbol": "compute"}]
    config["workspace"] = {"sources": [{"kind": "image", "image_path": "/opt/vendor", "destination": "vendor"}]}
    config["evaluation_tools"] = {"tools": {"gpu_asan": {"options": {"command": ["python3", "checks/memory.py"]}}}}
    spec = TaskSpec.from_mapping(config, task_id="examples/independent-task")
    normalized = spec.to_mapping()
    profile = resolve_task_profile(normalized)
    assert profile.language.value == "triton"
    assert profile.source_files == ("vendor/kernel.py",)
    assert profile.target_functions == ("compute",)
    assert profile.evidence["candidate"]["initial_state"] == "implemented"
    assert merge_task_tool_config(EvalToolsConfig.disabled(), normalized).enabled == ()
    workspace = tmp_path / "workspace"
    for path in ("vendor/kernel.py", "checks/driver.py", "checks/memory.py"):
        _write(workspace, path)
    evidence = capture_submission_evidence(workspace, normalized, tmp_path / "evidence")
    assert (evidence.files_dir / "vendor/kernel.py").is_file()
    assert (evidence.files_dir / "checks/memory.py").is_file()



def test_resume_checks_current_v2_candidate_and_protected_coverage(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    config = _v2_config()
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    load_submission_evidence(evidence.storage_dir, task_config=config)
    with pytest.raises(RuntimeError, match="does not cover"):
        load_submission_evidence(evidence.storage_dir, task_config=config, protected_paths=["custom/oracle.py"])
    changed = _v2_config(["new/kernel.hip"])
    with pytest.raises(RuntimeError, match="does not cover"):
        load_submission_evidence(evidence.storage_dir, task_config=changed)


def test_resume_tree_covers_newly_enumerated_protected_descendants(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    config = _v2_config()
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence", protected_paths=["custom"])
    _write(workspace, "custom/oracle.py")
    load_submission_evidence(evidence.storage_dir, task_config=config, protected_paths=["custom/oracle.py"])



def test_inline_command_code_is_not_mistaken_for_a_submission_path(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace, "source/kernel.hip")
    config = _v2_config()
    config["evaluation"]["runner"] = ["python3", "-c", "import os; print(os.path.normpath('../source/kernel.hip'))"]
    evidence = capture_submission_evidence(workspace, config, tmp_path / "evidence")
    assert (evidence.files_dir / "source/kernel.hip").is_file()
