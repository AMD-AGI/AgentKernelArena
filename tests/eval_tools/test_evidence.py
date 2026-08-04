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
