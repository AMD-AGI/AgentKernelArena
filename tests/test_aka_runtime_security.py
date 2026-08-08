import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src import aka_runtime
from src import campaign


GIT = "/usr/bin/git"


def _run(root: Path, *argv: str) -> None:
    subprocess.run([GIT, *argv], cwd=root, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _run(root, "init", "-q")
    _run(root, "config", "user.name", "AKA Test")
    _run(root, "config", "user.email", "aka@example.invalid")
    (root / "main.py").write_text("print('exact')\n", encoding="utf-8")
    policy = root / "policy.yaml"
    policy.write_text("formal: true\n", encoding="utf-8")
    _run(root, "add", "main.py", "policy.yaml")
    _run(root, "commit", "-qm", "fixture")
    return root


def test_execution_manifest_covers_every_tracked_file_and_materializes(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manifest = aka_runtime.capture_execution_manifest(root)

    assert [item["path"] for item in manifest["files"]] == [
        "main.py",
        "policy.yaml",
    ]
    assert manifest["source"]["file_count"] == 2
    assert manifest["source"]["worktree_filters"] == "not_invoked"
    assert aka_runtime.verify_execution_manifest(root, manifest) == manifest

    destination = tmp_path / manifest["manifest_sha256"]
    aka_runtime.materialize_execution_snapshot(root, manifest, destination)
    assert aka_runtime.verify_materialized_snapshot(destination, manifest) == manifest


def test_git_environment_is_sanitized_and_alternate_index_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "attacker-config"))
    (tmp_path / "attacker-config").write_text(
        "[filter \"evil\"]\n\tclean = /bin/false\n", encoding="utf-8"
    )

    manifest = aka_runtime.capture_execution_manifest(root)

    assert manifest["source"]["commit"] == subprocess.run(
        [GIT, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_shortcuts_fail_closed(tmp_path: Path, flag: str) -> None:
    root = _repository(tmp_path)
    _run(root, "update-index", flag, "main.py")

    with pytest.raises(aka_runtime.AkaRuntimeError, match="index|skip-worktree"):
        aka_runtime.capture_execution_manifest(root)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("core.fsmonitor", "/tmp/attacker-fsmonitor"),
        ("filter.attack.clean", "/bin/true"),
        ("include.path", "/tmp/attacker-config"),
    ],
)
def test_repository_git_bypass_config_fails_closed(
    tmp_path: Path, key: str, value: str
) -> None:
    root = _repository(tmp_path)
    _run(root, "config", "--local", key, value)

    with pytest.raises(aka_runtime.AkaRuntimeError, match="bypass config"):
        aka_runtime.capture_execution_manifest(root)


def test_tracked_byte_and_mode_mutations_fail_without_git_filters(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "main.py").write_text("print('mutated')\n", encoding="utf-8")
    with pytest.raises(aka_runtime.AkaRuntimeError, match="bytes differ"):
        aka_runtime.capture_execution_manifest(root)

    _run(root, "restore", "main.py")
    (root / "main.py").chmod(0o755)
    with pytest.raises(aka_runtime.AkaRuntimeError, match="mode differs"):
        aka_runtime.capture_execution_manifest(root)


def test_backend_closure_detects_nested_dependency_mutation(tmp_path: Path) -> None:
    package = tmp_path / "tool"
    dependency = package / "node_modules" / "dependency"
    dependency.mkdir(parents=True)
    launcher = package / "tool.js"
    launcher.write_text("#!/usr/bin/env node\nconsole.log('tool')\n", encoding="utf-8")
    launcher.chmod(0o755)
    (package / "package.json").write_text(
        json.dumps({"name": "tool", "dependencies": {"dependency": "1.0.0"}}),
        encoding="utf-8",
    )
    (dependency / "package.json").write_text(
        json.dumps({"name": "dependency", "version": "1.0.0"}),
        encoding="utf-8",
    )
    implementation = dependency / "index.js"
    implementation.write_text("module.exports = 1\n", encoding="utf-8")
    closure = aka_runtime.capture_backend_closure("codex", str(launcher))

    assert len(closure["components"]) == 2
    assert aka_runtime.verify_backend_closure(closure) == closure
    implementation.write_text("module.exports = 2\n", encoding="utf-8")
    with pytest.raises(aka_runtime.AkaRuntimeError, match="changed after capture"):
        aka_runtime.verify_backend_closure(closure)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_complete_v5_marker_deletion_cannot_create_a_live_formal_v4_run(
    tmp_path: Path,
) -> None:
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v4",
        "objective_policy_id": "aka.task-package-objective-and-protected-harness/v1",
        "prompt_policy_id": "aka.shared-objective-backend-native-context-receipted/v1",
        "candidate_persistence_policy_id": campaign.CANDIDATE_PERSISTENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
    }
    manifest = {
        "schema": "aka.matched-campaign/v1",
        "agent": {
            "template": "apex",
            "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v4",
        },
        "comparison_contract": comparison,
        "comparison_contract_sha256": _digest(comparison),
    }
    path = tmp_path / "campaign_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(campaign.CampaignError, match="comparison contract"):
        campaign._load_verified_campaign_manifest(tmp_path)
    historical = campaign.load_historical_campaign_manifest(tmp_path)
    assert historical == {
        "classification": "historical_non_scoreable",
        "scoreable": False,
        "comparison_generation": 4,
        "manifest": manifest,
    }
    assert campaign._expected_comparison_contract_sha256(tmp_path) is None
    assert campaign._expected_session_receipt_schema(tmp_path) is None


def _mount_observation(path: Path) -> dict:
    return {
        "path": str(path),
        "mount_id": 42,
        "parent_id": 1,
        "major_minor": "0:42",
        "root": "/",
        "mount_point": str(path),
        "mount_options": ["ro", "nosuid", "nodev"],
        "filesystem_type": "fuse.squashfuse",
        "source": "squashfuse",
        "super_options": ["ro"],
        "read_only": True,
        "nested_mounts": [],
    }


def test_mount_receipt_binds_sealed_squashfs_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "a" * 64
    observation = _mount_observation(tmp_path)
    monkeypatch.setattr(aka_runtime, "_current_snapshot_mount", lambda _root: observation)
    material = {
        "schema": aka_runtime.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "policy_id": aka_runtime.IMMUTABLE_MOUNT_POLICY,
        "manifest_sha256": digest,
        "image_sha256": "b" * 64,
        "memfd_seals": [
            "F_SEAL_WRITE",
            "F_SEAL_SHRINK",
            "F_SEAL_GROW",
            "F_SEAL_SEAL",
        ],
        "mount": observation,
    }
    receipt = {**material, "sha256": _digest(material)}

    assert aka_runtime.validate_immutable_mount_receipt(
        receipt, digest, tmp_path
    ) == receipt
    receipt["mount"]["read_only"] = False
    with pytest.raises(aka_runtime.AkaRuntimeError, match="invalid"):
        aka_runtime.validate_immutable_mount_receipt(receipt, digest, tmp_path)


def test_campaign_runtime_environment_revalidates_mounted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _repository(tmp_path)
    manifest = aka_runtime.capture_execution_manifest(source)
    runtime_root = tmp_path / manifest["manifest_sha256"]
    aka_runtime.materialize_execution_snapshot(source, manifest, runtime_root)
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    observation = _mount_observation(runtime_root)
    monkeypatch.setattr(aka_runtime, "_current_snapshot_mount", lambda _root: observation)
    receipt_material = {
        "schema": aka_runtime.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "policy_id": aka_runtime.IMMUTABLE_MOUNT_POLICY,
        "manifest_sha256": manifest["manifest_sha256"],
        "image_sha256": "b" * 64,
        "memfd_seals": [
            "F_SEAL_WRITE",
            "F_SEAL_SHRINK",
            "F_SEAL_GROW",
            "F_SEAL_SEAL",
        ],
        "mount": observation,
    }
    receipt = {**receipt_material, "sha256": _digest(receipt_material)}
    receipt_path = tmp_path / "mount-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    environment = {
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_ROOT": str(runtime_root),
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST": str(manifest_path),
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST_SHA256": manifest[
            "manifest_sha256"
        ],
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST_FILE_SHA256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT": str(receipt_path),
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT_SHA256": receipt["sha256"],
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT_FILE_SHA256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    state, runtime = campaign._aka_state_from_environment(runtime_root)
    assert state["execution_manifest_sha256"] == manifest["manifest_sha256"]
    assert runtime["mount_receipt_sha256"] == receipt["sha256"]

    (runtime_root / "main.py").write_text("print('attacker')\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="attestation is invalid"):
        campaign._aka_state_from_environment(runtime_root)
