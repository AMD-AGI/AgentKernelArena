# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src import aka_runtime
from src import apex_runtime
from src import immutable_runtime_mount
from src.apex_runtime import (
    ApexRuntimeError,
    create_immutable_mount_receipt,
    materialize_runtime,
    plan_runtime,
    runtime_command,
    runtime_environment,
    runtime_image_inputs,
    verify_runtime_snapshot,
)


def _run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True, capture_output=True)


def test_apex_runtime_cli_imports_from_its_script_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(apex_runtime.__file__).resolve()), "--help"],
        cwd="/",
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mount-receipt" in completed.stdout


def _runtime_checkout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "Apex"
    root.mkdir()
    (root / "src").mkdir()
    (root / "main.py").write_text(
        "import json, pathlib, subprocess, sys\n"
        "import apex_probe\n"
        "child = subprocess.check_output([sys.executable, '-c', "
        "'import json,sys; print(json.dumps({\"no_site\":sys.flags.no_site,' "
        "'\"executable\":sys.executable}))'], text=True)\n"
        "alias = subprocess.check_output(['python3', '-c', "
        "'import sys; print(sys.flags.no_site)'], text=True)\n"
        "print(json.dumps({'no_site': sys.flags.no_site, 'probe': apex_probe.VALUE, "
        "'child': json.loads(child), 'alias_no_site': int(alias)}))\n",
        encoding="utf-8",
    )
    (root / "src" / "apex_probe.py").write_text("VALUE = 'sealed'\n", encoding="utf-8")
    (root / ".gitignore").write_text(".venv\n.cache\n", encoding="utf-8")
    _run("/usr/bin/git", "init", "-q", cwd=root)
    _run("/usr/bin/git", "config", "user.name", "test", cwd=root)
    _run("/usr/bin/git", "config", "user.email", "test@example.invalid", cwd=root)
    _run("/usr/bin/git", "add", ".", cwd=root)
    _run("/usr/bin/git", "commit", "-qm", "fixture", cwd=root)

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv = root / ".venv"
    site = venv / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    os.symlink(sys.executable, venv / "bin" / "python")
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    external = tmp_path / "editable-runtime"
    external.mkdir()
    (external / "native.so").write_bytes(b"native-runtime-bytes")
    marker = tmp_path / "site-executed"
    (site / "editable.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('pth')\n"
        f"{external}\n",
        encoding="utf-8",
    )
    (site / "sitecustomize.py").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('site')\n",
        encoding="utf-8",
    )
    (site / "package-1.dist-info").mkdir()
    (site / "package-1.dist-info" / "RECORD").write_text(
        "package.py,,\n", encoding="utf-8"
    )
    return root, venv / "bin" / "python", external, marker


def _digest(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


_REQUESTED_MOUNT_OPTIONS = [
    "ro",
    "nodev",
    "nosuid",
    "default_permissions",
    "allow_other",
    "subtype=squashfuse",
]


def _host_access_policy(private_ancestor: Path) -> dict[str, object]:
    owner = {"uid": os.getuid(), "gid": os.getgid()}
    material = {
        "schema": aka_runtime.HOST_ACCESS_POLICY_SCHEMA,
        "policy_id": aka_runtime.HOST_ACCESS_POLICY_ID,
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "private_ancestor": {
            "path": str(private_ancestor),
            "device": 7,
            "inode": 11,
            "uid": owner["uid"],
            "gid": owner["gid"],
            "mode": 0o700,
        },
        "fuse_config": {
            "path": "/etc/fuse.conf",
            "device": 8,
            "inode": 12,
            "uid": 0,
            "gid": 0,
            "mode": 0o644,
            "nlink": 1,
            "size_bytes": 17,
            "sha256": "c" * 64,
            "user_allow_other": True,
        },
        "mount_owner": owner,
        "worker": dict(owner),
        "docker_daemon": {
            "uid": 0,
            "trusted_boundary": True,
            "access_via": "fuse_allow_other_with_private_ancestor_v1",
        },
    }
    return {**material, "sha256": _digest(material)}


def _observed_mount(snapshot: Path) -> dict[str, object]:
    return {
        "mount_id": 91,
        "device": "0:91",
        "root": "/",
        "mount_point": str(snapshot),
        "filesystem": "fuse.squashfuse",
        "mount_options": ["nodev", "nosuid", "ro"],
        "super_options": [
            "allow_other",
            "default_permissions",
            f"group_id={os.getgid()}",
            "ro",
            f"user_id={os.getuid()}",
        ],
        "read_only": True,
    }


def _runtime_service_evidence(
    snapshot: Path,
    manifest: dict[str, object],
    image_sha256: str = "f" * 64,
) -> dict[str, object]:
    host_policy = _host_access_policy(snapshot.parent)
    mount = _observed_mount(snapshot)
    image_input_sha256 = runtime_image_inputs(snapshot, manifest)["sha256"]
    receipt_material = {
        "schema": immutable_runtime_mount.MOUNT_RECEIPT_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "root": str(snapshot),
        "runtime_manifest_sha256": manifest["sha256"],
        "runtime_image_input_sha256": image_input_sha256,
        "image_sha256": image_sha256,
        "backing": {
            "kind": "sealed_memfd",
            "seals": list(immutable_runtime_mount._SEAL_NAMES),
        },
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "host_access_policy": host_policy,
        "mount": mount,
    }
    host_receipt = {**receipt_material, "sha256": _digest(receipt_material)}
    engine_material = {
        "schema": aka_runtime.ENGINE_EVIDENCE_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "receipt_sha256": host_receipt["sha256"],
        "runtime_image_input_sha256": image_input_sha256,
        "image": {
            "size_bytes": 4096,
            "sha256": image_sha256,
            "memfd_seals": list(immutable_runtime_mount._SEAL_NAMES),
        },
        "tools": {
            "mksquashfs": {"path": "/usr/bin/mksquashfs", "sha256": "e" * 64},
            "squashfuse": {"path": "/usr/bin/squashfuse", "sha256": "f" * 64},
        },
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "host_access_policy_sha256": host_policy["sha256"],
        "process": {"pid": 101, "starttime": 202, "foreground": True},
        "mountpoint_source": {
            "path": str(snapshot),
            "device": 7,
            "inode": 13,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "mode": 0o555,
        },
        "mount": mount,
        "inventory_verification": {
            "entry_count": 2,
            "inventory_sha256": image_input_sha256,
        },
        "write_probe_errno": 30,
    }
    engine_evidence = {**engine_material, "sha256": _digest(engine_material)}
    service_material = {
        "schema": aka_runtime.ENGINE_SERVICE_SCHEMA,
        "policy_id": aka_runtime.ENGINE_SERVICE_POLICY,
        "ready_path": str(snapshot.parent / "runtime-service-ready.json"),
        "service": {
            "pid": 303,
            "starttime": 404,
            "owner": dict(host_policy["mount_owner"]),
            "accepted_signals": ["SIGINT", "SIGTERM"],
            "engine_process": {"pid": 101, "starttime": 202},
        },
        "mount_receipt": host_receipt,
        "engine_evidence": engine_evidence,
    }
    return {**service_material, "sha256": _digest(service_material)}


def _immutable_receipt(
    snapshot: Path,
    manifest: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    mount = _observed_mount(snapshot)
    monkeypatch.setattr(apex_runtime, "_observed_immutable_mount", lambda _root: mount)
    service = _runtime_service_evidence(snapshot, manifest)
    material = {
        "schema": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "root": str(snapshot),
        "runtime_manifest_sha256": manifest["sha256"],
        "runtime_image_input_sha256": runtime_image_inputs(snapshot, manifest)["sha256"],
        "image_sha256": "f" * 64,
        "backing": {
            "kind": "sealed_memfd",
            "seals": list(apex_runtime._REQUIRED_MEMFD_SEALS),
        },
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "runtime_service_evidence_sha256": service["sha256"],
        "runtime_engine_evidence_sha256": service["engine_evidence"]["sha256"],
        "host_access_policy": service["mount_receipt"]["host_access_policy"],
        "mount": mount,
    }
    return {**material, "sha256": _digest(material)}


def test_snapshot_is_complete_sealed_and_executes_without_site_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, python, external, marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])

    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    receipt = _immutable_receipt(snapshot, manifest, monkeypatch)
    command = runtime_command(
        snapshot,
        manifest,
        [],
        immutable_mount_receipt=receipt,
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        **runtime_environment(snapshot, manifest),
    }
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(completed.stdout)
    assert output["no_site"] == 1
    assert output["probe"] == "sealed"
    assert output["child"]["no_site"] == 1
    assert output["child"]["executable"].endswith("/sealed-bin/python")
    assert output["alias_no_site"] == 1
    assert not marker.exists()
    assert (snapshot / "external/000/native.so").read_bytes() == b"native-runtime-bytes"
    assert stat.S_IMODE((snapshot / "repo/main.py").stat().st_mode) == 0o444
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o555
    assert (snapshot / "venv/bin/python").is_file()
    assert not (snapshot / "venv/bin/python").is_symlink()

    (root / "main.py").write_text("raise SystemExit('mutable checkout')\n")
    (external / "native.so").write_bytes(b"mutable")
    verify_runtime_snapshot(snapshot, plan.sha256)
    repeated = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(repeated.stdout)["probe"] == "sealed"


def test_execution_requires_an_immutable_mount_receipt(tmp_path: Path) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    with pytest.raises(ApexRuntimeError, match="receipt is required"):
        runtime_command(snapshot, manifest, [])


def test_mount_receipt_is_created_from_current_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    expected = _immutable_receipt(snapshot, manifest, monkeypatch)
    service = _runtime_service_evidence(snapshot, manifest)

    receipt = create_immutable_mount_receipt(
        snapshot,
        manifest,
        "f" * 64,
        service,
    )

    assert receipt == expected


def test_apex_mount_receipt_rejects_host_schema_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    receipt = _immutable_receipt(snapshot, manifest, monkeypatch)
    receipt["schema"] = immutable_runtime_mount.MOUNT_RECEIPT_SCHEMA
    receipt["sha256"] = _digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )

    with pytest.raises(ApexRuntimeError, match="invalid"):
        apex_runtime.validate_immutable_mount_receipt(snapshot, manifest, receipt)


def test_apex_mount_receipt_rejects_self_consistent_requested_option_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    receipt = _immutable_receipt(snapshot, manifest, monkeypatch)
    receipt["requested_mount_options"] = [
        option
        for option in _REQUESTED_MOUNT_OPTIONS
        if option != "default_permissions"
    ]
    receipt["sha256"] = _digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )

    with pytest.raises(ApexRuntimeError, match="invalid"):
        apex_runtime.validate_immutable_mount_receipt(snapshot, manifest, receipt)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_git_index_shortcuts_are_rejected(tmp_path: Path, flag: str) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    _run("/usr/bin/git", "update-index", flag, "main.py", cwd=root)
    with pytest.raises(ApexRuntimeError, match="index flags"):
        plan_runtime(root, python, declared_roots=[external])


def test_git_environment_is_sanitized_and_exact_bytes_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    hostile = tmp_path / "hostile.gitconfig"
    hostile.write_text("[core]\nignoreStat = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    first = plan_runtime(root, python, declared_roots=[external])
    assert first.manifest["git"]["git_environment_sanitized"] is True

    (root / "main.py").write_text("print('not HEAD')\n", encoding="utf-8")
    _run("/usr/bin/git", "update-index", "--assume-unchanged", "main.py", cwd=root)
    with pytest.raises(ApexRuntimeError):
        plan_runtime(root, python, declared_roots=[external])


def test_runtime_metadata_and_native_bytes_change_the_digest(tmp_path: Path) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    first = plan_runtime(root, python, declared_roots=[external])
    parent = tmp_path / "shared.runtime"
    first_snapshot = materialize_runtime(first, parent)
    record = next((root / ".venv").rglob("RECORD"))
    record.write_text("package.py,sha256=changed,7\n", encoding="utf-8")
    second = plan_runtime(root, python, declared_roots=[external])
    assert second.sha256 != first.sha256
    (external / "native.so").write_bytes(b"changed-native")
    third = plan_runtime(root, python, declared_roots=[external])
    assert third.sha256 != second.sha256

    third_snapshot = materialize_runtime(third, parent)
    assert first_snapshot != third_snapshot
    assert materialize_runtime(third, parent) == third_snapshot
    verify_runtime_snapshot(first_snapshot, first.sha256)
    verify_runtime_snapshot(third_snapshot, third.sha256)


def test_declared_external_roots_must_match_exactly(tmp_path: Path) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    with pytest.raises(ApexRuntimeError, match="declared Apex external roots"):
        plan_runtime(root, python, declared_roots=[])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ApexRuntimeError, match="declared Apex external roots"):
        plan_runtime(root, python, declared_roots=[external, other])


def test_snapshot_verifier_rejects_tampering(tmp_path: Path) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    target = snapshot / "repo/main.py"
    target.chmod(0o644)
    target.write_text("print('tampered')\n", encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(ApexRuntimeError, match="snapshot file changed"):
        verify_runtime_snapshot(snapshot, plan.sha256)


def test_git_filter_is_never_executed(tmp_path: Path) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    filter_marker = tmp_path / "filter-executed"
    filter_script = tmp_path / "hostile-filter"
    filter_script.write_text(
        "#!/bin/sh\nprintf executed > \"$1\"\ncat\n",
        encoding="utf-8",
    )
    filter_script.chmod(0o755)
    (root / ".gitattributes").write_text("main.py filter=hostile\n", encoding="utf-8")
    _run("/usr/bin/git", "add", ".gitattributes", cwd=root)
    _run("/usr/bin/git", "commit", "-qm", "attributes", cwd=root)
    _run(
        "/usr/bin/git",
        "config",
        "filter.hostile.smudge",
        f"{filter_script} {filter_marker}",
        cwd=root,
    )
    _run(
        "/usr/bin/git",
        "config",
        "filter.hostile.clean",
        f"{filter_script} {filter_marker}",
        cwd=root,
    )

    plan_runtime(root, python, declared_roots=[external])
    assert not filter_marker.exists()


def test_head_change_during_object_reads_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    original = apex_runtime._tracked_file_entry
    changed = False

    def racing_entry(binding: object, entry: dict[str, str]) -> dict[str, object]:
        nonlocal changed
        value = original(binding, entry)
        if not changed:
            changed = True
            _run(
                "/usr/bin/git",
                "commit",
                "--allow-empty",
                "-qm",
                "raced head",
                cwd=root,
            )
        return value

    monkeypatch.setattr(apex_runtime, "_tracked_file_entry", racing_entry)
    with pytest.raises(ApexRuntimeError, match="HEAD changed"):
        plan_runtime(root, python, declared_roots=[external])


def test_absolute_symlinks_are_rejected_but_relative_links_stay_inside(
    tmp_path: Path,
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    os.symlink("native.so", external / "relative-native.so")
    plan = plan_runtime(root, python, declared_roots=[external])
    snapshot = materialize_runtime(plan, tmp_path / "relative.runtime")
    copied = snapshot / "external/000/relative-native.so"
    assert copied.is_symlink()
    assert copied.resolve(strict=True).is_relative_to(snapshot / "external/000")

    (external / "relative-native.so").unlink()
    os.symlink("/tmp", external / "absolute")
    with pytest.raises(ApexRuntimeError, match="absolute runtime symlink"):
        plan_runtime(root, python, declared_roots=[external])


def test_managed_magpie_and_inferencex_are_in_runtime_closure(
    tmp_path: Path,
) -> None:
    root, python, external, _marker = _runtime_checkout(tmp_path)
    managed = root / ".cache/apex-dependencies"
    magpie = managed / "magpie"
    inferencex = managed / "inferencex"
    (magpie / "Magpie").mkdir(parents=True)
    (inferencex / "runners").mkdir(parents=True)
    (magpie / "Magpie/__init__.py").write_text("LOCKED = True\n", encoding="utf-8")
    (inferencex / "runners/run.py").write_text("LOCKED = True\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts/dependencies.lock.json").write_text(
        json.dumps(
            {
                "dependencies": {
                    "magpie": {"managed_checkout": "magpie"},
                    "inferencex": {"managed_checkout": "inferencex"},
                }
            }
        ),
        encoding="utf-8",
    )
    _run("/usr/bin/git", "add", "scripts/dependencies.lock.json", cwd=root)
    _run("/usr/bin/git", "commit", "-qm", "dependency lock", cwd=root)
    expected = sorted((external, magpie, inferencex), key=str)

    plan = plan_runtime(root, python, declared_roots=expected)
    assert list(plan.external_roots) == expected
    snapshot = materialize_runtime(plan, tmp_path / "managed.runtime")
    destinations = {
        item["source"]["path"]: item["destination"]
        for item in plan.manifest["roots"]
    }
    assert (
        snapshot / destinations[str(magpie)] / "Magpie/__init__.py"
    ).read_text(encoding="utf-8") == "LOCKED = True\n"
    assert (
        snapshot / destinations[str(inferencex)] / "runners/run.py"
    ).read_text(encoding="utf-8") == "LOCKED = True\n"
