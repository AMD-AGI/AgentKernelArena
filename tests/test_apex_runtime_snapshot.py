# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.apex_runtime import (
    ApexRuntimeError,
    materialize_runtime,
    plan_runtime,
    runtime_command,
    runtime_environment,
    verify_runtime_snapshot,
)


def _run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True, capture_output=True)


def _runtime_checkout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "Apex"
    root.mkdir()
    (root / "src").mkdir()
    (root / "main.py").write_text(
        "import json, pathlib, sys\n"
        "import apex_probe\n"
        "print(json.dumps({'no_site': sys.flags.no_site, 'probe': apex_probe.VALUE}))\n",
        encoding="utf-8",
    )
    (root / "src" / "apex_probe.py").write_text("VALUE = 'sealed'\n", encoding="utf-8")
    (root / ".gitignore").write_text(".venv\n", encoding="utf-8")
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


def test_snapshot_is_complete_sealed_and_executes_without_site_hooks(
    tmp_path: Path,
) -> None:
    root, python, external, marker = _runtime_checkout(tmp_path)
    plan = plan_runtime(root, python, declared_roots=[external])

    snapshot = materialize_runtime(plan, tmp_path / "attempt.runtime")
    manifest = verify_runtime_snapshot(snapshot, plan.sha256)
    command = runtime_command(snapshot, manifest, [])
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

    assert json.loads(completed.stdout) == {"no_site": 1, "probe": "sealed"}
    assert not marker.exists()
    assert (snapshot / "external/000/native.so").read_bytes() == b"native-runtime-bytes"
    assert stat.S_IMODE((snapshot / "repo/main.py").stat().st_mode) == 0o444
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o555

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
