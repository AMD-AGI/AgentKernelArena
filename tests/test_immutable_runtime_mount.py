from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src import immutable_runtime_mount as immutable


def _inventory(entries: list[dict[str, object]]) -> dict[str, object]:
    material = {
        "schema": immutable.IMAGE_INPUT_SCHEMA,
        "policy_id": immutable.IMAGE_INPUT_POLICY_ID,
        "runtime_manifest_sha256": "a" * 64,
        "entries": entries,
        "entries_sha256": immutable._canonical_digest(entries),
        "normalization": {
            "uid": 0,
            "gid": 0,
            "mtime_epoch": 0,
            "xattrs": "none",
            "ordering": "utf8_posix_path_ascending",
            "format": "squashfs",
            "compression": "caller_pinned_and_receipted",
        },
    }
    return {**material, "sha256": immutable._canonical_digest(material)}


def _runtime_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    staging = tmp_path / "staging"
    mountpoint = tmp_path / ("a" * 64)
    data = staging / "data"
    staging.mkdir()
    mountpoint.mkdir()
    data.mkdir()
    payload = b"immutable-runtime\n"
    executable = b"#!/bin/sh\nexit 0\n"
    (data / "payload.bin").write_bytes(payload)
    (staging / "runner").write_bytes(executable)
    (staging / "alias").symlink_to("data/payload.bin")
    for path, mode in (
        (data / "payload.bin", 0o444),
        (staging / "runner", 0o555),
        (data, 0o555),
        (staging, 0o555),
    ):
        path.chmod(mode)
    entries: list[dict[str, object]] = [
        {"path": ".", "type": "directory", "mode": 0o555},
        {
            "path": "alias",
            "type": "symlink",
            "mode": 0o777,
            "target": "data/payload.bin",
            "target_sha256": hashlib.sha256(b"data/payload.bin").hexdigest(),
        },
        {"path": "data", "type": "directory", "mode": 0o555},
        {
            "path": "data/payload.bin",
            "type": "file",
            "mode": 0o444,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        {
            "path": "runner",
            "type": "file",
            "mode": 0o555,
            "size": len(executable),
            "sha256": hashlib.sha256(executable).hexdigest(),
        },
    ]
    return staging, mountpoint, _inventory(entries)


def _make_staging_removable(staging: Path) -> None:
    for path in sorted(staging.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
        elif not path.is_symlink():
            path.chmod(0o644)
    staging.chmod(0o755)


def test_real_mount_is_deterministic_sealed_verified_and_exactly_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    marker = tmp_path / "path-tool-ran"
    fake_bin.mkdir()
    for name in ("mksquashfs", "squashfuse", "fusermount3"):
        tool = fake_bin / name
        tool.write_text(
            f"#!/bin/sh\nprintf bad > {marker}\nexit 90\n", encoding="utf-8"
        )
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    entries = immutable.validate_image_inventory(inventory)
    try:
        first, _first_tool = immutable._build_image(staging, entries)
        os.utime(staging / "runner", (1_700_000_000, 1_700_000_000))
        second, _second_tool = immutable._build_image(staging, entries)
        assert first == second
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()

        try:
            handle = immutable.mount_immutable_runtime(
                staging, mountpoint, inventory, timeout_seconds=5
            )
        except immutable.ImmutableRuntimeMountError as error:
            if "permission denied" in str(error).lower() or "/dev/fuse" in str(error):
                pytest.skip(f"FUSE is unavailable in this environment: {error}")
            raise
        with handle:
            assert handle.receipt["schema"] == immutable.MOUNT_RECEIPT_SCHEMA
            assert handle.receipt["policy_id"] == immutable.MOUNT_POLICY_ID
            assert handle.receipt["backing"] == {
                "kind": "sealed_memfd",
                "seals": list(immutable._SEAL_NAMES),
            }
            assert handle.receipt["mount"]["filesystem"] == "fuse.squashfuse"
            assert handle.receipt["mount"]["read_only"] is True
            assert {"ro", "nodev", "nosuid"}.issubset(
                handle.receipt["mount"]["mount_options"]
            )
            assert "noexec" not in handle.receipt["mount"]["mount_options"]
            assert mountpoint.name == inventory["runtime_manifest_sha256"]
            assert fcntl.fcntl(handle.image_fd, fcntl.F_GET_SEALS) == (
                immutable._SEAL_MASK
            )
            assert (mountpoint / "data/payload.bin").read_bytes() == (
                b"immutable-runtime\n"
            )
            assert os.readlink(mountpoint / "alias") == "data/payload.bin"
            subprocess.run(
                [str(mountpoint / "runner")],
                check=True,
                capture_output=True,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            assert handle.evidence["tools"]["mksquashfs"]["source"][
                "canonical_path"
            ] == "/usr/bin/mksquashfs"
            assert handle.evidence["tools"]["squashfuse"]["source"][
                "canonical_path"
            ] == "/usr/bin/squashfuse"
            assert handle.evidence["tools"]["squashfuse"]["sealed_exec"][
                "transport"
            ] == "sealed_memfd_proc_self_fd"
            cmdline = Path(f"/proc/{handle.process.pid}/cmdline").read_bytes().split(
                b"\x00"
            )
            assert cmdline[0].startswith(b"/proc/self/fd/")
            assert cmdline[-3].startswith(b"/proc/self/fd/")
            assert cmdline[-2].startswith(b"/proc/self/fd/")
            assert not marker.exists()
        cleanup = handle.cleanup_evidence
        assert cleanup is not None
        assert cleanup["mount_absent"] is True
        assert cleanup["mountpoint_empty"] is True
        assert cleanup["image_memfd_closed"] is True
        assert cleanup["forced_process_stop"] is False
        assert handle.close() == cleanup
        assert not marker.exists()
    finally:
        _make_staging_removable(staging)


@pytest.mark.parametrize("attack", ["bytes", "extra", "inventory_digest"])
def test_preverified_staging_and_inventory_fail_closed(
    tmp_path: Path, attack: str
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    try:
        if attack == "bytes":
            (staging / "data").chmod(0o755)
            (staging / "data/payload.bin").chmod(0o644)
            (staging / "data/payload.bin").write_bytes(b"substituted\n")
            (staging / "data/payload.bin").chmod(0o444)
            (staging / "data").chmod(0o555)
        elif attack == "extra":
            staging.chmod(0o755)
            (staging / "undeclared").write_bytes(b"bad")
            staging.chmod(0o555)
        else:
            inventory["sha256"] = "0" * 64
        with pytest.raises(immutable.ImmutableRuntimeMountError):
            immutable.mount_immutable_runtime(staging, mountpoint, inventory)
        assert not immutable._mountinfo(mountpoint)[0]
    finally:
        _make_staging_removable(staging)


def test_tool_pin_rejects_substituted_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    substituted = tmp_path / "squashfuse"
    substituted.write_bytes(Path("/usr/bin/squashfuse").read_bytes() + b"tamper")
    substituted.chmod(0o755)
    monkeypatch.setattr(immutable, "_SQUASHFUSE", substituted)
    with pytest.raises(immutable.ImmutableRuntimeMountError, match="identity changed"):
        immutable._sealed_tool(
            substituted,
            expected_sha256=immutable._SQUASHFUSE_SHA256,
            expected_size=immutable._SQUASHFUSE_SIZE,
        )


def test_sealed_squashfuse_executable_has_exact_bytes_and_four_seals() -> None:
    descriptor, identity = immutable._sealed_tool(
        immutable._SQUASHFUSE,
        expected_sha256=immutable._SQUASHFUSE_SHA256,
        expected_size=immutable._SQUASHFUSE_SIZE,
    )
    try:
        payload = os.pread(descriptor, immutable._SQUASHFUSE_SIZE + 1, 0)
        assert len(payload) == immutable._SQUASHFUSE_SIZE
        assert hashlib.sha256(payload).hexdigest() == immutable._SQUASHFUSE_SHA256
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == immutable._SEAL_MASK
        assert identity["sealed_exec"]["seals"] == list(immutable._SEAL_NAMES)
    finally:
        os.close(descriptor)


def test_mountinfo_reports_nested_mounts_for_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    payload = (
        f"10 1 0:10 / {root} ro,nosuid,nodev,noexec - "
        "fuse.squashfuse squashfuse ro\n"
        f"11 10 0:11 / {nested} rw - tmpfs tmpfs rw\n"
    )
    original = Path.read_text

    def fake_read_text(path: Path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            return payload
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    mount, nested_mounts = immutable._mountinfo(root)

    assert mount is not None
    assert mount["read_only"] is True
    assert nested_mounts == [str(nested)]


def test_mounted_inventory_verifier_rejects_content_substitution(
    tmp_path: Path,
) -> None:
    staging, _mountpoint, inventory = _runtime_tree(tmp_path)
    entries = immutable.validate_image_inventory(inventory)
    try:
        (staging / "data").chmod(0o755)
        (staging / "data/payload.bin").chmod(0o644)
        (staging / "data/payload.bin").write_bytes(b"same-size-substitute")
        (staging / "data/payload.bin").chmod(0o444)
        (staging / "data").chmod(0o555)
        with pytest.raises(immutable.ImmutableRuntimeMountError, match="file changed"):
            immutable._verify_tree(staging, entries, mounted=False)
    finally:
        _make_staging_removable(staging)


def test_post_mount_verification_failure_unmounts_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    original = immutable._verify_tree

    def fail_mounted(root, entries, *, mounted):
        if mounted:
            raise immutable.ImmutableRuntimeMountError("adversarial mounted bytes")
        return original(root, entries, mounted=mounted)

    monkeypatch.setattr(immutable, "_verify_tree", fail_mounted)
    try:
        with pytest.raises(
            immutable.ImmutableRuntimeMountError, match="adversarial mounted bytes"
        ):
            immutable.mount_immutable_runtime(
                staging, mountpoint, inventory, timeout_seconds=5
            )
        mount, nested = immutable._mountinfo(mountpoint)
        assert mount is None
        assert nested == []
        assert list(mountpoint.iterdir()) == []
    finally:
        _make_staging_removable(staging)


def _inventory_file(tmp_path: Path, inventory: dict[str, object]) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _skip_if_fuse_unavailable(detail: str) -> None:
    lowered = detail.lower()
    if "permission denied" in lowered or "/dev/fuse" in lowered:
        pytest.skip(f"FUSE is unavailable in this environment: {detail}")


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_mount_service_publishes_ready_and_signal_cleans_exactly(
    tmp_path: Path, stop_signal: signal.Signals
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    inventory_path = _inventory_file(tmp_path, inventory)
    ready_path = tmp_path / "service-ready.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(immutable.__file__).resolve()),
            "serve",
            "--staging-root",
            str(staging),
            "--inventory-json",
            str(inventory_path),
            "--mountpoint",
            str(mountpoint),
            "--ready-json",
            str(ready_path),
            "--timeout-seconds",
            "5",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                pytest.fail(f"mount service readiness timed out: {stdout} {stderr}")
            time.sleep(0.05)
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            _skip_if_fuse_unavailable(stderr)
            pytest.fail(
                f"mount service failed before readiness: {stdout} {stderr}"
            )
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        material = dict(ready)
        digest = material.pop("sha256")
        assert digest == immutable._canonical_digest(material)
        assert ready["schema"] == immutable.SERVICE_READY_SCHEMA
        assert ready["policy_id"] == immutable.SERVICE_POLICY_ID
        assert ready["service"]["pid"] == process.pid
        assert ready["mount_receipt"]["sha256"] == ready["engine_evidence"][
            "receipt_sha256"
        ]
        assert ready["mount_receipt"]["root"] == str(mountpoint)
        mount, nested = immutable._mountinfo(mountpoint)
        assert mount == ready["mount_receipt"]["mount"]
        assert nested == []
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, f"{stdout} {stderr}"
        assert not ready_path.exists()
        mount, nested = immutable._mountinfo(mountpoint)
        assert mount is None
        assert nested == []
        assert list(mountpoint.iterdir()) == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        _make_staging_removable(staging)


def test_mount_service_publication_failure_unmounts_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    ready_path = tmp_path / "service-ready.json"

    def fail_publication(_target, _evidence):
        raise immutable.ImmutableRuntimeMountError("injected ready failure")

    monkeypatch.setattr(immutable._ReadyTarget, "publish", fail_publication)
    try:
        with pytest.raises(immutable.ImmutableRuntimeMountError) as raised:
            immutable.serve_immutable_runtime(
                staging,
                mountpoint,
                inventory,
                ready_path,
                timeout_seconds=5,
            )
        _skip_if_fuse_unavailable(str(raised.value))
        assert "injected ready failure" in str(raised.value)
        assert not ready_path.exists()
        mount, nested = immutable._mountinfo(mountpoint)
        assert mount is None
        assert nested == []
        assert list(mountpoint.iterdir()) == []
    finally:
        _make_staging_removable(staging)


@pytest.mark.parametrize("kind", ["existing", "symlink", "relative"])
def test_mount_service_rejects_unsafe_ready_target_without_mounting(
    tmp_path: Path, kind: str
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    ready_path: Path
    if kind == "relative":
        ready_path = Path("relative-ready.json")
    else:
        ready_path = tmp_path / "service-ready.json"
        if kind == "existing":
            ready_path.write_text("do-not-overwrite\n", encoding="utf-8")
        else:
            ready_path.symlink_to(tmp_path / "missing-target")
    try:
        with pytest.raises(immutable.ImmutableRuntimeMountError):
            immutable.serve_immutable_runtime(
                staging,
                mountpoint,
                inventory,
                ready_path,
                timeout_seconds=5,
            )
        mount, nested = immutable._mountinfo(mountpoint)
        assert mount is None
        assert nested == []
        if kind == "existing":
            assert ready_path.read_text(encoding="utf-8") == "do-not-overwrite\n"
        elif kind == "symlink":
            assert ready_path.is_symlink()
    finally:
        _make_staging_removable(staging)


def test_ready_publication_never_overwrites_a_racing_target(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "service-ready.json"
    target = immutable._ReadyTarget.open(ready_path)
    try:
        ready_path.write_text("racing-owner\n", encoding="utf-8")
        with pytest.raises(
            immutable.ImmutableRuntimeMountError, match="already exists"
        ):
            target.publish({"schema": "test-ready/v1"})
        assert ready_path.read_text(encoding="utf-8") == "racing-owner\n"
        assert not list(tmp_path.glob(".service-ready.json.tmp-*"))
    finally:
        target.close()
