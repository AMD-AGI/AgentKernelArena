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
from typing import Any

import pytest

from src import aka_runtime
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
    tmp_path.chmod(0o700)
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


def _mount_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "private_ancestor": tmp_path,
        "fuse_config": Path("/etc/fuse.conf"),
        "worker_uid": os.getuid(),
        "worker_gid": os.getgid(),
    }


def _make_staging_removable(staging: Path) -> None:
    for path in sorted(staging.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
        elif not path.is_symlink():
            path.chmod(0o644)
    staging.chmod(0o755)


def test_mksquashfs_sort_priorities_cover_large_runtime_without_overflow() -> None:
    assert [immutable._mksquashfs_sort_priority(index, 5) for index in range(5)] == [
        5,
        4,
        3,
        2,
        1,
    ]
    assert immutable._mksquashfs_sort_priority(0, 32_767) == 32_767
    assert immutable._mksquashfs_sort_priority(32_766, 32_767) == 1
    assert immutable._mksquashfs_sort_priority(0, 32_768) == 32_767
    assert immutable._mksquashfs_sort_priority(32_767, 32_768) == -32_768
    with pytest.raises(immutable.ImmutableRuntimeMountError):
        immutable._mksquashfs_sort_priority(0, 0)
    with pytest.raises(immutable.ImmutableRuntimeMountError):
        immutable._mksquashfs_sort_priority(-1, 1)
    with pytest.raises(immutable.ImmutableRuntimeMountError):
        immutable._mksquashfs_sort_priority(1, 1)

    total = 91_725
    priorities = [
        immutable._mksquashfs_sort_priority(index, total)
        for index in range(total)
    ]
    assert priorities[0] == immutable._MKSQUASHFS_MAX_SORT_PRIORITY
    assert priorities[-1] == immutable._MKSQUASHFS_MIN_SORT_PRIORITY
    assert len(set(priorities)) == 65_536
    assert all(
        left >= right for left, right in zip(priorities, priorities[1:])
    )
    assert all(
        not (first == second == third)
        for first, second, third in zip(
            priorities, priorities[1:], priorities[2:]
        )
    )
    assert all(
        immutable._MKSQUASHFS_MIN_SORT_PRIORITY
        <= priority
        <= immutable._MKSQUASHFS_MAX_SORT_PRIORITY
        for priority in priorities
    )
    assert immutable._mksquashfs_sort_path(r"#with space/back\slash") == (
        r"\#with\ space/back\\slash"
    )


def test_real_image_sort_file_accepts_escaped_safe_paths(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    names = ("#leading", "back\\slash", "with space")
    entries: dict[str, dict[str, object]] = {
        ".": {"path": ".", "type": "directory", "mode": 0o555}
    }
    for name in sorted(names, key=str.encode):
        payload = name.encode()
        path = staging / name
        path.write_bytes(payload)
        path.chmod(0o444)
        entries[name] = {
            "path": name,
            "type": "file",
            "mode": 0o444,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    staging.chmod(0o555)

    try:
        image, _tool = immutable._build_image(staging, entries)
        assert image
    finally:
        _make_staging_removable(staging)


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
                staging,
                mountpoint,
                inventory,
                timeout_seconds=5,
                **_mount_kwargs(tmp_path),
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
            assert "allow_other" in handle.receipt["mount"]["super_options"]
            assert handle.receipt["requested_mount_options"] == list(
                immutable._MOUNT_OPTIONS
            )
            policy = handle.receipt["host_access_policy"]
            assert policy["mount_owner"] == {
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
            assert policy["worker"] == policy["mount_owner"]
            assert policy["private_ancestor"]["mode"] == 0o700
            assert policy["fuse_config"]["path"] == "/etc/fuse.conf"
            assert policy["fuse_config"]["uid"] == 0
            assert policy["fuse_config"]["gid"] == 0
            assert policy["fuse_config"]["mode"] & 0o022 == 0
            assert policy["fuse_config"]["nlink"] == 1
            assert policy["fuse_config"]["user_allow_other"] is True
            assert handle.evidence["host_access_policy_sha256"] == policy[
                "sha256"
            ]
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


@pytest.mark.skipif(
    os.environ.get("AKA_RUN_DOCKER_FUSE_INTEGRATION") != "1",
    reason="requires the trusted host Docker daemon and passwordless root probe",
)
def test_real_mount_is_owner_root_and_docker_readable_but_host_private(
    tmp_path: Path,
) -> None:
    """Exercise the exact cross-principal boundary used by formal Docker runs."""

    if subprocess.run(
        ["sudo", "-n", "/usr/bin/true"],
        check=False,
        capture_output=True,
    ).returncode != 0:
        pytest.skip("passwordless sudo is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", "busybox:latest"],
        check=False,
        capture_output=True,
    ).returncode != 0:
        pytest.skip("the pinned local busybox probe image is unavailable")

    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    try:
        try:
            handle = immutable.mount_immutable_runtime(
                staging,
                mountpoint,
                inventory,
                timeout_seconds=5,
                **_mount_kwargs(tmp_path),
            )
        except immutable.ImmutableRuntimeMountError as error:
            _skip_if_fuse_unavailable(str(error))
            raise
        with handle:
            payload = mountpoint / "data/payload.bin"
            assert payload.read_bytes() == b"immutable-runtime\n"
            root_probe = subprocess.run(
                ["sudo", "-n", "/usr/bin/stat", "--", str(payload)],
                check=False,
                capture_output=True,
            )
            assert root_probe.returncode == 0, root_probe.stderr.decode(
                "utf-8", "replace"
            )
            third_user_probe = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "-u",
                    "nobody",
                    "/usr/bin/stat",
                    "--",
                    str(payload),
                ],
                check=False,
                capture_output=True,
            )
            assert third_user_probe.returncode != 0
            docker_probe = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--network=none",
                    "--read-only",
                    "--volume",
                    f"{mountpoint}:/runtime:ro",
                    "busybox:latest",
                    "cat",
                    "/runtime/data/payload.bin",
                ],
                check=False,
                capture_output=True,
                timeout=30,
            )
            assert docker_probe.returncode == 0, docker_probe.stderr.decode(
                "utf-8", "replace"
            )
            assert docker_probe.stdout == b"immutable-runtime\n"
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
            immutable.mount_immutable_runtime(
                staging, mountpoint, inventory, **_mount_kwargs(tmp_path)
            )
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


def test_mount_command_requires_daemon_readable_default_permissions() -> None:
    assert immutable._MOUNT_OPTIONS == (
        "ro",
        "nodev",
        "nosuid",
        "default_permissions",
        "allow_other",
        "subtype=squashfuse",
    )


def test_wait_for_mount_rejects_owner_only_fuse_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    process = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    owner_only = {
        "mount_id": 1,
        "device": "0:1",
        "root": "/",
        "mount_point": str(root),
        "filesystem": "fuse.squashfuse",
        "mount_options": ["nodev", "nosuid", "ro"],
        "super_options": ["ro", "user_id=1000"],
        "read_only": True,
    }
    monkeypatch.setattr(immutable, "_mountinfo", lambda _root: (owner_only, []))
    try:
        with pytest.raises(
            immutable.ImmutableRuntimeMountError,
            match="lacks the required read-only access policy",
        ):
            immutable._wait_for_mount(root, process, 1)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


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
                staging,
                mountpoint,
                inventory,
                timeout_seconds=5,
                **_mount_kwargs(tmp_path),
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
            "--private-ancestor",
            str(tmp_path),
            "--fuse-config",
            "/etc/fuse.conf",
            "--worker-uid",
            str(os.getuid()),
            "--worker-gid",
            str(os.getgid()),
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
        assert ready["service"]["owner"] == {
            "uid": os.getuid(),
            "gid": os.getgid(),
        }
        assert ready["service"]["accepted_signals"] == ["SIGINT", "SIGTERM"]
        assert ready["service"]["engine_process"] == {
            "pid": ready["engine_evidence"]["process"]["pid"],
            "starttime": ready["engine_evidence"]["process"]["starttime"],
        }
        assert ready["service"]["engine_process"]["pid"] != process.pid
        persisted_service = tmp_path / "persisted-service-evidence.json"
        persisted_service.write_bytes(ready_path.read_bytes())
        persisted_service.chmod(0o444)
        assert aka_runtime.load_runtime_service_evidence(
            persisted_service,
            file_sha256=hashlib.sha256(persisted_service.read_bytes()).hexdigest(),
            content_sha256=ready["sha256"],
            manifest_sha256=inventory["runtime_manifest_sha256"],
            image_sha256=ready["mount_receipt"]["image_sha256"],
        ) == ready
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
            process.send_signal(signal.SIGTERM)
            try:
                process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        _make_staging_removable(staging)


def test_persisted_service_evidence_recovers_sigkill_orphan_exactly(
    tmp_path: Path,
) -> None:
    staging, mountpoint, inventory = _runtime_tree(tmp_path)
    inventory_path = _inventory_file(tmp_path, inventory)
    ready_path = tmp_path / "service-ready.json"
    persisted = tmp_path / "persisted-service-evidence.json"
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
            "--private-ancestor",
            str(tmp_path),
            "--fuse-config",
            "/etc/fuse.conf",
            "--worker-uid",
            str(os.getuid()),
            "--worker-gid",
            str(os.getgid()),
            "--timeout-seconds",
            "5",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    recovered = False
    ready: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + 20
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("mount service readiness timed out")
            time.sleep(0.05)
        if process.poll() is not None:
            _stdout, stderr = process.communicate(timeout=5)
            _skip_if_fuse_unavailable(stderr)
            pytest.fail(f"mount service failed before readiness: {stderr}")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        persisted.write_bytes(ready_path.read_bytes())
        persisted.chmod(0o444)
        file_sha256 = hashlib.sha256(persisted.read_bytes()).hexdigest()
        controller_starttime = ready["service"]["starttime"]
        engine_process = ready["service"]["engine_process"]

        process.kill()
        process.communicate(timeout=5)
        assert process.returncode == -signal.SIGKILL
        cleanup = aka_runtime.recover_runtime_service(
            persisted,
            file_sha256=file_sha256,
            content_sha256=ready["sha256"],
            manifest_sha256=inventory["runtime_manifest_sha256"],
            image_sha256=ready["mount_receipt"]["image_sha256"],
            controller_pid=process.pid,
            controller_starttime=controller_starttime,
            mountpoint=mountpoint,
            private_ancestor=tmp_path,
            timeout_seconds=5,
        )
        recovered = True
        assert cleanup["schema"] == "aka.immutable-runtime-mount-recovery/v1"
        assert cleanup["controller"]["verified_exited"] is True
        assert cleanup["engine"]["verified_exited"] is True
        assert cleanup["controller"]["pid"] == process.pid
        assert cleanup["controller"]["starttime"] == controller_starttime
        assert cleanup["engine"]["pid"] == engine_process["pid"]
        assert cleanup["engine"]["starttime"] == engine_process["starttime"]
        assert cleanup["runtime_service_evidence_sha256"] == ready["sha256"]
        assert cleanup["runtime_engine_evidence_sha256"] == ready[
            "engine_evidence"
        ]["sha256"]
        assert cleanup["host_mount_receipt_sha256"] == ready["mount_receipt"][
            "sha256"
        ]
        assert cleanup["mount_absent"] is True
        assert cleanup["mountpoint_empty"] is True
        assert immutable._mountinfo(mountpoint) == (None, [])
        try:
            observed_starttime = immutable._process_starttime(engine_process["pid"])
        except immutable.ImmutableRuntimeMountError:
            observed_starttime = None
        assert observed_starttime != engine_process["starttime"]
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        if not recovered and ready is not None:
            persisted.chmod(0o444)
            try:
                aka_runtime.recover_runtime_service(
                    persisted,
                    file_sha256=hashlib.sha256(persisted.read_bytes()).hexdigest(),
                    content_sha256=ready["sha256"],
                    manifest_sha256=inventory["runtime_manifest_sha256"],
                    image_sha256=ready["mount_receipt"]["image_sha256"],
                    controller_pid=process.pid,
                    controller_starttime=ready["service"]["starttime"],
                    mountpoint=mountpoint,
                    private_ancestor=tmp_path,
                    timeout_seconds=5,
                )
            except (aka_runtime.AkaRuntimeError, OSError):
                pass
        _make_staging_removable(staging)


def test_exact_pidfd_accepts_exit_between_open_and_starttime_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, writable = os.pipe()

    def exited_during_identity_read(_pid: int) -> int:
        raise immutable.ImmutableRuntimeMountError("process exited")

    monkeypatch.setattr(os, "pidfd_open", lambda _pid, _flags: descriptor)
    monkeypatch.setattr(immutable, "_process_starttime", exited_during_identity_read)
    monkeypatch.setattr(immutable, "_pidfd_exited", lambda _fd, _timeout: True)
    try:
        assert immutable._open_exact_pidfd(123, 456, "test") == descriptor
    finally:
        os.close(descriptor)
        os.close(writable)


@pytest.mark.parametrize("disappear_on", [signal.SIGTERM, signal.SIGKILL])
def test_exact_pidfd_signal_esrch_is_success_only_after_verified_exit(
    monkeypatch: pytest.MonkeyPatch, disappear_on: signal.Signals
) -> None:
    exited = False
    sent: list[signal.Signals] = []

    def pidfd_exited(_descriptor: int, timeout_seconds: float) -> bool:
        if exited:
            return True
        return disappear_on == signal.SIGKILL and timeout_seconds == 0 and bool(sent)

    def send(_descriptor: int, signum: int) -> None:
        nonlocal exited
        selected = signal.Signals(signum)
        sent.append(selected)
        if selected == disappear_on:
            exited = True
            raise ProcessLookupError

    monkeypatch.setattr(immutable, "_pidfd_exited", pidfd_exited)
    monkeypatch.setattr(signal, "pidfd_send_signal", send)

    result = immutable._stop_exact_pidfd(
        77,
        pid=123,
        starttime=456,
        label="test",
        timeout_seconds=0.01,
    )

    assert result["pid"] == 123
    assert result["starttime"] == 456
    assert result["verified_exited"] is True
    assert sent[-1] == disappear_on


def test_exact_pidfd_signal_esrch_fails_when_pidfd_is_not_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(immutable, "_pidfd_exited", lambda _fd, _timeout: False)

    def disappeared(_descriptor: int, _signum: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(signal, "pidfd_send_signal", disappeared)

    with pytest.raises(
        immutable.ImmutableRuntimeMountError,
        match="without an exited pidfd",
    ):
        immutable._stop_exact_pidfd(
            77,
            pid=123,
            starttime=456,
            label="test",
            timeout_seconds=0.01,
        )


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
                **_mount_kwargs(tmp_path),
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
                **_mount_kwargs(tmp_path),
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
