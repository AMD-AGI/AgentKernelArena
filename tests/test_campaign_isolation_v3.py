"""Attack-focused tests for the formal role-based mount boundary."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import logging
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import campaign_isolation


apex_launcher = importlib.import_module("agents.apex.launch_agent")


def _formal_config() -> dict[str, object]:
    return {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {"fresh_session": True},
    }


def _disable_requirements_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_requirements_identity",
        lambda: (Path("/etc/codex/requirements.toml"), {"sha256": "f" * 64}),
    )


def _role_tree(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    roles = {
        "scored_workspace": attempt / "workspace",
        "sealed_task_contract": attempt / "task-contract",
        "apex_runtime": tmp_path / "trusted-apex-runtime",
        "apex_artifacts": attempt / "apex-artifacts",
        "backend_home": attempt / "backend-home",
    }
    for path in roles.values():
        path.mkdir(parents=True)
    return data_root, roles


def _mount_entry(
    *,
    mount_id: int,
    target: Path,
    major_minor: str,
    root: Path = Path("/source"),
    access: str = "ro",
) -> campaign_isolation._MountInfo:
    return campaign_isolation._MountInfo(
        mount_id=mount_id,
        parent_id=1,
        major_minor=major_minor,
        root=root,
        mount_point=target,
        mount_options=(access,),
        filesystem_type="ext4",
        source="/dev/test",
        super_options=("rw",),
    )


@pytest.fixture
def self_mount_boundary() -> Iterator[tuple[int, int, int, int]]:
    pid = os.getpid()
    pidfd = os.pidfd_open(pid, 0)
    try:
        yield (
            pid,
            campaign_isolation._mount_namespace_inode(pid),
            pidfd,
            campaign_isolation._process_starttime(pid),
        )
    finally:
        os.close(pidfd)


def test_visible_mount_uses_proc_root_o_path_fdinfo_id(
    tmp_path, monkeypatch, self_mount_boundary
) -> None:
    pid, namespace_id, pidfd, starttime = self_mount_boundary
    target = tmp_path / "target"
    target.mkdir()
    metadata = target.stat()
    major_minor = f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"
    visible = _mount_entry(
        mount_id=17,
        target=target,
        major_minor=major_minor,
    )
    covered = _mount_entry(
        mount_id=999,
        target=target,
        major_minor=major_minor,
    )
    # The visible entry is deliberately not the highest mount ID.  Device and
    # exact path are also identical, so neither heuristic can select it.
    monkeypatch.setattr(campaign_isolation, "_descriptor_mount_id", lambda _fd: 17)

    observed, target_stat, hidden = campaign_isolation._visible_namespace_mount(
        pid=pid,
        target=target,
        table={999: covered, 17: visible},
        mount_namespace_id=namespace_id,
        init_pidfd=pidfd,
        init_starttime=starttime,
    )

    assert observed.mount_id == 17
    assert (target_stat.st_dev, target_stat.st_ino) == (
        metadata.st_dev,
        metadata.st_ino,
    )
    assert [entry.mount_id for entry in hidden] == [999]


@pytest.mark.parametrize("attack", ["missing_id", "wrong_exact_target"])
def test_visible_mount_fails_closed_for_unbound_fdinfo_id(
    tmp_path, monkeypatch, attack, self_mount_boundary
) -> None:
    pid, namespace_id, pidfd, starttime = self_mount_boundary
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.setattr(campaign_isolation, "_descriptor_mount_id", lambda _fd: 17)
    entry_target = target if attack == "missing_id" else tmp_path / "other"
    entry_id = 18 if attack == "missing_id" else 17
    table = {
        entry_id: _mount_entry(
            mount_id=entry_id,
            target=entry_target,
            major_minor="0:1",
        )
    }

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="visible mount id does not name the exact target",
    ):
        campaign_isolation._visible_namespace_mount(
            pid=pid,
            target=target,
            table=table,
            mount_namespace_id=namespace_id,
            init_pidfd=pidfd,
            init_starttime=starttime,
        )


def test_visible_mount_rechecks_namespace_after_fd_observation(
    tmp_path, monkeypatch, self_mount_boundary
) -> None:
    pid, _namespace_id, pidfd, starttime = self_mount_boundary
    target = tmp_path / "target"
    target.mkdir()
    namespace_ids = iter((41, 42))
    monkeypatch.setattr(
        campaign_isolation,
        "_mount_namespace_inode",
        lambda _pid: next(namespace_ids),
    )
    monkeypatch.setattr(campaign_isolation, "_descriptor_mount_id", lambda _fd: 17)
    table = {
        17: _mount_entry(
            mount_id=17,
            target=target,
            major_minor="0:1",
        )
    }

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="namespace init identity changed during mount attestation",
    ):
        campaign_isolation._visible_namespace_mount(
            pid=pid,
            target=target,
            table=table,
            mount_namespace_id=41,
            init_pidfd=pidfd,
            init_starttime=starttime,
        )


@pytest.mark.parametrize("attack", ["pidfd_exit", "starttime_change"])
def test_visible_mount_rechecks_pinned_process_after_fd_observation(
    tmp_path, monkeypatch, self_mount_boundary, attack
) -> None:
    pid, namespace_id, pidfd, starttime = self_mount_boundary
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.setattr(campaign_isolation, "_descriptor_mount_id", lambda _fd: 17)
    if attack == "pidfd_exit":
        readiness = iter((False, True))
        monkeypatch.setattr(
            campaign_isolation,
            "_pidfd_ready",
            lambda _fd, _timeout: next(readiness),
        )
    else:
        starttimes = iter((starttime, starttime + 1))
        monkeypatch.setattr(
            campaign_isolation,
            "_process_starttime",
            lambda _pid: next(starttimes),
        )
    table = {
        17: _mount_entry(
            mount_id=17,
            target=target,
            major_minor="0:1",
        )
    }

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="namespace init identity changed during mount attestation",
    ):
        campaign_isolation._visible_namespace_mount(
            pid=pid,
            target=target,
            table=table,
            mount_namespace_id=namespace_id,
            init_pidfd=pidfd,
            init_starttime=starttime,
        )


def test_mount_attestation_rejects_changed_mountinfo_table(
    tmp_path, monkeypatch, self_mount_boundary
) -> None:
    pid, namespace_id, pidfd, starttime = self_mount_boundary
    target = tmp_path / "target"
    target.mkdir()
    original = {
        17: _mount_entry(
            mount_id=17,
            target=target,
            major_minor="0:1",
        )
    }
    changed = {
        **original,
        18: _mount_entry(
            mount_id=18,
            target=tmp_path / "covered",
            major_minor="0:1",
        ),
    }
    monkeypatch.setattr(
        campaign_isolation,
        "_namespace_mountinfo_table",
        lambda _pid: changed,
    )

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="mount table changed during blocked attestation",
    ):
        campaign_isolation._assert_namespace_mountinfo_unchanged(
            pid=pid,
            mount_namespace_id=namespace_id,
            init_pidfd=pidfd,
            init_starttime=starttime,
            expected=original,
        )


def test_covered_exact_mounts_are_runtime_only_and_projection_stable() -> None:
    target = Path("/tmp/runtime-pin")
    visible = _mount_entry(
        mount_id=31,
        target=target,
        major_minor="8:1",
        root=Path("/runtime-source"),
    )
    inherited = _mount_entry(
        mount_id=30,
        target=target,
        major_minor="8:1",
        root=Path("/runtime-source"),
    )
    source_mount = {
        "mount_point": str(target),
        "major_minor": "8:1",
        "root": "/runtime-source",
        "access": "read_only",
        "filesystem_type": "ext4",
        "source": "/dev/test",
        "super_options": ["rw"],
    }
    campaign_isolation._validate_role_covered_mounts(
        role="apex_runtime",
        path=target,
        expected_access="read_only",
        source_mount=source_mount,
        visible=visible,
        covered=(inherited,),
    )

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="role has covered exact mounts",
    ):
        campaign_isolation._validate_role_covered_mounts(
            role="scored_workspace",
            path=target,
            expected_access="read_only",
            source_mount=source_mount,
            visible=visible,
            covered=(inherited,),
        )
    changed = _mount_entry(
        mount_id=30,
        target=target,
        major_minor="8:1",
        root=Path("/different-source"),
    )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="covered mount changed projection",
    ):
        campaign_isolation._validate_role_covered_mounts(
            role="apex_runtime",
            path=target,
            expected_access="read_only",
            source_mount=source_mount,
            visible=visible,
            covered=(changed,),
        )
    jointly_changed_visible = _mount_entry(
        mount_id=31,
        target=target,
        major_minor="8:1",
        root=Path("/jointly-different-source"),
    )
    jointly_changed_inherited = _mount_entry(
        mount_id=30,
        target=target,
        major_minor="8:1",
        root=Path("/jointly-different-source"),
    )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="covered mount changed projection",
    ):
        campaign_isolation._validate_role_covered_mounts(
            role="apex_runtime",
            path=target,
            expected_access="read_only",
            source_mount=source_mount,
            visible=jointly_changed_visible,
            covered=(jointly_changed_inherited,),
        )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="runtime mount stack is not canonical",
    ):
        campaign_isolation._validate_role_covered_mounts(
            role="apex_runtime",
            path=target,
            expected_access="read_only",
            source_mount=source_mount,
            visible=visible,
            covered=(),
        )


def test_role_mount_real_bwrap_preserves_exact_access_classes(
    tmp_path, monkeypatch
) -> None:
    data_root, roles = _role_tree(tmp_path)
    workspace_file = roles["scored_workspace"] / "kernel.py"
    contract_file = roles["sealed_task_contract"] / "task_spec.json"
    runtime_file = roles["apex_runtime"] / "runtime.py"
    workspace_file.write_text("baseline\n", encoding="utf-8")
    contract_file.write_text("{}\n", encoding="utf-8")
    runtime_file.write_text("runtime\n", encoding="utf-8")
    sibling_secret = data_root / "run/task/attempt_02/secret"
    sibling_secret.parent.mkdir(parents=True)
    sibling_secret.write_text("hidden\n", encoding="utf-8")
    tmp_sentinel = Path("/tmp") / f"aka-v2-{tmp_path.name}"
    shm_sentinel = Path("/dev/shm") / f"aka-v2-{tmp_path.name}"
    tmp_sentinel.write_text("host\n", encoding="utf-8")
    shm_sentinel.write_text("host\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)
    probe = (
        "import json,os,pathlib,sys\n"
        "workspace,contract,runtime,artifacts,home,sibling,tmp_s,shm_s=map(pathlib.Path,sys.argv[1:])\n"
        "def denied(root):\n"
        " try: (root/'forbidden').write_text('bad'); return False\n"
        " except OSError: return True\n"
        "fd_targets=[]\n"
        "for fd in pathlib.Path('/proc/self/fd').iterdir():\n"
        " try: fd_targets.append(os.readlink(fd))\n"
        " except OSError: pass\n"
        "result={'workspace_ro':denied(workspace),'contract_ro':denied(contract),'runtime_ro':denied(runtime),'artifact_rw':not denied(artifacts),'home_rw':not denied(home),'sibling_hidden':not sibling.exists(),'tmp_private':not tmp_s.exists(),'shm_private':not shm_s.exists(),'mount_fds_closed':not any(str(p) in fd_targets for p in (workspace,contract,runtime,artifacts,home))}\n"
        "pathlib.Path('/tmp/private-ok').write_text('ok')\n"
        "pathlib.Path('/dev/shm/private-ok').write_text('ok')\n"
        "print(json.dumps(result,sort_keys=True))\n"
    )
    system_python = str(Path("/usr/bin/python3").resolve(strict=True))
    command = campaign_isolation.wrap_attempt_command(
        [
            system_python,
            "-c",
            probe,
            *(str(roles[name]) for name in (
                "scored_workspace",
                "sealed_task_contract",
                "apex_runtime",
                "apex_artifacts",
                "backend_home",
            )),
            str(sibling_secret),
            str(tmp_sentinel),
            str(shm_sentinel),
        ],
        eval_config=_formal_config(),
        writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
        read_only_roots=(
            roles["scored_workspace"],
            roles["sealed_task_contract"],
        ),
        trusted_read_only_roots=(roles["apex_runtime"],),
        mount_roles=roles,
        private_proc=False,
    )
    receipt = campaign_isolation.attempt_mount_receipt(command)
    try:
        outcome = apex_launcher._run_apex(
            command,
            cwd=roles["apex_artifacts"],
            backend="codex",
            timeout_seconds=10,
            output_limit=1024 * 1024,
            logger=logging.getLogger(__name__),
        )
        assert outcome.exit_code == 0, outcome.output
        assert all(json.loads(outcome.stdout).values())
        assert receipt["schema"] == campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA_V3
        assert set(receipt["campaign_data_identity"]["mount"]) == {
            "mount_id",
            "parent_id",
            "major_minor",
            "root",
            "mount_point",
        }
        assert set(receipt["roles"]["read_only"]) == {
            "scored_workspace",
            "sealed_task_contract",
            "apex_runtime",
        }
        assert set(receipt["roles"]["persistent_writable"]) == {
            "apex_artifacts",
            "backend_home",
        }
        assert receipt["outer_bubblewrap"]["canonical_path"] == "/usr/bin/bwrap"
        assert receipt["outer_bubblewrap"]["source"]["sha256"] == (
            campaign_isolation._OUTER_BWRAP_SHA256
        )
        assert receipt["outer_bubblewrap"]["sealed_exec"] == {
            "transport": "sealed_memfd_proc_self_fd",
            "size_bytes": campaign_isolation._OUTER_BWRAP_SIZE_BYTES,
            "sha256": campaign_isolation._OUTER_BWRAP_SHA256,
            "seals": list(campaign_isolation._OUTER_BWRAP_SEALS),
        }
        namespace = receipt["namespace_mounts"]
        assert namespace["policy"] == "blocked_namespace_mount_attestation_v2"
        assert namespace["visible_mount_resolution_policy"] == (
            "proc_root_o_path_fdinfo_mnt_id_v1"
        )
        assert namespace["root"]["access"] == "read_only"
        assert namespace["root"]["covered_mount_ids"] == []
        assert namespace["campaign_data_root"]["access"] == "read_write"
        assert namespace["closed_set"] is True
        assert namespace["aliases_absent"] is True
        for role in ("scored_workspace", "sealed_task_contract", "apex_runtime"):
            observed = namespace["roles"]["read_only"][role]
            assert receipt["roles"]["read_only"][role]["mount"] == observed[
                "source"
            ]["mount"]
            assert observed["source"]["mount"] != observed["target"]["mount"]
            assert observed["target"]["access"] == "read_only"
            assert observed["target"]["device"] == observed["source"]["device"]
            assert observed["target"]["inode"] == observed["source"]["inode"]
            assert observed["target"]["covered_mount_ids"] == []
        for role in ("apex_artifacts", "backend_home"):
            observed = namespace["roles"]["persistent_writable"][role]
            assert receipt["roles"]["persistent_writable"][role]["mount"] == observed[
                "source"
            ]["mount"]
            assert observed["source"]["mount"] != observed["target"]["mount"]
            assert observed["target"]["access"] == "read_write"
            assert observed["target"]["device"] == observed["source"]["device"]
            assert observed["target"]["inode"] == observed["source"]["inode"]
            assert observed["target"]["covered_mount_ids"] == []
        material = dict(receipt)
        digest = material.pop("sha256")
        assert digest == campaign_isolation.canonical_digest(material)
        assert workspace_file.read_text(encoding="utf-8") == "baseline\n"
        assert contract_file.read_text(encoding="utf-8") == "{}\n"
        assert runtime_file.read_text(encoding="utf-8") == "runtime\n"
        assert (roles["apex_artifacts"] / "forbidden").is_file()
        assert (roles["backend_home"] / "forbidden").is_file()
    finally:
        campaign_isolation.release_attempt_command_fds(command)
        tmp_sentinel.unlink(missing_ok=True)
        shm_sentinel.unlink(missing_ok=True)


def test_role_mount_rejects_scored_workspace_as_writable(
    tmp_path, monkeypatch
) -> None:
    data_root, roles = _role_tree(tmp_path)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="role classes differ",
    ):
        campaign_isolation.wrap_attempt_command(
            ["/bin/true"],
            eval_config=_formal_config(),
            writable_roots=(
                roles["scored_workspace"],
                roles["apex_artifacts"],
                roles["backend_home"],
            ),
            read_only_roots=(
                roles["sealed_task_contract"],
            ),
            trusted_read_only_roots=(roles["apex_runtime"],),
            mount_roles=roles,
        )


def test_role_mount_requires_one_external_trusted_runtime(
    tmp_path, monkeypatch
) -> None:
    data_root, roles = _role_tree(tmp_path)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)
    internal_runtime = data_root / "run/task/attempt_01/internal-runtime"
    internal_runtime.mkdir()
    internal_roles = {**roles, "apex_runtime": internal_runtime}

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="trusted read-only root must be outside campaign data root",
    ):
        campaign_isolation.wrap_attempt_command(
            ["/bin/true"],
            eval_config=_formal_config(),
            writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
            read_only_roots=(
                roles["scored_workspace"],
                roles["sealed_task_contract"],
            ),
            trusted_read_only_roots=(internal_runtime,),
            mount_roles=internal_roles,
        )

    extra = tmp_path / "undeclared-trusted-root"
    extra.mkdir()
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="role classes differ",
    ):
        campaign_isolation.wrap_attempt_command(
            ["/bin/true"],
            eval_config=_formal_config(),
            writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
            read_only_roots=(
                roles["scored_workspace"],
                roles["sealed_task_contract"],
            ),
            trusted_read_only_roots=(roles["apex_runtime"], extra),
            mount_roles=roles,
        )


def test_mount_identity_rejects_bind_alias_and_nested_mount(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias_identity = {
        "device": 7,
        "inode": 9,
        "mount": {
            "mount_id": 10,
            "major_minor": "8:1",
            "root": "/source",
        },
    }
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="filesystem aliases",
    ):
        campaign_isolation._reject_mount_aliases(
            {first: alias_identity, second: dict(alias_identity)}
        )

    descriptor = os.open(first, os.O_DIRECTORY | getattr(os, "O_PATH", os.O_RDONLY))
    try:
        mount_id = campaign_isolation._descriptor_mount_id(descriptor)
    finally:
        os.close(descriptor)
    table = campaign_isolation._mountinfo_table()
    table[max(table) + 1] = campaign_isolation._MountInfo(
        mount_id=max(table) + 1,
        parent_id=mount_id,
        major_minor=table[mount_id].major_minor,
        root=table[mount_id].root / "nested",
        mount_point=first / "nested",
    )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="undeclared nested mounts",
    ):
        campaign_isolation._open_mount_root(first, table=table)


def test_pinned_mount_fd_defeats_validation_to_exec_path_replacement(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign"
    workspace = data_root / "run/attempt/workspace"
    artifacts = data_root / "run/attempt/artifacts"
    workspace.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (workspace / "kernel.py").write_text("trusted\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)
    command = campaign_isolation.wrap_attempt_command(
        ["/bin/sh", "-c", f"cat {workspace / 'kernel.py'}"],
        eval_config=_formal_config(),
        writable_roots=(artifacts,),
        read_only_roots=(workspace,),
        private_proc=False,
    )
    original = workspace.with_name("workspace-original")
    workspace.rename(original)
    workspace.mkdir()
    (workspace / "kernel.py").write_text("replacement\n", encoding="utf-8")
    outcome = apex_launcher._run_apex(
        command,
        cwd=artifacts,
        backend="codex",
        timeout_seconds=10,
        output_limit=1024,
        logger=logging.getLogger(__name__),
    )
    assert outcome.exit_code == 0, outcome.output
    assert outcome.stdout == b"trusted\n"


def test_outer_bwrap_exec_ignores_adversarial_path_and_is_sealed(
    tmp_path, monkeypatch
) -> None:
    data_root, roles = _role_tree(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    marker = tmp_path / "fake-bwrap-executed"
    fake_bin.mkdir()
    fake = fake_bin / "bwrap"
    fake.write_text(
        f"#!/bin/sh\nprintf bad > {marker}\nexit 91\n", encoding="utf-8"
    )
    fake.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)

    command = campaign_isolation.wrap_attempt_command(
        ["/bin/true"],
        eval_config=_formal_config(),
        writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
        read_only_roots=(
            roles["scored_workspace"],
            roles["sealed_task_contract"],
        ),
        trusted_read_only_roots=(roles["apex_runtime"],),
        mount_roles=roles,
        private_proc=False,
    )
    assert command[0].startswith("/proc/self/fd/")
    receipt = campaign_isolation.attempt_mount_receipt(command)
    outcome = apex_launcher._run_apex(
        command,
        cwd=roles["apex_artifacts"],
        backend="codex",
        timeout_seconds=10,
        output_limit=1024,
        logger=logging.getLogger(__name__),
    )

    assert outcome.exit_code == 0
    assert not marker.exists()
    assert receipt["outer_bubblewrap"]["canonical_path"] == "/usr/bin/bwrap"
    assert receipt["outer_bubblewrap"]["sealed_exec"]["sha256"] == (
        campaign_isolation._OUTER_BWRAP_SHA256
    )
    assert receipt["namespace_mounts"]["closed_set"] is True


def test_outer_bwrap_memfd_has_exact_bytes_and_irrevocable_seals() -> None:
    descriptor, identity = campaign_isolation._sealed_outer_bwrap()
    try:
        expected_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
        )
        payload = os.pread(
            descriptor, campaign_isolation._OUTER_BWRAP_SIZE_BYTES + 1, 0
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == expected_seals
        assert len(payload) == campaign_isolation._OUTER_BWRAP_SIZE_BYTES
        assert hashlib.sha256(payload).hexdigest() == (
            campaign_isolation._OUTER_BWRAP_SHA256
        )
        assert identity["source"]["sha256"] == identity["sealed_exec"]["sha256"]
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "attack", ["writable_workspace", "nested_mount", "nested_runtime_mount"]
)
def test_mount_attestation_fails_before_untrusted_exec(
    tmp_path, monkeypatch, attack
) -> None:
    data_root, roles = _role_tree(tmp_path)
    executed = roles["apex_artifacts"] / "untrusted-executed"
    nested_root = (
        roles["apex_runtime"]
        if attack == "nested_runtime_mount"
        else roles["scored_workspace"]
    )
    nested = nested_root / "nested"
    nested.mkdir()
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    _disable_requirements_probe(monkeypatch)
    command = campaign_isolation.wrap_attempt_command(
        ["/bin/sh", "-c", f"printf bad > {executed}"],
        eval_config=_formal_config(),
        writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
        read_only_roots=(
            roles["scored_workspace"],
            roles["sealed_task_contract"],
        ),
        trusted_read_only_roots=(roles["apex_runtime"],),
        mount_roles=roles,
        private_proc=False,
    )
    if attack == "writable_workspace":
        for index, value in enumerate(command):
            if (
                value == str(roles["scored_workspace"])
                and index >= 2
                and command[index - 2] == "--ro-bind-fd"
            ):
                command[index - 2] = "--bind-fd"
                break
        else:
            raise AssertionError("workspace bind was not found")
        error = "differs from pinned source"
    else:
        insertion = command.index("--json-status-fd")
        command[insertion:insertion] = ["--tmpfs", str(nested)]
        error = "closed declared set|undeclared mounts"

    with pytest.raises(campaign_isolation.CampaignIsolationError, match=error):
        apex_launcher._run_apex(
            command,
            cwd=roles["apex_artifacts"],
            backend="codex",
            timeout_seconds=10,
            output_limit=1024,
            logger=logging.getLogger(__name__),
        )
    assert not executed.exists()


def _run_outer_inner_mount_child(data_root: Path, runtime: Path) -> int:
    attempt = data_root / "run/task/attempt_01"
    roles = {
        "scored_workspace": attempt / "workspace",
        "sealed_task_contract": attempt / "task-contract",
        "apex_runtime": runtime,
        "apex_artifacts": attempt / "apex-artifacts",
        "backend_home": attempt / "backend-home",
    }
    os.environ["AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT"] = str(data_root)
    campaign_isolation._codex_requirements_identity = lambda: (
        Path("/etc/codex/requirements.toml"),
        {"sha256": "f" * 64},
    )
    command = campaign_isolation.wrap_attempt_command(
        ["/bin/true"],
        eval_config=_formal_config(),
        writable_roots=(roles["apex_artifacts"], roles["backend_home"]),
        read_only_roots=(
            roles["scored_workspace"],
            roles["sealed_task_contract"],
        ),
        trusted_read_only_roots=(roles["apex_runtime"],),
        mount_roles=roles,
        private_proc=False,
    )
    receipt = campaign_isolation.attempt_mount_receipt(command)
    try:
        outcome = apex_launcher._run_apex(
            command,
            cwd=roles["apex_artifacts"],
            backend="codex",
            timeout_seconds=10,
            output_limit=1024 * 1024,
            logger=logging.getLogger(__name__),
        )
        if outcome.exit_code != 0:
            sys.stderr.buffer.write(outcome.output)
            return outcome.exit_code
        runtime_target = receipt["namespace_mounts"]["roles"]["read_only"][
            "apex_runtime"
        ]["target"]
        print(
            json.dumps(
                {
                    "schema": receipt["schema"],
                    "policy": receipt["namespace_mounts"]["policy"],
                    "visible_mount_resolution_policy": receipt[
                        "namespace_mounts"
                    ]["visible_mount_resolution_policy"],
                    "runtime_visible_mount_id": runtime_target["mount"]["mount_id"],
                    "runtime_covered_mount_ids": runtime_target[
                        "covered_mount_ids"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        campaign_isolation.release_attempt_command_fds(command)


def test_real_outer_runtime_bind_survives_inner_bwrap_attestation(tmp_path) -> None:
    data_source = tmp_path / "outer-data-source"
    data_target = tmp_path / "outer-data-target"
    runtime_source = tmp_path / "outer-runtime-source"
    runtime_target = tmp_path / "outer-runtime-target"
    attempt = data_source / "run/task/attempt_01"
    for path in (
        attempt / "workspace",
        attempt / "task-contract",
        attempt / "apex-artifacts",
        attempt / "backend-home",
        data_target,
        runtime_source,
        runtime_target,
    ):
        path.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(repository), inherited_pythonpath)
        if part
    )
    outer_prefix: list[str] = []
    if os.geteuid() != 0:
        sudo_probe = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            check=False,
        )
        if sudo_probe.returncode != 0:
            pytest.skip("nested mount-namespace regression requires passwordless sudo")
        outer_prefix = ["sudo", "-n"]
    completed = subprocess.run(
        [
            *outer_prefix,
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            "/proc",
            "/proc",
            "--dev-bind",
            "/dev",
            "/dev",
            "--bind",
            str(data_source),
            str(data_target),
            "--ro-bind",
            str(runtime_source),
            str(runtime_target),
            "--chdir",
            str(repository),
            "--",
            "/usr/bin/env",
            f"PYTHONPATH={environment['PYTHONPATH']}",
            sys.executable,
            str(Path(__file__).resolve()),
            "--outer-inner-mount-child",
            str(data_target),
            str(runtime_target),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert set(evidence) == {
        "policy",
        "runtime_covered_mount_ids",
        "runtime_visible_mount_id",
        "schema",
        "visible_mount_resolution_policy",
    }
    assert evidence["policy"] == "blocked_namespace_mount_attestation_v2"
    assert evidence["schema"] == campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA_V3
    assert evidence["visible_mount_resolution_policy"] == (
        "proc_root_o_path_fdinfo_mnt_id_v1"
    )
    assert type(evidence["runtime_visible_mount_id"]) is int
    assert evidence["runtime_visible_mount_id"] > 0
    assert len(evidence["runtime_covered_mount_ids"]) == 1
    assert type(evidence["runtime_covered_mount_ids"][0]) is int
    assert evidence["runtime_covered_mount_ids"][0] > 0
    assert evidence["runtime_visible_mount_id"] not in evidence[
        "runtime_covered_mount_ids"
    ]


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--outer-inner-mount-child":
        raise SystemExit(
            _run_outer_inner_mount_child(Path(sys.argv[2]), Path(sys.argv[3]))
        )
    raise SystemExit("unsupported test helper invocation")
