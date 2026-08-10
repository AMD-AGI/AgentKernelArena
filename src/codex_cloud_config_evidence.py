#!/usr/bin/env python3
"""Evidence and state primitives for private Codex cloud-config refreshes.

Formal AgentKernelArena campaigns use this host-side helper to obtain a fresh
signed Codex cloud-config envelope without invoking a model. This module owns
bounded stable reads, private state, CLI-closure identity, and policy. The
companion refresh module owns command execution, publication, receipts, and the
supervisor lifecycle; the Docker launcher owns its process and bind mount.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

try:
    from .aka_runtime import AkaRuntimeError, capture_backend_closure
except ImportError:  # Script execution places src/ directly on sys.path.
    from aka_runtime import AkaRuntimeError, capture_backend_closure


CONTROL_SCHEMA = "aka.formal-codex-cloud-config-refresh-control/v1"
HEALTH_SCHEMA = "aka.formal-codex-cloud-config-refresh-health/v1"
RECEIPT_SCHEMA = "aka.formal-codex-cloud-config-refresh/v3"
RECEIPT_POLICY = "private_auth_only_app_server_refresh_v3"
PRIVATE_ROOT_PREFIX = "agentkernelarena-formal-codex."
MAX_JSON_BYTES = 1024 * 1024
MINIMUM_REFRESH_SCHEDULING_SLACK_SECONDS = 30
DEFAULT_REFRESH_EARLY_SECONDS = 900
DEFAULT_MINIMUM_TTL_SECONDS = 630
DEFAULT_MAXIMUM_ENVELOPE_LIFETIME_SECONDS = 7_200
DEFAULT_CLOCK_SKEW_SECONDS = 300
_CLOUD_CONFIG_TIMESTAMP = re.compile(
    r"^(?P<seconds>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?Z$"
)
ALLOWED_ENVIRONMENT = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)


class RefreshError(RuntimeError):
    """A fail-closed refresh error with a receipt-safe error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SupervisorStop(RuntimeError):
    """Internal control flow for an intentional supervisor shutdown."""


@dataclasses.dataclass(frozen=True)
class Policy:
    timeout_seconds: int = 30
    term_grace_seconds: int = 5
    output_limit_bytes: int = 65_536
    refresh_early_seconds: int = DEFAULT_REFRESH_EARLY_SECONDS
    minimum_ttl_seconds: int = DEFAULT_MINIMUM_TTL_SECONDS
    maximum_envelope_lifetime_seconds: int = (
        DEFAULT_MAXIMUM_ENVELOPE_LIFETIME_SECONDS
    )
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS

    def validate(self) -> None:
        values = dataclasses.asdict(self)
        if any(not isinstance(value, int) or value <= 0 for value in values.values()):
            raise RefreshError("invalid_refresh_policy")
        required_refresh_window = (
            self.minimum_ttl_seconds
            + self.timeout_seconds
            + 2 * self.term_grace_seconds
            + MINIMUM_REFRESH_SCHEDULING_SLACK_SECONDS
        )
        if self.refresh_early_seconds <= required_refresh_window:
            raise RefreshError("invalid_refresh_ttl_margin")


@dataclasses.dataclass(frozen=True)
class CliIdentity:
    node_prefix: str
    launcher_path: str
    launcher_resolved_path: str
    launcher_sha256: str
    node_resolved_path: str
    node_sha256: str
    backend_runtime_closure_sha256: str


@dataclasses.dataclass
class RefreshState:
    root: str
    root_device: int
    root_inode: int
    uid: int
    gid: int
    campaign_data_root: str
    work_home: str
    published_directory: str
    cli: CliIdentity
    owner_pid: int
    owner_starttime: int
    policy: Policy
    sequence: int = 0
    anchor_bundle_sha256: str = ""
    next_refresh_epoch: int = 0


@dataclasses.dataclass(frozen=True)
class CommandResult:
    exit_code: int
    timed_out: bool
    failure: str | None


@dataclasses.dataclass(frozen=True)
class RefreshOutcome:
    status: str
    failure: str | None
    sequence: int
    bundle_sha256: str
    next_refresh_epoch: int
    receipt_sha256: str
    receipt_path: str
    promoted: bool


@dataclasses.dataclass
class SupervisorContext:
    stopped: threading.Event = dataclasses.field(default_factory=threading.Event)
    process: subprocess.Popen[bytes] | None = None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RefreshError("duplicate_json_key")
        result[key] = value
    return result


def _strict_json(raw: bytes, failure: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except RefreshError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RefreshError(failure) from error


def _stable_read(path: Path, limit: int, *, allow_empty: bool = False) -> bytes:
    for _ in range(5):
        try:
            descriptor = os.open(
                path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except OSError as error:
            raise RefreshError("unsafe_or_missing_file") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RefreshError("unsafe_regular_file")
            if before.st_size < 0 or before.st_size > limit:
                raise RefreshError("invalid_file_size")
            if not allow_empty and before.st_size == 0:
                raise RefreshError("empty_file")
            chunks: list[bytes] = []
            total = 0
            while total <= limit:
                chunk = os.read(descriptor, min(65_536, limit + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total == before.st_size and all(
            getattr(before, field) == getattr(after, field) for field in fields
        ):
            return b"".join(chunks)
        time.sleep(0.05)
    raise RefreshError("unstable_file")


def _stable_sha256(path: Path) -> str:
    for _ in range(5):
        try:
            descriptor = os.open(
                path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except OSError as error:
            raise RefreshError("unsafe_cli_file") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RefreshError("unsafe_cli_file")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total == before.st_size and all(
            getattr(before, field) == getattr(after, field) for field in fields
        ):
            return digest.hexdigest()
        time.sleep(0.05)
    raise RefreshError("unstable_cli_file")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_timestamp(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise RefreshError("invalid_expiry")
    match = _CLOUD_CONFIG_TIMESTAMP.fullmatch(value)
    if match is None:
        raise RefreshError("invalid_expiry")
    fraction = (match.group("fraction") or "").ljust(6, "0")[:6]
    try:
        parsed = dt.datetime.strptime(
            match.group("seconds"), "%Y-%m-%dT%H:%M:%S"
        ).replace(
            microsecond=int(fraction or "0"),
            tzinfo=dt.timezone.utc,
        )
    except ValueError as error:
        raise RefreshError("invalid_expiry") from error
    return parsed


def _process_starttime(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, separator, fields = raw.rpartition(") ")
        if not separator:
            return None
        value = fields.split()[19]
        return int(value)
    except (OSError, IndexError, UnicodeError, ValueError):
        return None


def _process_identity_matches(pid: int, starttime: int) -> bool:
    return pid > 1 and starttime > 0 and _process_starttime(pid) == starttime


def _validate_private_root(
    root: Path, expected_device: int | None = None, expected_inode: int | None = None
) -> os.stat_result:
    if (
        not root.is_absolute()
        or root.parent != Path("/tmp")
        or not root.name.startswith(PRIVATE_ROOT_PREFIX)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", root.name)
    ):
        raise RefreshError("unsafe_private_root_path")
    try:
        identity = root.lstat()
    except OSError as error:
        raise RefreshError("missing_private_root") from error
    if (
        stat.S_ISLNK(identity.st_mode)
        or not stat.S_ISDIR(identity.st_mode)
        or stat.S_IMODE(identity.st_mode) != 0o700
        or identity.st_uid != os.getuid()
        or identity.st_gid != os.getgid()
    ):
        raise RefreshError("unsafe_private_root_identity")
    if expected_device is not None and identity.st_dev != expected_device:
        raise RefreshError("private_root_device_changed")
    if expected_inode is not None and identity.st_ino != expected_inode:
        raise RefreshError("private_root_inode_changed")
    return identity


def _write_all_exclusive(path: Path, value: bytes, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
    )
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RefreshError("short_write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, value: bytes, mode: int = 0o600) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RefreshError("short_atomic_write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_auth(raw: bytes) -> None:
    auth = _strict_json(raw, "invalid_auth_json")
    tokens = auth.get("tokens") if isinstance(auth, dict) else None
    account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
    if (
        not isinstance(auth, dict)
        or auth.get("auth_mode") != "chatgpt"
        or not isinstance(account_id, str)
        or not account_id
    ):
        raise RefreshError("invalid_chatgpt_auth_identity")


def _read_account_id(auth_path: Path) -> tuple[bytes, str]:
    raw = _stable_read(auth_path, MAX_JSON_BYTES)
    _validate_auth(raw)
    auth = _strict_json(raw, "invalid_auth_json")
    return raw, auth["tokens"]["account_id"]


def _capture_cli_identity(node_prefix: Path) -> CliIdentity:
    prefix = node_prefix.resolve(strict=True)
    launcher = prefix / "bin" / "codex"
    node = prefix / "bin" / "node"
    try:
        launcher_resolved = launcher.resolve(strict=True)
        node_resolved = node.resolve(strict=True)
    except OSError as error:
        raise RefreshError("missing_codex_cli_closure") from error
    if not os.access(launcher, os.X_OK) or not os.access(node, os.X_OK):
        raise RefreshError("non_executable_codex_cli_closure")
    original_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{prefix}/bin:/usr/bin:/bin"
    try:
        closure = capture_backend_closure("codex", str(launcher))
    except AkaRuntimeError as error:
        raise RefreshError("invalid_codex_runtime_closure") from error
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path
    closure_sha256 = closure.get("closure_sha256")
    if not isinstance(closure_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", closure_sha256
    ):
        raise RefreshError("invalid_codex_runtime_closure_digest")
    return CliIdentity(
        node_prefix=str(prefix),
        launcher_path=str(launcher),
        launcher_resolved_path=str(launcher_resolved),
        launcher_sha256=_stable_sha256(launcher_resolved),
        node_resolved_path=str(node_resolved),
        node_sha256=_stable_sha256(node_resolved),
        backend_runtime_closure_sha256=closure_sha256,
    )


def _validate_cli_identity(identity: CliIdentity) -> None:
    current = _capture_cli_identity(Path(identity.node_prefix))
    if current != identity:
        raise RefreshError("codex_cli_identity_changed")


def _state_material(state: RefreshState) -> dict[str, Any]:
    material: dict[str, Any] = {
        "schema": CONTROL_SCHEMA,
        "root": state.root,
        "root_identity": {
            "device": state.root_device,
            "inode": state.root_inode,
            "uid": state.uid,
            "gid": state.gid,
        },
        "campaign_data_root": state.campaign_data_root,
        "work_home": state.work_home,
        "published_directory": state.published_directory,
        "cli": dataclasses.asdict(state.cli),
        "owner": {"pid": state.owner_pid, "starttime": state.owner_starttime},
        "policy": dataclasses.asdict(state.policy),
        "sequence": state.sequence,
        "anchor_bundle_sha256": state.anchor_bundle_sha256,
        "next_refresh_epoch": state.next_refresh_epoch,
    }
    material["sha256"] = hashlib.sha256(_canonical_bytes(material)).hexdigest()
    return material


def _write_state(state: RefreshState) -> None:
    _validate_private_root(
        Path(state.root), state.root_device, state.root_inode
    )
    _atomic_write(
        Path(state.root) / "control.json",
        _canonical_bytes(_state_material(state)) + b"\n",
    )


def _load_state(root: Path, device: int, inode: int) -> RefreshState:
    _validate_private_root(root, device, inode)
    material = _strict_json(
        _stable_read(root / "control.json", MAX_JSON_BYTES), "invalid_control_json"
    )
    if not isinstance(material, dict) or material.get("schema") != CONTROL_SCHEMA:
        raise RefreshError("invalid_control_schema")
    digest = material.pop("sha256", None)
    if not isinstance(digest, str) or digest != hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest():
        raise RefreshError("invalid_control_digest")
    try:
        identity = material["root_identity"]
        owner = material["owner"]
        policy = Policy(**material["policy"])
        cli = CliIdentity(**material["cli"])
        state = RefreshState(
            root=material["root"],
            root_device=identity["device"],
            root_inode=identity["inode"],
            uid=identity["uid"],
            gid=identity["gid"],
            campaign_data_root=material["campaign_data_root"],
            work_home=material["work_home"],
            published_directory=material["published_directory"],
            cli=cli,
            owner_pid=owner["pid"],
            owner_starttime=owner["starttime"],
            policy=policy,
            sequence=material["sequence"],
            anchor_bundle_sha256=material["anchor_bundle_sha256"],
            next_refresh_epoch=material["next_refresh_epoch"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RefreshError("invalid_control_shape") from error
    policy.validate()
    expected_work = str(root / "work-home")
    expected_published = str(root / "published" / ".codex")
    if (
        state.root != str(root)
        or state.root_device != device
        or state.root_inode != inode
        or state.uid != os.getuid()
        or state.gid != os.getgid()
        or state.work_home != expected_work
        or state.published_directory != expected_published
        or state.sequence < 0
        or state.next_refresh_epoch < 0
        or (
            state.anchor_bundle_sha256
            and not re.fullmatch(r"[0-9a-f]{64}", state.anchor_bundle_sha256)
        )
    ):
        raise RefreshError("invalid_control_binding")
    return state


def prepare_state(
    auth_source: Path,
    node_prefix: Path,
    campaign_data_root: Path,
    owner_pid: int,
    owner_starttime: int,
    *,
    policy: Policy | None = None,
) -> RefreshState:
    selected_policy = policy or Policy()
    selected_policy.validate()
    if not _process_identity_matches(owner_pid, owner_starttime):
        raise RefreshError("invalid_campaign_owner_identity")
    data_root = campaign_data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise RefreshError("invalid_campaign_data_root")
    cli = _capture_cli_identity(node_prefix)
    auth_raw = _stable_read(auth_source, MAX_JSON_BYTES)
    _validate_auth(auth_raw)
    root = Path(tempfile.mkdtemp(prefix=PRIVATE_ROOT_PREFIX, dir="/tmp"))
    os.chmod(root, 0o700)
    identity = _validate_private_root(root)
    try:
        work_home = root / "work-home"
        work_codex = work_home / ".codex"
        published = root / "published" / ".codex"
        for directory in (
            work_home,
            work_codex,
            published.parent,
            published,
            root / "empty-cwd",
            root / "tmp",
            root / "cache",
        ):
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
        _write_all_exclusive(work_codex / "auth.json", auth_raw, 0o600)
        state = RefreshState(
            root=str(root),
            root_device=identity.st_dev,
            root_inode=identity.st_ino,
            uid=identity.st_uid,
            gid=identity.st_gid,
            campaign_data_root=str(data_root),
            work_home=str(work_home),
            published_directory=str(published),
            cli=cli,
            owner_pid=owner_pid,
            owner_starttime=owner_starttime,
            policy=selected_policy,
        )
        _write_state(state)
        return state
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
