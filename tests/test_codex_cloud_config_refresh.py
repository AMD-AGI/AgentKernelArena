from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

evidence = importlib.import_module("src.codex_cloud_config_evidence")
refresh = importlib.import_module("src.codex_cloud_config_refresh")


_ACCOUNT_ID = "00000000-0000-0000-0000-000000000001"


def _owner_identity(pid: int | None = None) -> tuple[int, int]:
    selected = pid or os.getpid()
    starttime = evidence._process_starttime(selected)
    assert starttime is not None
    return selected, starttime


def _write_auth(home: Path) -> Path:
    codex = home / ".codex"
    codex.mkdir(parents=True)
    path = codex / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "account_id": _ACCOUNT_ID,
                    "access_token": "secret-access-token",
                    "refresh_token": "secret-refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _write_fake_runtime(tmp_path: Path) -> tuple[Path, Path]:
    prefix = tmp_path / "node-prefix"
    package = prefix / "lib/node_modules/codex-fixture"
    dependency = package / "node_modules/fixture-dependency"
    binary = prefix / "bin"
    dependency.mkdir(parents=True)
    binary.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "codex-fixture",
                "dependencies": {"fixture-dependency": "1.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (dependency / "package.json").write_text(
        json.dumps({"name": "fixture-dependency", "version": "1.0.0"}),
        encoding="utf-8",
    )
    implementation = dependency / "index.js"
    implementation.write_text("module.exports = 1;\n", encoding="utf-8")
    launcher = package / "codex.py"
    launcher.write_text(
        r'''#!/usr/bin/env python3
import base64
import datetime
import json
import os
import sys
import time

assert sys.argv[1:] == ["app-server", "--listen", "stdio://"]
root = os.environ["CODEX_HOME"]
cache_path = os.path.join(root, "cloud-config-bundle-cache.json")
if os.path.exists(cache_path):
    raise SystemExit(87)
mode_path = os.path.join(root, "test-mode")
mode = open(mode_path, encoding="utf-8").read().strip() if os.path.exists(mode_path) else "success"
if mode == "nonzero":
    raise SystemExit(23)
if mode == "timeout":
    time.sleep(30)
bundle_path = os.path.join(root, "test-bundle")
bundle = open(bundle_path, encoding="utf-8").read().strip() if os.path.exists(bundle_path) else "fixture-a"
lifetime_path = os.path.join(root, "test-lifetime")
lifetime = int(open(lifetime_path, encoding="utf-8").read()) if os.path.exists(lifetime_path) else 3600
offset_path = os.path.join(root, "test-cached-offset")
offset = int(open(offset_path, encoding="utf-8").read()) if os.path.exists(offset_path) else 0
now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset)
auth = json.load(open(os.path.join(root, "auth.json"), encoding="utf-8"))
payload = {
    "version": 1,
    "cached_at": now.isoformat().replace("+00:00", "Z"),
    "expires_at": (now + datetime.timedelta(seconds=lifetime)).isoformat().replace("+00:00", "Z"),
    "chatgpt_user_id": "private-user-not-for-receipt",
    "account_id": auth["tokens"]["account_id"],
    "bundle": {
        "config_toml": {"marker": bundle},
        "requirements_toml": {"enterprise_managed": []},
    },
}
cache = {
    "signature": base64.b64encode(b"s" * 32).decode(),
    "signed_payload": payload,
}
with open(cache_path, "w", encoding="utf-8") as output:
    json.dump(cache, output, sort_keys=True, separators=(",", ":"))
print("benign app-server diagnostic", file=sys.stderr)
''',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (binary / "codex").symlink_to(launcher)
    (binary / "node").symlink_to("/bin/true")
    return prefix, implementation


def _policy() -> evidence.Policy:
    return evidence.Policy(
        timeout_seconds=2,
        term_grace_seconds=1,
        output_limit_bytes=4096,
        refresh_early_seconds=900,
        minimum_ttl_seconds=630,
        maximum_envelope_lifetime_seconds=7200,
        clock_skew_seconds=300,
    )


def _bootstrap(tmp_path: Path) -> tuple[evidence.RefreshState, refresh.RefreshOutcome, Path]:
    home = tmp_path / "host-home"
    auth = _write_auth(home)
    host_cache = home / ".codex/cloud-config-bundle-cache.json"
    host_cache.write_text("host-cache-must-not-change\n", encoding="utf-8")
    prefix, _ = _write_fake_runtime(tmp_path)
    data = tmp_path / "campaign-data"
    data.mkdir()
    owner_pid, owner_starttime = _owner_identity()
    state, outcome = refresh.bootstrap(
        auth,
        prefix,
        data,
        owner_pid,
        owner_starttime,
        policy=_policy(),
    )
    assert host_cache.read_text(encoding="utf-8") == "host-cache-must-not-change\n"
    return state, outcome, host_cache


def _cleanup(state: evidence.RefreshState) -> None:
    root = Path(state.root)
    if root.exists():
        refresh.cleanup_private_root(root, state.root_device, state.root_inode)


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01Z",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00.1234567890Z",
    ],
)
def test_cloud_config_timestamp_requires_canonical_utc_shape(value: str) -> None:
    with pytest.raises(evidence.RefreshError, match="invalid_expiry"):
        evidence._parse_timestamp(value)


def test_bootstrap_and_same_bundle_refresh_are_private_and_receipted(
    tmp_path: Path,
) -> None:
    state, initial, _ = _bootstrap(tmp_path)
    try:
        root = Path(state.root)
        published = Path(state.published_directory)
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert {path.name for path in published.iterdir()} == {
            "auth.json",
            "cloud-config-bundle-cache.json",
        }
        first_cache = (published / "cloud-config-bundle-cache.json").read_bytes()
        receipt = json.loads(Path(initial.receipt_path).read_text(encoding="utf-8"))
        assert receipt["schema"] == refresh.RECEIPT_SCHEMA
        assert receipt["policy_id"] == refresh.RECEIPT_POLICY
        assert receipt["status"] == "success"
        assert receipt["command"]["model_invocation"] is False
        assert receipt["command"]["stdout"]["size_bytes"] == 0
        assert receipt["command"]["stderr"]["size_bytes"] > 0
        assert receipt["cache"]["envelope_lifetime_seconds"] <= 7200
        assert receipt["cache"]["remaining_ttl_seconds"] >= 630
        assert receipt["cli"]["backend_runtime_closure_sha256"] == (
            state.cli.backend_runtime_closure_sha256
        )
        assert stat.S_IMODE(Path(initial.receipt_path).stat().st_mode) == 0o444
        receipt_text = Path(initial.receipt_path).read_text(encoding="utf-8")
        assert "secret-access-token" not in receipt_text
        assert "private-user-not-for-receipt" not in receipt_text
        assert _ACCOUNT_ID not in receipt_text

        time.sleep(0.01)
        second = refresh.refresh_once(state, "scheduled")
        assert second.status == "success"
        assert second.bundle_sha256 == initial.bundle_sha256
        assert (published / "cloud-config-bundle-cache.json").read_bytes() != first_cache
        assert second.receipt_sha256 != initial.receipt_sha256
    finally:
        _cleanup(state)


def test_bundle_and_runtime_closure_drift_never_replace_last_good_cache(
    tmp_path: Path,
) -> None:
    state, _, _ = _bootstrap(tmp_path)
    try:
        published_cache = Path(state.published_directory) / "cloud-config-bundle-cache.json"
        original = published_cache.read_bytes()
        work_codex = Path(state.work_home) / ".codex"
        (work_codex / "test-bundle").write_text("fixture-b\n", encoding="utf-8")
        drift = refresh.refresh_once(state, "scheduled")
        assert drift.status == "fatal"
        assert drift.failure == "bundle_changed"
        assert drift.promoted is False
        assert published_cache.read_bytes() == original
        drift_receipt = json.loads(Path(drift.receipt_path).read_text(encoding="utf-8"))
        assert drift_receipt["bundle_matches_initial"] is False
        assert drift_receipt["cache"]["bundle_sha256"] != state.anchor_bundle_sha256

        (work_codex / "test-bundle").unlink()
        dependency = (
            Path(state.cli.launcher_resolved_path).parent
            / "node_modules/fixture-dependency/index.js"
        )
        dependency.write_text("module.exports = 2;\n", encoding="utf-8")
        state.sequence = drift.sequence
        closure_drift = refresh.refresh_once(state, "scheduled")
        assert closure_drift.status == "fatal"
        assert closure_drift.failure == "codex_cli_identity_changed"
        assert published_cache.read_bytes() == original
    finally:
        _cleanup(state)


@pytest.mark.parametrize(
    ("marker", "value", "failure"),
    [
        ("test-lifetime", "9000", "invalid_envelope_lifetime"),
        ("test-lifetime", "100", "insufficient_envelope_ttl"),
        ("test-cached-offset", "600", "cached_at_in_future"),
    ],
)
def test_invalid_envelope_time_bounds_fail_closed(
    tmp_path: Path, marker: str, value: str, failure: str
) -> None:
    state, _, _ = _bootstrap(tmp_path)
    try:
        published_cache = Path(state.published_directory) / "cloud-config-bundle-cache.json"
        original = published_cache.read_bytes()
        (Path(state.work_home) / ".codex" / marker).write_text(value, encoding="utf-8")
        outcome = refresh.refresh_once(state, "scheduled")
        assert outcome.status == "fatal"
        assert outcome.failure == failure
        assert published_cache.read_bytes() == original
    finally:
        _cleanup(state)


def test_policy_rejects_refresh_window_that_can_cross_consumer_ttl() -> None:
    with pytest.raises(evidence.RefreshError, match="invalid_refresh_ttl_margin"):
        evidence.Policy(
            timeout_seconds=30,
            term_grace_seconds=5,
            refresh_early_seconds=600,
            minimum_ttl_seconds=630,
        ).validate()


def test_refresh_schedule_stays_ahead_of_consumer_ttl_and_command_deadline(
    tmp_path: Path,
) -> None:
    state, initial, _ = _bootstrap(tmp_path)
    try:
        receipt = json.loads(Path(initial.receipt_path).read_text(encoding="utf-8"))
        expires_at = evidence._parse_timestamp(
            json.loads(
                (Path(state.published_directory) / "cloud-config-bundle-cache.json")
                .read_text(encoding="utf-8")
            )["signed_payload"]["expires_at"]
        )
        refresh_lead = int(expires_at.timestamp()) - initial.next_refresh_epoch
        required = (
            state.policy.minimum_ttl_seconds
            + state.policy.timeout_seconds
            + state.policy.term_grace_seconds
            + evidence.MINIMUM_REFRESH_SCHEDULING_SLACK_SECONDS
        )
        assert refresh_lead == state.policy.refresh_early_seconds
        assert refresh_lead > required
        assert receipt["schema"] == "aka.formal-codex-cloud-config-refresh/v3"
        assert receipt["policy_id"] == "private_auth_only_app_server_refresh_v3"
    finally:
        _cleanup(state)


def test_short_envelope_cannot_publish_with_refresh_epoch_in_the_past(
    tmp_path: Path,
) -> None:
    state, _, _ = _bootstrap(tmp_path)
    try:
        published_cache = Path(state.published_directory) / "cloud-config-bundle-cache.json"
        original = published_cache.read_bytes()
        (Path(state.work_home) / ".codex/test-lifetime").write_text(
            "800", encoding="utf-8"
        )
        outcome = refresh.refresh_once(state, "scheduled")
        assert outcome.status == "fatal"
        assert outcome.failure == "insufficient_envelope_refresh_window"
        assert published_cache.read_bytes() == original
    finally:
        _cleanup(state)


def test_receipt_publish_is_no_clobber_concurrent_and_metadata_strict(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    auth = _write_auth(home)
    prefix, _ = _write_fake_runtime(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    owner_pid, owner_starttime = _owner_identity()
    state = evidence.prepare_state(
        auth,
        prefix,
        data,
        owner_pid,
        owner_starttime,
        policy=_policy(),
    )
    material = {"schema": "fixture/v1", "status": "success"}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda _index: refresh._persist_receipt(state, 1, material),
                    range(8),
                )
            )
        assert len(set(results)) == 1
        _, path_text = results[0]
        path = Path(path_text)
        assert stat.S_IMODE(path.stat().st_mode) == 0o444

        path.chmod(0o600)
        with pytest.raises(evidence.RefreshError, match="unsafe_published_receipt"):
            refresh._persist_receipt(state, 1, material)

        collision_material = {"schema": "fixture/v1", "status": "fatal"}
        digest = hashlib.sha256(evidence._canonical_bytes(collision_material)).hexdigest()
        collision = data / f"codex-cloud-config-refresh-000002-{digest}.json"
        collision.write_text("different\n", encoding="utf-8")
        collision.chmod(0o444)
        with pytest.raises(evidence.RefreshError, match="receipt_digest_collision"):
            refresh._persist_receipt(state, 2, collision_material)
    finally:
        _cleanup(state)


def test_scheduled_failure_receipts_fatal_and_terms_exact_owner(
    tmp_path: Path,
) -> None:
    owner = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    state: evidence.RefreshState | None = None
    supervisor: subprocess.Popen[bytes] | None = None
    try:
        owner_starttime = None
        for _ in range(100):
            owner_starttime = evidence._process_starttime(owner.pid)
            if owner_starttime is not None:
                break
            time.sleep(0.01)
        assert owner_starttime is not None
        home = tmp_path / "home"
        auth = _write_auth(home)
        prefix, _ = _write_fake_runtime(tmp_path)
        data = tmp_path / "data"
        data.mkdir()
        state, _ = refresh.bootstrap(
            auth,
            prefix,
            data,
            owner.pid,
            owner_starttime,
            policy=_policy(),
        )
        (Path(state.work_home) / ".codex/test-mode").write_text(
            "nonzero\n", encoding="utf-8"
        )
        state.next_refresh_epoch = int(time.time()) + 1
        evidence._write_state(state)
        helper = Path(refresh.__file__).resolve()
        supervisor = subprocess.Popen(
            [
                sys.executable,
                str(helper),
                "supervise",
                "--root",
                state.root,
                "--device",
                str(state.root_device),
                "--inode",
                str(state.root_inode),
            ]
        )
        assert supervisor.wait(timeout=8) == 1
        assert owner.wait(timeout=5) == -15
        fatal = (Path(state.root) / "fatal").read_text(encoding="ascii").strip()
        assert len(fatal) == 64
        receipts = sorted(data.glob("codex-cloud-config-refresh-*.json"))
        assert len(receipts) >= 2
        assert any(
            json.loads(path.read_text(encoding="utf-8"))["failure"]
            == "command_failed"
            for path in receipts
        )
    finally:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=5)
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)
        if state is not None:
            _cleanup(state)
