import logging
import importlib
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main as aka_main
from src import campaign
from src import campaign_isolation
from src import evaluator
from main import claim_next_descriptor, initialize_parallel_queue


codex_launcher = importlib.import_module("agents.codex.launch_agent")


def _policy() -> dict:
    return {
        "comparison": "apex_vs_codex",
        "attempts": 3,
        "attempt_timeout_seconds": 3600,
        "apex_internal_allowance_seconds": 3600,
        "task_timeout_seconds": 25200,
        "evaluator_allowance_seconds": 3600,
        "selection_policy": "correctness_then_measured_rate_v1",
        "workspace_policy": "fresh_per_attempt",
        "gpu_policy": "deterministic_task_gpu_v1",
        "require_clean_checkouts": True,
    }


def _runtime_isolation_receipt(*, yama_ptrace_scope: int = 1) -> dict:
    return {
        "schema": "aka.runtime-isolation-receipt/v1",
        "policy": {
            "docker_capabilities": "drop_all",
            "docker_no_new_privileges": True,
            "proc_escape_guard": "yama_ptrace_scope_and_live_parent_root_fd_probe_v1",
        },
        "outer_runtime": {
            "effective_uid": 1000,
            "effective_gid": 1000,
            "supplementary_gids": [44, 109],
            "capabilities": {
                "CapInh": 0,
                "CapPrm": 0,
                "CapEff": 0,
                "CapBnd": 0,
                "CapAmb": 0,
            },
            "no_new_privileges": True,
            "seccomp_mode": 0,
            "seccomp_filters": 0,
            "apparmor_profile": "unconfined",
            "yama_ptrace_scope": yama_ptrace_scope,
        },
        "bubblewrap": {
            "resolved_path": "/usr/bin/bwrap",
            "sha256": "8" * 64,
            "version": "bubblewrap 1.0",
        },
        "attempt_probe": {
            "campaign_data_hidden": True,
            "parent_process_visible_in_inherited_proc": True,
            "parent_root_escape_blocked": True,
            "parent_fd_escape_blocked": True,
            "proc_mount_read_only": True,
            "pid_namespace_unshared": True,
            "ipc_namespace_unshared": True,
            "private_shm": True,
            "no_new_privileges": True,
            "effective_capabilities_zero": True,
            "bounding_capabilities_zero": True,
            "all_capability_sets_zero": True,
            "seccomp_disabled": True,
        },
    }


def test_runtime_isolation_receipt_records_only_stable_namespace_evidence(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(tmp_path))
    outer = {
        **_runtime_isolation_receipt()["outer_runtime"],
        "pid_namespace": "pid:[101]",
        "ipc_namespace": "ipc:[202]",
    }
    probe = _runtime_isolation_receipt()["attempt_probe"]
    monkeypatch.setattr(
        campaign_isolation, "_outer_runtime_observation", lambda: outer
    )
    monkeypatch.setattr(
        campaign_isolation,
        "_bubblewrap_identity",
        lambda: (
            Path("/usr/bin/bwrap"),
            _runtime_isolation_receipt()["bubblewrap"],
        ),
    )
    observed: dict[str, object] = {}

    def fake_probe(*, binary, data_root, outer):
        observed.update(binary=binary, data_root=data_root, outer=outer)
        return probe

    monkeypatch.setattr(campaign_isolation, "_attempt_escape_probe", fake_probe)

    receipt = campaign_isolation.runtime_isolation_receipt()

    assert receipt["outer_runtime"] == _runtime_isolation_receipt()["outer_runtime"]
    assert "pid_namespace" not in receipt["outer_runtime"]
    assert "ipc_namespace" not in receipt["outer_runtime"]
    assert observed["outer"] is outer
    assert receipt["attempt_probe"]["parent_root_escape_blocked"] is True


def test_attempt_escape_probe_rejects_non_policy_proc_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(tmp_path))
    outer = {
        "pid_namespace": "pid:[101]",
        "ipc_namespace": "ipc:[202]",
    }

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                **_runtime_isolation_receipt()["attempt_probe"],
                "parent_fd_escape_blocked": False,
            }
        )

    monkeypatch.setattr(
        campaign_isolation.subprocess, "run", lambda *_args, **_kwargs: Completed()
    )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="isolation proof is incomplete",
    ):
        campaign_isolation._attempt_escape_probe(
            binary=Path("/usr/bin/bwrap"), data_root=tmp_path, outer=outer
        )


def test_outer_runtime_isolation_fails_closed_on_yama_or_capabilities(monkeypatch) -> None:
    status = {
        "CapInh": "0",
        "CapPrm": "0",
        "CapEff": "0",
        "CapBnd": "0",
        "CapAmb": "0",
        "NoNewPrivs": "1",
        "Seccomp": "0",
        "Seccomp_filters": "0",
    }
    monkeypatch.setattr(campaign_isolation, "_proc_status", lambda: dict(status))
    monkeypatch.setattr(campaign_isolation.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(campaign_isolation.os, "getegid", lambda: 1000)
    monkeypatch.setattr(campaign_isolation.os, "getgroups", lambda: [109, 44, 109])
    monkeypatch.setattr(
        campaign_isolation.os, "readlink", lambda path: f"namespace:{path}"
    )

    real_read_text = Path.read_text

    def unsafe_yama(path, *args, **kwargs):
        if str(path) == "/proc/self/attr/current":
            return "unconfined\n"
        if str(path) == "/proc/sys/kernel/yama/ptrace_scope":
            return "0\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unsafe_yama)
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="ptrace_scope"):
        campaign_isolation._outer_runtime_observation()

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "unconfined\n"
            if str(path) == "/proc/self/attr/current"
            else "1\n"
            if str(path) == "/proc/sys/kernel/yama/ptrace_scope"
            else real_read_text(path, *args, **kwargs)
        ),
    )
    status["CapBnd"] = "1"
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="CapBnd"):
        campaign_isolation._outer_runtime_observation()

    status["CapBnd"] = "0"
    status["NoNewPrivs"] = "0"
    with pytest.raises(
        campaign_isolation.CampaignIsolationError, match="no-new-privileges"
    ):
        campaign_isolation._outer_runtime_observation()

    status["NoNewPrivs"] = "1"
    status["Seccomp_filters"] = "1"
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="seccomp"):
        campaign_isolation._outer_runtime_observation()

    status["Seccomp_filters"] = "0"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "docker-default (enforce)\n"
            if str(path) == "/proc/self/attr/current"
            else "1\n"
            if str(path) == "/proc/sys/kernel/yama/ptrace_scope"
            else real_read_text(path, *args, **kwargs)
        ),
    )
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="AppArmor"):
        campaign_isolation._outer_runtime_observation()


def _write_result(workspace: Path, optimized_ms: float, *, correct: bool = True) -> None:
    case = {
        "test_case_id": "case-1",
        "params": {"n": 64},
        "benchmark_method": "cuda_event",
        "benchmark_samples": 100,
        "benchmark_effective_repeats": 100,
    }
    (workspace / "baseline_perf.yaml").write_text(
        yaml.safe_dump({"test_cases": [{**case, "execution_time_ms": 2.0}]}),
        encoding="utf-8",
    )
    (workspace / "optimized_perf.yaml").write_text(
        yaml.safe_dump({"test_cases": [{**case, "execution_time_ms": optimized_ms}]}),
        encoding="utf-8",
    )
    payload = {
        "task_name": "triton2triton/vllm/example",
        "pass_compilation": True,
        "pass_correctness": correct,
        "base_execution_time": 2.0,
        "best_optimized_execution_time": optimized_ms,
        "speedup_ratio": 2.0 / optimized_ms if optimized_ms else 0.0,
        "valid_optimized_cases": 1 if optimized_ms else 0,
        "valid_baseline_cases": 1,
        "baseline_benchmark_methods": ["cuda_event"],
        "optimized_benchmark_methods": ["cuda_event"],
        "benchmark_method_consistent": True,
        "speedup_calculation_error_message": None,
    }
    (workspace / "task_result.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def test_formal_attempt_mount_keeps_workspace_read_only_and_artifacts_private(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign-data"
    attempt = data_root / "run/task/attempt_01"
    sibling = data_root / "run/task/attempt_02"
    workspace = attempt / "workspace"
    artifact_root = attempt / ".workspace_apex/run-id"
    attempt_home = attempt / ".agent-home"
    workspace.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    attempt_home.mkdir(parents=True)
    sibling.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline\n", encoding="utf-8")
    (sibling / "secret.txt").write_text("prior attempt\n", encoding="utf-8")
    state_root = tmp_path / "agent-state"
    state_root.mkdir()
    (state_root / "history.jsonl").write_text("prior context\n", encoding="utf-8")
    shm_sentinel = Path("/dev/shm") / f"aka-ipc-sentinel-{tmp_path.name}"
    shm_sentinel.write_text("host-visible\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root))
    command = campaign_isolation.wrap_attempt_command(
        [
            "/bin/sh",
            "-c",
            (
                f"test ! -e {sibling} && "
                f"test ! -e {state_root / 'history.jsonl'} && "
                f"test ! -e {shm_sentinel} && "
                f"test \"$(cat {source})\" = baseline && "
                f"! printf changed > {source} && "
                f"printf artifact > {artifact_root / 'probe.txt'} && "
                f"printf home > {attempt_home / 'probe.txt'}"
            ),
        ],
        eval_config={
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {"fresh_session": True},
        },
        writable_roots=(artifact_root, attempt_home),
        read_only_roots=(workspace,),
    )
    proc_index = command.index("/proc")
    assert command[proc_index - 1] == "--ro-bind"
    assert "--proc" not in command

    try:
        completed = campaign.subprocess.run(command, capture_output=True, text=True)

        assert completed.returncode == 0, completed.stderr
        assert source.read_text(encoding="utf-8") == "baseline\n"
        assert (artifact_root / "probe.txt").read_text(encoding="utf-8") == "artifact"
        assert (attempt_home / "probe.txt").read_text(encoding="utf-8") == "home"
        assert (sibling / "secret.txt").read_text(encoding="utf-8") == "prior attempt\n"
        assert shm_sentinel.read_text(encoding="utf-8") == "host-visible\n"
    finally:
        shm_sentinel.unlink(missing_ok=True)


def test_formal_codex_home_copies_auth_only(tmp_path, monkeypatch) -> None:
    state = tmp_path / "agent-state/.codex"
    state.mkdir(parents=True)
    (state / "auth.json").write_text('{"token":"fixture"}\n', encoding="utf-8")
    (state / "history.jsonl").write_text("prior context\n", encoding="utf-8")
    (state / "config.toml").write_text("model = 'wrong'\n", encoding="utf-8")
    attempt = tmp_path / "campaign/run/attempt_01"
    attempt.mkdir(parents=True)
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state.parent))
    home = campaign_isolation.prepare_attempt_home(
        {
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "receipt_path": str(attempt / "session_receipt.json"),
            },
        },
        backend="codex",
    )

    assert home is not None
    assert (home / ".codex/auth.json").is_file()
    assert not (home / ".codex/history.jsonl").exists()
    assert not (home / ".codex/config.toml").exists()


def _write_campaign_codex_contract(run_directory: Path) -> dict:
    codex = {
        "model": "gpt-5.5",
        "effort": "xhigh",
        "codex_version": "codex-cli test",
        "codex_binary_sha256": "a" * 64,
        "max_turns": 25,
        "turn_policy": "structured_agent_turn_v1",
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
        },
    }
    gpu = {
        "gpu_boundary_plan_sha256": "d" * 64,
        "devices": [
            {
                "host_device_id": "0",
                "unique_id": "0x0000000000000001",
                "render_nodes": ["/dev/dri/renderD128"],
            }
        ],
        "exclusivity": {
            "sha256": "e" * 64,
            "exclusivity_verified": True,
        },
    }
    comparison_contract = {"codex": codex, "runtime": {"gpu": gpu}}
    comparison_digest = hashlib.sha256(
        json.dumps(
            comparison_contract, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    path = run_directory / "campaign_manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "comparison_contract_sha256": comparison_digest,
                "comparison_contract": comparison_contract,
                "runtime": {"gpu": gpu},
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o444)
    return codex


def _write_valid_codex_receipt(receipt_path: Path, codex_contract: dict) -> Path:
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    artifact_dir.mkdir(parents=True)
    source_metadata = {"sha256": "1" * 64, "size_bytes": 1, "mode": "0644"}
    before_manifest = {"kernel.py": source_metadata}
    after_manifest = {"kernel.py": source_metadata}
    artifact_payloads = {
        "raw_stdout": (artifact_dir / "raw_stdout.jsonl", b"{}\n"),
        "raw_stderr": (artifact_dir / "raw_stderr.txt", b""),
        "formatted_transcript": (
            artifact_dir / "formatted_transcript.txt",
            b"assistant: done\n",
        ),
        "workspace_before_manifest": (
            artifact_dir / "workspace_before_manifest.json",
            json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
        ),
        "workspace_after_manifest": (
            artifact_dir / "workspace_after_manifest.json",
            json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
        ),
    }
    artifacts = {}
    for name, (path, payload) in artifact_payloads.items():
        path.write_bytes(payload)
        path.chmod(0o444)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "mode": "0444",
        }
    artifact_dir.chmod(0o555)
    receipt = {
        "schema": "agentkernelarena.codex-attempt-receipt/v1",
        "session_succeeded": True,
        "thread_id": "thread-test",
        "session_id": "session-test",
        "exit_code": 0,
        "timed_out": False,
        "effective_timeout_seconds": 3599.0,
        "process_group_cleanup": {
            "verification_performed": True,
            "verified_absent": True,
        },
        "capture": {
            "readers_completed": True,
            "errors": [],
            "stdout": {
                "limit_bytes": 16 * 1024 * 1024,
                "retained_bytes": 3,
                "discarded_bytes": 0,
                "truncated": False,
            },
            "stderr": {
                "limit_bytes": 16 * 1024 * 1024,
                "retained_bytes": 0,
                "discarded_bytes": 0,
                "truncated": False,
            },
        },
        "turn_budget": {
            "policy": "structured_agent_turn_v1",
            "max_turns": 25,
            "observed_turns": 1,
            "budget_exceeded": False,
            "enforcement_failed": False,
            "stop_reason": None,
        },
        "workspace_integrity": {
            "policy": "declared_source_only_sanitized_v1",
            "editable_files": ["kernel.py"],
            "raw_after_manifest_sha256": hashlib.sha256(
                json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_manifest_error": None,
            "raw_changes": {
                "before_manifest_sha256": hashlib.sha256(
                    json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "after_manifest_sha256": hashlib.sha256(
                    json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "created_files": [],
                "deleted_files": [],
                "changed_files": [],
                "unauthorized_changed_files": [],
                "editable_mode_changes": [],
            },
            "sanitization": {
                "performed": True,
                "candidate_retained": True,
                "baseline_restored": True,
            },
            "final_changes": {
                "before_manifest_sha256": hashlib.sha256(
                    json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "after_manifest_sha256": hashlib.sha256(
                    json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "created_files": [],
                "deleted_files": [],
                "changed_files": [],
                "unauthorized_changed_files": [],
                "editable_mode_changes": [],
            },
            "errors": [],
            "passed": True,
        },
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "d" * 64,
            "boundary_receipt_sha256": "f" * 64,
            "exclusivity_receipt_sha256": "e" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": "0",
            "unique_id": "0x0000000000000001",
            "allowed_render_nodes": ["/dev/dri/renderD128"],
            "observed_devices": [],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": "0x0000000000000001"},
                "torch": {"device_count": 1},
            },
        },
        "codex": {
            "binary_sha256": codex_contract["codex_binary_sha256"],
            "version": codex_contract["codex_version"],
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
        },
        "invocation": {
            "argv_without_prompt": [
                "codex",
                "exec",
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
            ],
            "prompt_sha256": "b" * 64,
            "workspace": str(receipt_path.parent / "workspace"),
            "editable_files": ["kernel.py"],
            "max_turns": 25,
            "turn_policy": "structured_agent_turn_v1",
            "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
            "isolation": codex_contract["isolation"],
        },
        "aggregated_usage": {"events": 1, "input_tokens": 1, "output_tokens": 1},
        "artifacts": artifacts,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)
    return artifact_payloads["raw_stdout"][0]


def _write_valid_apex_receipt(receipt_path: Path, codex_contract: dict) -> None:
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    artifact_dir.mkdir(parents=True)
    baseline_hashes = {"source/kernel.py": "1" * 64}
    invocation = {
        "schema": "apex.agent-invocation/v1",
        "cli_name": "codex",
        "cli_version": codex_contract["codex_version"],
        "executable_path": "/usr/bin/codex",
        "resolved_executable_path": "/usr/bin/codex",
        "entrypoint_sha256": codex_contract["codex_binary_sha256"],
        "argv": [
            "/usr/bin/codex",
            "exec",
            "--strict-config",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
        ],
        "workspace": "/attempt/apex-workspace",
        "prompt_transport": "stdin",
        "requested_allowed_files": ["source/kernel.py"],
        "allowed_files_enforced_by_cli": False,
        "max_turns": 25,
        "turn_policy": "structured_agent_turn_v1",
        "isolation": {
            key: value
            for key, value in codex_contract["isolation"].items()
            if key != "mount_scope"
        }
        | {"response_token_limit": "not_supported_context_advisory_only"},
    }
    transcript = {
        "schema": "apex.agent-transcript/v1",
        "backend": "codex",
        "model": codex_contract["model"],
        "effort": codex_contract["effort"],
        "invocation": invocation,
        "budget": {"exceeded": False, "enforcement_failed": False},
        "events": [],
        "semantic_events": [],
        "usage": None,
        "cost": None,
    }
    agent_payload = {
        "backend": "codex",
        "model": codex_contract["model"],
        "effort": codex_contract["effort"],
        "exit_code": 0,
        "timed_out": False,
        "budget_exceeded": False,
        "budget_enforcement_failed": False,
        "invocation": invocation,
        "artifacts": [],
    }
    event_values = [
        (1, "evt-agent", "agent_completed", agent_payload, None, "txn-agent"),
        (2, "evt-decision", "decision", {"verdict": "revert"}, "evt-agent", "txn-decision"),
    ]
    events = []
    for sequence, event_id, event_type, payload, parent, transaction_id in event_values:
        material = {
            "sequence": sequence,
            "event_id": event_id,
            "run_id": "run-test",
            "event_type": event_type,
            "payload": payload,
            "parent_event_id": parent,
            "idempotency_key": f"key-{sequence}",
            "transaction_id": transaction_id,
            "created_at_ns": sequence,
        }
        events.append(material | {"checksum": campaign._canonical_json_digest(material)})
    journal = artifact_dir / "event_journal.sqlite"
    connection = sqlite3.connect(journal)
    try:
        connection.executescript(
            """
            CREATE TABLE transactions (
              transaction_id TEXT PRIMARY KEY, first_sequence INTEGER,
              last_sequence INTEGER, event_count INTEGER, checksum TEXT
            );
            CREATE TABLE events (
              sequence INTEGER PRIMARY KEY, event_id TEXT, run_id TEXT,
              event_type TEXT, payload_json TEXT, parent_event_id TEXT,
              idempotency_key TEXT, transaction_id TEXT, created_at_ns INTEGER,
              checksum TEXT
            );
            """
        )
        for event in events:
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event["sequence"],
                    event["event_id"],
                    event["run_id"],
                    event["event_type"],
                    json.dumps(
                        event["payload"], sort_keys=True, separators=(",", ":")
                    ),
                    event["parent_event_id"],
                    event["idempotency_key"],
                    event["transaction_id"],
                    event["created_at_ns"],
                    event["checksum"],
                ),
            )
            tx_checksum = campaign._canonical_json_digest(
                {
                    "transaction_id": event["transaction_id"],
                    "event_checksums": [event["checksum"]],
                }
            )
            connection.execute(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
                (
                    event["transaction_id"],
                    event["sequence"],
                    event["sequence"],
                    1,
                    tx_checksum,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    task_spec = {
        "task_id": "task-test",
        "agent_backend": "codex",
        "agent_options": {
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
        },
        "budget": {"max_turns": 25, "timeout_seconds": 3600},
        "baseline": {"file_hashes": baseline_hashes},
    }
    result = {
        "schema_version": 1,
        "run_id": "run-test",
        "task_id": "task-test",
        "status": "no_gain",
        "reason_code": "baseline_is_best",
        "applied": False,
        "external_verification_required": True,
        "bundle_path": None,
        "bundle_digest": None,
        "changed_files": [],
        "baseline_lock": {
            "resolution_hash": "2" * 64,
            "file_hashes": baseline_hashes,
        },
        "internal_verdict": "revert",
        "internal_verdict_ref": "evt-decision",
        "event_journal_ref": {
            "path": "/attempt/events.sqlite",
            "head_event_id": events[-1]["event_id"],
            "head_checksum": events[-1]["checksum"],
        },
        "artifact_store_ref": {"path": "/attempt/artifacts", "receipt_digests": []},
        "error": None,
    }
    payloads = {
        "task_spec": ("task_spec.json", json.dumps(task_spec).encode()),
        "apex_stdout": ("apex_stdout.txt", b"done\n"),
        "apex_stderr": ("apex_stderr.txt", b""),
        "apex_result": ("apex_result.json", json.dumps(result).encode()),
        "event_journal": ("event_journal.sqlite", journal.read_bytes()),
        "agent_transcript": (
            "agent_transcript.json",
            json.dumps(transcript).encode(),
        ),
    }
    artifacts = {}
    for name, (filename, content) in payloads.items():
        path = artifact_dir / filename
        if name != "event_journal":
            path.write_bytes(content)
        path.chmod(0o444)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "mode": "0444",
        }
    artifact_dir.chmod(0o555)
    receipt = {
        "schema": "agentkernelarena.apex-attempt-receipt/v1",
        "session_succeeded": True,
        "terminal_status": "no_gain",
        "exit_code": 0,
        "timed_out": False,
        "budgets": {
            "inner_agent_timeout_seconds": 3600,
            "apex_internal_allowance_seconds": 3600,
            "outer_timeout_seconds": 7200.0,
            "effective_outer_timeout_seconds": 7200.0,
        },
        "process_group_cleanup": {
            "verification_performed": True,
            "verified_absent": True,
        },
        "capture": {"readers_completed": True, "errors": []},
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "d" * 64,
            "boundary_receipt_sha256": "f" * 64,
            "exclusivity_receipt_sha256": "e" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": "0",
            "unique_id": "0x0000000000000001",
            "allowed_render_nodes": ["/dev/dri/renderD128"],
            "observed_devices": [],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": "0x0000000000000001"},
                "torch": {"device_count": 1},
            },
        },
        "apex": {},
        "task_spec_sha256": artifacts["task_spec"]["sha256"],
        "outer_isolation": codex_contract["isolation"],
        "workspace_integrity": {
            "policy": "read_only_until_adapter_bundle_apply_v1",
            "baseline_manifest_sha256": "9" * 64,
            "pre_apply_manifest_sha256": "9" * 64,
            "pre_apply_unchanged": True,
        },
        "codex": {
            "binary_sha256": codex_contract["codex_binary_sha256"],
            "version": codex_contract["codex_version"],
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
        },
        "invocation": invocation,
        "lineage": {
            "run_id": "run-test",
            "result_sha256": artifacts["apex_result"]["sha256"],
            "journal_head_event_id": events[-1]["event_id"],
            "journal_head_checksum": events[-1]["checksum"],
            "event_count": 2,
            "transcript_sha256": artifacts["agent_transcript"]["sha256"],
            "event_artifact_digests": [],
            "internal_verdict": "revert",
            "internal_verdict_ref": "evt-decision",
        },
        "artifacts": artifacts,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)


def test_three_fresh_sessions_are_centrally_ranked_with_stable_tie_break(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _write_campaign_codex_contract(run_directory)
    observed: list[Path] = []
    times = [2.0, 1.0, 1.0]

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        (workspace / "solution.py").write_text(f"attempt = {attempt}\n", encoding="utf-8")
        _write_result(workspace, times[attempt - 1])
        observed.append(workspace)
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=str(tmp_path / "config.yaml"),
        run_directory=run_directory,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is True
    assert canonical is not None
    assert len(observed) == 3
    assert len({workspace.parent for workspace in observed}) == 3
    assert (canonical / "solution.py").read_text(encoding="utf-8") == "attempt = 2\n"
    selected = yaml.safe_load((canonical / "task_result.yaml").read_text())
    assert selected["campaign_evidence"]["selected_attempt"] == 2
    assert selected["campaign_evidence"]["is_apex_canonical_300_sample_grade"] is False
    attempts = yaml.safe_load(
        (
            run_directory
            / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml"
        ).read_text()
    )
    assert attempts["all_attempts_centrally_evaluated"] is True
    assert [record["attempt"] for record in attempts["attempts"]] == [1, 2, 3]


def test_missing_central_report_is_retained_and_invalidates_campaign(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _write_campaign_codex_contract(run_directory)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        (workspace / "solution.py").write_text(f"attempt = {attempt}\n", encoding="utf-8")
        if attempt != 2:
            _write_result(workspace, float(attempt))
        return attempt != 2, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=str(tmp_path / "config.yaml"),
        run_directory=run_directory,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (
            run_directory
            / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml"
        ).read_text()
    )
    assert evidence["attempts"][1]["central_evaluator_report"] is None
    assert evidence["attempts"][1]["workspace"].endswith(
        "attempt_02/triton2triton_vllm_example_20260807_000000"
    )


def test_manifest_names_native_100_repetition_score_not_apex_grade(
    tmp_path, monkeypatch
) -> None:
    run_config = tmp_path / "run.yaml"
    task_config = tmp_path / "task.yaml"
    run_config.write_text("campaign: {}\n", encoding="utf-8")
    task_config.write_text("task_type: triton2triton\n", encoding="utf-8")
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "example@sha256:one")
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS",
        '["example@sha256:' + "2" * 64 + '"]',
    )
    monkeypatch.setattr(
        campaign,
        "_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False, "status_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_apex_state_from_environment",
        lambda: {"commit": "c" * 40, "dirty": False, "status_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_agent_manifest",
        lambda _root, agent, _policy_value: {
            "template": agent,
            "backend": "codex",
            "model": "gpt-5.5",
            "effort": "xhigh",
            "permission_mode": "workspace_write_isolated",
            "inner_max_iterations": 1,
            "attempt_timeout_seconds": 3600,
            "max_turns": 25,
            "turn_policy": "structured_agent_turn_v1",
            "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
            "codex_version": "codex 1.0",
            "codex_binary_sha256": "e" * 64,
            "isolation": {
                "approval": "never_via_strict_config",
                "execpolicy_rules": "ignored",
                "project_instructions": "backend_default_may_load",
                "sandbox": "workspace-write",
                "session": "ephemeral",
                "user_config": "ignored",
                "mount_scope": "attempt_only_bubblewrap",
            },
        },
    )
    monkeypatch.setattr(
        campaign,
        "_gpu_inventory",
        lambda _config, _tasks: {
            "ordered_host_gpu_ids": ["0"],
            "task_mapping": [{"task_name": "example", "assigned_host_gpu_id": "0"}],
        },
    )
    monkeypatch.setattr(campaign, "_evaluator_manifest", lambda _root: {"main.py": "f" * 64})
    monkeypatch.setattr(
        campaign, "runtime_isolation_receipt", _runtime_isolation_receipt
    )

    manifest = campaign.build_campaign_manifest(
        eval_config={"campaign": _policy()},
        run_config_path=run_config,
        task_config_paths={"example": str(task_config)},
        agent_name="codex",
    )

    assert manifest is not None
    assert manifest["measurement"]["configured_repetitions_per_test_case"] == 100
    assert manifest["measurement"]["is_apex_canonical_300_sample_grade"] is False
    assert manifest["configuration"]["tasks"][0]["config_sha256"]
    assert len(manifest["comparison_contract_sha256"]) == 64


def test_outer_timeout_must_cover_attempts_and_evaluator() -> None:
    policy = _policy()
    policy["task_timeout_seconds"] = 10800
    try:
        campaign.parse_campaign_policy({"campaign": policy})
    except campaign.CampaignError as error:
        assert "evaluator allowance" in str(error)
    else:
        raise AssertionError("undersized task timeout must be rejected")


def test_formal_manifest_rejects_dirty_agent_kernel_arena_checkout(
    tmp_path, monkeypatch
) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text("campaign: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": True, "status_sha256": "b" * 64},
    )

    with pytest.raises(campaign.CampaignError, match="clean AgentKernelArena"):
        campaign.build_campaign_manifest(
            eval_config={"campaign": _policy()},
            run_config_path=run_config,
            task_config_paths={},
            agent_name="codex",
        )


def test_campaign_manifest_is_published_read_only(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    comparison_contract = {
        "runtime": {"isolation": _runtime_isolation_receipt()}
    }

    def current_manifest(**_kwargs):
        comparison = json.loads(json.dumps(comparison_contract))
        digest = hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "comparison_contract_sha256": digest,
            "comparison_contract": comparison,
        }

    monkeypatch.setattr(
        campaign,
        "build_campaign_manifest",
        current_manifest,
    )
    config = tmp_path / "config.yaml"
    config.write_text("campaign: {}\n", encoding="utf-8")

    path = campaign.ensure_campaign_manifest(
        run_directory=run,
        eval_config={"campaign": _policy()},
        run_config_path=config,
        task_config_paths={},
        agent_name="codex",
    )

    assert path is not None
    assert path.stat().st_mode & 0o222 == 0

    comparison_contract["runtime"]["isolation"] = _runtime_isolation_receipt(
        yama_ptrace_scope=2
    )
    with pytest.raises(campaign.CampaignError, match="provenance changed"):
        campaign.ensure_campaign_manifest(
            run_directory=run,
            eval_config={"campaign": _policy()},
            run_config_path=config,
            task_config_paths={},
            agent_name="codex",
        )


def test_formal_image_manifest_requires_nonempty_daemon_repo_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "image:tag")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "[]")

    with pytest.raises(campaign.CampaignError, match="repo digest"):
        campaign._image_manifest()


def test_codex_campaign_inner_iteration_override_does_not_change_default() -> None:
    agent_config = {"max_iterations": 3, "campaign_max_iterations": 1}

    normal = codex_launcher._prompt_agent_config(agent_config, {})
    matched = codex_launcher._prompt_agent_config(
        agent_config, {"campaign": {"comparison": "apex_vs_codex"}}
    )

    assert normal["max_iterations"] == 3
    assert matched["max_iterations"] == 1
    assert agent_config["max_iterations"] == 3


def test_deterministic_gpu_mapping_and_worker_affinity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1,2,3,4,5,6,7")
    run = tmp_path / "run"
    run.mkdir()
    tasks = {f"task/{index}": f"/task/{index}/config.yaml" for index in range(10)}
    context = {
        "run_directory": run,
        "task_config_dict": tasks,
        "timestamp": "20260807_000000",
        "agent": SimpleNamespace(value="codex"),
        "logger": logging.getLogger(__name__),
        "config": {"campaign": _policy()},
    }

    initialize_parallel_queue(context)

    pending = sorted((run / ".parallel/pending").glob("*.yaml"))
    assignments = [yaml.safe_load(path.read_text())["assigned_host_gpu_id"] for path in pending]
    assert assignments == ["0", "1", "2", "3", "4", "5", "6", "7", "0", "1"]
    claimed = claim_next_descriptor(run, "gpu7", logging.getLogger(__name__), "7")
    assert claimed is not None
    assert yaml.safe_load(claimed.read_text())["assigned_host_gpu_id"] == "7"


def test_gpu_inventory_binds_every_pool_device_and_task_mapping(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_ARCH", "gfx950")
    plan_digest = "a" * 64
    plan_path = "/run/agentkernelarena/formal-gpu-boundary-plan.json"
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN", plan_path)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256", plan_digest)
    exclusivity_digest = "b" * 64
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT", "/run/aka-gpu-lease.json"
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT_SHA256", exclusivity_digest
    )
    plan = {
        "sha256": plan_digest,
        "kfd_device": {"path": "/dev/kfd", "major": 235, "minor": 0},
        "devices": [
            {
                "host_gpu_id": str(index),
                "unique_id": f"0x{index + 1:016x}",
                "serial_number": f"SERIAL-{index}",
                "card_series": "AMD Instinct MI355X",
                "gfx_version": "gfx950",
                "render_nodes": [{"path": f"/dev/dri/renderD{128 + index}"}],
            }
            for index in range(8)
        ],
    }
    monkeypatch.setattr(campaign, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        campaign,
        "load_gpu_lease_receipt",
        lambda *_args, **_kwargs: {
            "sha256": exclusivity_digest,
            "exclusivity_verified": True,
        },
    )
    tasks = [f"task/{index}" for index in range(10)]

    inventory = campaign._gpu_inventory({"target_gpu_model": "MI355X"}, tasks)

    assert [device["host_device_id"] for device in inventory["devices"]] == [
        str(index) for index in range(8)
    ]
    assert [item["assigned_host_gpu_id"] for item in inventory["task_mapping"]] == [
        "0", "1", "2", "3", "4", "5", "6", "7", "0", "1"
    ]
    assert inventory["gpu_boundary_plan_sha256"] == plan_digest
    assert inventory["exclusivity"]["exclusivity_verified"] is True


def test_comparison_contract_hash_ignores_treatment_template(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    task_config = task_root / "config.yaml"
    task_config.write_text("task_type: triton2triton\n", encoding="utf-8")
    apex_run = tmp_path / "apex-run.yaml"
    codex_run = tmp_path / "codex-run.yaml"
    apex_run.write_text("agent: {template: apex}\n", encoding="utf-8")
    codex_run.write_text("agent: {template: codex}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "image:tag")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64)
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS",
        '["image@sha256:' + "2" * 64 + '"]',
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1")
    monkeypatch.setattr(
        campaign,
        "_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False, "status_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_apex_state_from_environment",
        lambda: {"commit": "c" * 40, "dirty": False, "status_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_agent_manifest",
        lambda _root, name, _policy_value: {
            "template": name,
            "backend": "codex",
            "model": "gpt-5.5",
            "effort": "xhigh",
            "permission_mode": "workspace_write_isolated",
            "inner_max_iterations": 1,
            "attempt_timeout_seconds": 3600,
            "max_turns": 25,
            "turn_policy": "structured_agent_turn_v1",
            "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
            "codex_version": "codex 1.0",
            "codex_binary_sha256": "e" * 64,
            "isolation": {
                "approval": "never_via_strict_config",
                "execpolicy_rules": "ignored",
                "project_instructions": "backend_default_may_load",
                "sandbox": "workspace-write",
                "session": "ephemeral",
                "user_config": "ignored",
                "mount_scope": "attempt_only_bubblewrap",
            },
            "agent_config_sha256": ("1" if name == "apex" else "2") * 64,
        },
    )
    monkeypatch.setattr(campaign, "_evaluator_manifest", lambda _root: {"main.py": "f" * 64})
    monkeypatch.setattr(
        campaign, "runtime_isolation_receipt", _runtime_isolation_receipt
    )
    monkeypatch.setattr(
        campaign,
        "_gpu_inventory",
        lambda _config, names: {
            "ordered_host_gpu_ids": ["0", "1"],
            "task_mapping": campaign.deterministic_task_gpu_mapping(names),
        },
    )
    kwargs = {
        "eval_config": {"campaign": _policy()},
        "task_config_paths": {"example": str(task_config)},
    }
    apex = campaign.build_campaign_manifest(
        **kwargs, run_config_path=apex_run, agent_name="apex"
    )
    codex = campaign.build_campaign_manifest(
        **kwargs, run_config_path=codex_run, agent_name="codex"
    )
    assert apex is not None and codex is not None
    assert apex["comparison_contract_sha256"] == codex["comparison_contract_sha256"]
    assert apex["agent"]["template"] != codex["agent"]["template"]


def test_comparison_contract_projects_run_specific_gpu_lease_receipts() -> None:
    policy = campaign.CampaignPolicy(**_policy())
    agent = {
        "backend": "codex",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "permission_mode": "workspace_write_isolated",
        "inner_max_iterations": 1,
        "attempt_timeout_seconds": 3600,
        "max_turns": 25,
        "turn_policy": "structured_agent_turn_v1",
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "codex_version": "codex test",
        "codex_binary_sha256": "1" * 64,
        "isolation": {},
    }

    def runtime(run_name: str, runner_pid: int, receipt_digest: str) -> dict:
        return {
            "docker": {"image_id": "sha256:" + "2" * 64},
            "isolation": _runtime_isolation_receipt(),
            "gpu": {
                "gpu_boundary_plan_sha256": "3" * 64,
                "devices": [{"host_device_id": "0", "unique_id": "0x01"}],
                "exclusivity": {
                    "policy": "physical_unique_id_flock_plus_kfd_preflight_v1",
                    "gpu_boundary_plan_sha256": "3" * 64,
                    "run_name": run_name,
                    "runner_pid": runner_pid,
                    "observed_at_ns": runner_pid * 10,
                    "sha256": receipt_digest,
                    "protected_device_paths": ["/dev/kfd", "/dev/dri/renderD128"],
                    "leases": [
                        {"unique_id": "0x01", "lock_path": f"/tmp/{run_name}.lock"}
                    ],
                    "exclusivity_verified": True,
                },
            },
        }

    kwargs = {
        "policy": policy,
        "measurement": {},
        "repositories": {},
        "agent": agent,
        "evaluator": {},
        "tasks": [],
    }
    first = campaign._comparison_contract(
        runtime=runtime("apex-run", 101, "a" * 64), **kwargs
    )
    second = campaign._comparison_contract(
        runtime=runtime("codex-run", 202, "b" * 64), **kwargs
    )

    assert first == second
    changed = runtime("codex-run", 202, "b" * 64)
    changed["gpu"]["gpu_boundary_plan_sha256"] = "4" * 64
    assert campaign._comparison_contract(runtime=changed, **kwargs) != first
    changed_isolation = runtime("codex-run", 202, "b" * 64)
    changed_isolation["isolation"] = _runtime_isolation_receipt(
        yama_ptrace_scope=2
    )
    assert campaign._comparison_contract(runtime=changed_isolation, **kwargs) != first


def test_fake_clock_enforces_remaining_session_reservation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    now = [0.0]
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        now[0] += 12000.0
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=str(tmp_path / "config.yaml"),
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
        clock=lambda: now[0],
    )
    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml").read_text()
    )
    assert evidence["attempts"][1]["eligibility_errors"] == [
        "outer_task_deadline_cannot_cover_remaining_sessions"
    ]


def test_cumulative_central_evaluator_allowance_is_enforced(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        kwargs["eval_config"]["campaign_attempt"]["evaluation_elapsed_seconds"] = 1201.0
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=str(tmp_path / "config.yaml"),
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
        clock=lambda: 0.0,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml").read_text()
    )
    assert evidence["evaluator_elapsed_seconds"] == 3603.0
    assert evidence["within_evaluator_allowance"] is False
    assert "evaluator_allowance_exceeded" in evidence["attempts"][2][
        "eligibility_errors"
    ]


def test_evaluator_command_timeout_is_bounded_by_hard_deadline(tmp_path, monkeypatch) -> None:
    observed: list[float] = []

    def fake_run(_command, _workspace, *, timeout, **_kwargs):
        observed.append(timeout)
        return True, "", ""

    monkeypatch.setattr(evaluator, "run_command", fake_run)
    passed, error = evaluator.evaluate_compilation(
        tmp_path,
        {"compile_command": ["python compile.py"], "compile_timeout": 100},
        deadline_monotonic=15.0,
        clock=lambda: 10.0,
    )

    assert passed is True and error is None
    assert observed == [5.0]


def test_eligibility_rejects_case_mismatch_and_speedup_error(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_result(workspace, 1.0)
    optimized = yaml.safe_load((workspace / "optimized_perf.yaml").read_text())
    optimized["test_cases"][0]["test_case_id"] = "different"
    (workspace / "optimized_perf.yaml").write_text(yaml.safe_dump(optimized))
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    report["speedup_calculation_error_message"] = "mismatch"
    report["pass_correctness"] = False

    errors = campaign._evaluation_eligibility_errors(workspace, report)

    assert "baseline_optimized_testcase_set_or_order_mismatch" in errors
    assert "speedup_calculation_error" in errors
    assert "central_correctness_failed" in errors


def test_safe_projection_rejects_symlinks(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "regular.py").write_text("value = 1\n")
    (source / "escape").symlink_to("/tmp")

    try:
        campaign._safe_copy_workspace(source, tmp_path / "destination")
    except campaign.CampaignError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("unsafe workspace symlink must be rejected")


def test_direct_codex_receipt_artifacts_and_pinned_identity_are_recomputed(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    raw_stdout = _write_valid_codex_receipt(receipt_path, codex_contract)

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert receipt is not None
    assert errors == []
    (receipt_path.parent / f".{receipt_path.stem}.artifacts").chmod(0o700)

    raw_stdout.chmod(0o644)
    raw_stdout.write_text("tampered\n", encoding="utf-8")
    raw_stdout.chmod(0o444)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "direct_codex_raw_stdout_hash_mismatch" in errors
    assert "direct_codex_raw_stdout_size_mismatch" in errors
    raw_stdout.parent.chmod(0o700)


def test_apex_receipt_recomputes_result_event_invocation_and_transcript_lineage(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(receipt_path, codex_contract)

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert receipt is not None
    assert errors == []
    receipt_path.chmod(0o644)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["workspace_integrity"]["pre_apply_unchanged"] = False
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "apex_workspace_pre_apply_integrity_invalid" in errors
    (receipt_path.parent / f".{receipt_path.stem}.artifacts").chmod(0o700)


@pytest.mark.parametrize("agent_name", ["apex", "codex"])
def test_formal_agents_missing_receipts_are_diagnostic_only_and_never_complete(
    tmp_path, monkeypatch, agent_name
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=SimpleNamespace(value=agent_name),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=str(tmp_path / "config.yaml"),
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml").read_text()
    )
    assert all(
        "missing_agent_session_receipt" in attempt["eligibility_errors"]
        for attempt in evidence["attempts"]
    )


def test_three_valid_direct_codex_receipts_allow_canonical_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        _write_valid_codex_receipt(receipt_path, codex_contract)
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value="codex"),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=str(tmp_path / "config.yaml"),
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is True
        assert canonical is not None
    finally:
        for artifact_dir in run.rglob(".session_receipt.artifacts"):
            artifact_dir.chmod(0o700)


def test_three_valid_apex_receipts_allow_canonical_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        _write_valid_apex_receipt(receipt_path, codex_contract)
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value="apex"),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=str(tmp_path / "config.yaml"),
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is True
        assert canonical is not None
        evidence = yaml.safe_load(
            (
                run
                / ".campaign_attempts/triton2triton_vllm_example/task_campaign.yaml"
            ).read_text()
        )
        assert all(
            attempt["session_receipt_binding"]["schema"]
            == "agentkernelarena.apex-attempt-receipt/v1"
            for attempt in evidence["attempts"]
        )
    finally:
        for artifact_dir in run.rglob(".session_receipt.artifacts"):
            artifact_dir.chmod(0o700)


def test_failed_agent_session_still_gets_diagnostic_evaluation_but_returns_failure(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("task_type: triton2triton\n", encoding="utf-8")
    attempt = {"task_deadline_monotonic": 1000.0}

    monkeypatch.setattr(aka_main, "setup_workspace", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(
        aka_main, "evaluate_compilation", lambda *args, **kwargs: (True, None)
    )
    monkeypatch.setattr(aka_main, "measure_baseline", lambda *args, **kwargs: [])
    monkeypatch.setattr(aka_main, "snapshot_workspace_harness", lambda *_args: object())
    monkeypatch.setattr(aka_main, "verify_workspace_harness", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        aka_main, "materialize_perf_helpers_in_workspace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        aka_main,
        "evaluate_kernel",
        lambda *args, **kwargs: {
            "pass_compilation": True,
            "pass_correctness": True,
        },
    )

    def write_result(path, *_args, **_kwargs):
        _write_result(path, 1.0)

    monkeypatch.setattr(aka_main, "write_task_result", write_result)
    monkeypatch.setattr(aka_main, "is_task_complete", lambda *_args, **_kwargs: True)

    def failed_launcher(**_kwargs):
        raise RuntimeError("session failed")

    completed, observed_workspace = aka_main._run_single_task(
        eval_config={"campaign_attempt": attempt},
        agent=aka_main.AgentType.CODEX,
        agent_launcher=failed_launcher,
        task_name="triton2triton/vllm/example",
        task_config_dir=str(config_path),
        run_directory=tmp_path / "run",
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
    )

    assert completed is False
    assert observed_workspace == workspace
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert report["agent_session_succeeded"] is False
    assert report["agent_session_error_type"] == "RuntimeError"
    assert "evaluation_elapsed_seconds" in attempt
