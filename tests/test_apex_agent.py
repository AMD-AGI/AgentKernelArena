"""Offline contract tests for the Apex AgentKernelArena adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import signal
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest
import yaml

from src import apex_runtime as apex_runtime_contract
from src import aka_runtime
from src import immutable_runtime_mount
from src.agent_turn_budget import FORMAL_MATCHED_MAX_TURNS
from src import campaign_isolation
from src.campaign_isolation import WrappedAttemptCommand
from src.apex_runtime import materialize_runtime, plan_runtime
from src.module_registration import AgentType, load_agent_launcher
from src.prompt_builder import prompt_builder as render_task_prompt

apex_launcher = importlib.import_module("agents.apex.launch_agent")

_BACKEND_RUNTIME_CLOSURE_SHA256 = "e" * 64
_FORMAL_TASK_COMPONENT = apex_launcher.campaign_task_path_component("task")

_REQUESTED_MOUNT_OPTIONS = [
    "ro",
    "nodev",
    "nosuid",
    "default_permissions",
    "allow_other",
    "subtype=squashfuse",
]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _runtime_service_evidence(
    snapshot: Path,
    manifest: dict[str, object],
    mount: dict[str, object],
    image_sha256: str = "f" * 64,
) -> dict[str, object]:
    host_policy = _host_access_policy(snapshot.parent)
    image_input_sha256 = apex_runtime_contract.runtime_image_inputs(
        snapshot, manifest
    )["sha256"]
    receipt_material = {
        "schema": immutable_runtime_mount.MOUNT_RECEIPT_SCHEMA,
        "policy_id": apex_runtime_contract.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
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
        "policy_id": apex_runtime_contract.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
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
    engine = {**engine_material, "sha256": _digest(engine_material)}
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
        "engine_evidence": engine,
    }
    return {**service_material, "sha256": _digest(service_material)}


def _campaign_binding_stub() -> dict[str, object]:
    return {
        "schema": "aka.attempt-campaign-binding/v1",
        "formal_execution_sha256": "a" * 64,
        "campaign_manifest_path": "/campaign/campaign_manifest.yaml",
        "campaign_manifest_sha256": "b" * 64,
        "comparison_contract_sha256": "c" * 64,
        "backend_runtime_closure_sha256": _BACKEND_RUNTIME_CLOSURE_SHA256,
        "task_package_manifest_sha256": "d" * 64,
        "task_config_sha256": "f" * 64,
        "task_name": "task",
        "task_index": 1,
        "total_tasks": 1,
        "attempt_index": 1,
        "attempt_count": 3,
        "assigned_host_gpu_id": "0",
    }


def _install_campaign_binding(
    *,
    eval_config: dict[str, object],
    task_config_path: Path,
    manifest_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    package_files = {
        path.relative_to(task_config_path.parent).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(task_config_path.parent.rglob("*"))
        if path.is_file()
    }
    package_digest = apex_launcher._canonical_digest(package_files)
    config_digest = hashlib.sha256(task_config_path.read_bytes()).hexdigest()
    policy = {"attempts": 3}
    comparison = {
        "policy": policy,
        "codex": {
            "backend_runtime_closure_sha256": _BACKEND_RUNTIME_CLOSURE_SHA256
        },
    }
    manifest = {
        "schema": "aka.matched-campaign/v1",
        "formal_execution_sha256": "a" * 64,
        "policy": policy,
        "comparison_contract": comparison,
        "comparison_contract_sha256": apex_launcher._canonical_digest(comparison),
        "agent": {
            "backend_runtime_closure_sha256": _BACKEND_RUNTIME_CLOSURE_SHA256
        },
        "configuration": {
            "tasks": [
                {
                    "task_index": 1,
                    "task_name": "task",
                    "config_path": str(task_config_path.resolve(strict=True)),
                    "config_sha256": config_digest,
                    "package_files_sha256": package_files,
                    "package_manifest_sha256": package_digest,
                }
            ]
        },
        "runtime": {
            "gpu": {
                "task_mapping": [
                    {
                        "task_index": 1,
                        "task_name": "task",
                        "assigned_host_gpu_id": "0",
                    }
                ]
            }
        },
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (manifest_root / "campaign_manifest.yaml").resolve()
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
    )
    manifest_path.chmod(0o444)
    binding = {
        "schema": "aka.attempt-campaign-binding/v1",
        "formal_execution_sha256": manifest["formal_execution_sha256"],
        "campaign_manifest_path": str(manifest_path),
        "campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "backend_runtime_closure_sha256": _BACKEND_RUNTIME_CLOSURE_SHA256,
        "task_package_manifest_sha256": package_digest,
        "task_config_sha256": config_digest,
        "task_name": "task",
        "task_index": 1,
        "total_tasks": 1,
        "attempt_index": 1,
        "attempt_count": 3,
        "assigned_host_gpu_id": "0",
    }
    attempt = eval_config.setdefault("campaign_attempt", {})
    assert isinstance(attempt, dict)
    attempt.update(
        {
            "formal_execution_sha256": binding["formal_execution_sha256"],
            "campaign_manifest_path": binding["campaign_manifest_path"],
            "campaign_manifest_sha256": binding["campaign_manifest_sha256"],
            "comparison_contract_sha256": binding[
                "comparison_contract_sha256"
            ],
            "backend_runtime_closure_sha256": binding[
                "backend_runtime_closure_sha256"
            ],
            "task_package_manifest_sha256": binding[
                "task_package_manifest_sha256"
            ],
            "task_config_sha256": binding["task_config_sha256"],
            "task_name": "task",
            "task_index": 1,
            "total_tasks": 1,
            "index": 1,
            "count": 3,
            "assigned_host_gpu_id": "0",
        }
    )
    eval_config["assigned_host_gpu_id"] = "0"
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    return binding


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_apex_supervisor_verifies_normal_process_group_exit(tmp_path) -> None:
    outcome = apex_launcher._run_apex(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        backend="codex",
        timeout_seconds=5,
        output_limit=1024,
        logger=logging.getLogger(__name__),
    )

    assert outcome.exit_code == 0
    assert outcome.timed_out is False
    assert outcome.stdout == b"done\n"
    assert outcome.cleanup["verification_performed"] is True
    assert outcome.cleanup["verified_absent"] is True
    assert outcome.readers_completed is True
    assert outcome.capture_errors == ()


def test_apex_supervisor_releases_wrapped_fd_when_spawn_fails(
    tmp_path, monkeypatch
) -> None:
    descriptor = os.memfd_create("aka-test-apex-spawn")
    command = WrappedAttemptCommand(["/bin/true"], pass_fds=(descriptor,))

    def fail_spawn(*_args, **_kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(apex_launcher.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="spawn failed"):
        apex_launcher._run_apex(
            command,
            cwd=tmp_path,
            backend="codex",
            timeout_seconds=5,
            output_limit=1024,
            logger=logging.getLogger(__name__),
        )

    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_apex_supervisor_timeout_kills_and_verifies_process_namespace_group(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(apex_launcher, "_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(apex_launcher, "_KILL_GRACE_SECONDS", 2.0)
    code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    outcome = apex_launcher._run_apex(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        backend="codex",
        timeout_seconds=0.1,
        output_limit=1024,
        logger=logging.getLogger(__name__),
    )

    assert outcome.timed_out is True
    assert outcome.exit_code == -int(signal.SIGKILL)
    assert outcome.cleanup["sigterm_sent"] is True
    assert outcome.cleanup["sigkill_sent"] is True
    assert outcome.cleanup["verified_absent"] is True


def _task(tmp_path: Path, *, task_type: str = "triton2triton") -> tuple[Path, Path]:
    workspace = tmp_path / "run" / "task_workspace"
    source = workspace / "source" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    scripts = workspace / "scripts"
    scripts.mkdir()
    (scripts / "task_runner.py").write_text("# trusted fixture\n", encoding="utf-8")
    config = (
        tmp_path
        / "repo"
        / "tasks"
        / task_type
        / "vllm"
        / "sample_kernel"
        / "config.yaml"
    )
    _write_yaml(
        config,
        {
            "source_file_path": ["source/kernel.py"],
            "target_kernel_functions": ["sample_kernel"],
            "compile_command": ["python3 scripts/task_runner.py compile"],
            "correctness_command": ["python3 scripts/task_runner.py correctness"],
            "performance_command": ["python3 scripts/task_runner.py performance"],
            "task_type": task_type,
            "prompt": {"instructions": "Optimize the sample kernel.", "cheatsheet": None},
        },
    )
    return workspace, config


def _agent_config() -> dict[str, object]:
    return {
        "backend": "codex",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "supported_task_types": ["triton2triton"],
        "max_iterations": 3,
        "campaign_max_iterations": 1,
        "max_turns": 50,
        "timeout_seconds": 120,
        "compile_timeout_seconds": 10,
        "correctness_timeout_seconds": 20,
        "performance_timeout_seconds": 30,
    }


def _spec(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    spec = apex_launcher._build_task_spec(
        task_config_path=config_path,
        task_config=task_config,
        eval_config={"target_gpu_model": "MI355X"},
        agent_config=_agent_config(),
        workspace=workspace,
        artifact_root=artifact_root,
        prompt="BASE PROMPT",
    )
    return workspace, artifact_root, spec


def _make_bundle(
    artifact_root: Path,
    spec: dict[str, object],
    *,
    after: int = 2,
    changed_files: list[str] | None = None,
) -> tuple[Path, str, dict[str, object]]:
    bundle = artifact_root / "bundle"
    bundle.mkdir()
    patch = bundle / "candidate.patch"
    patch.write_text(
        "--- a/source/kernel.py\n"
        "+++ b/source/kernel.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        f"+value = {after}\n",
        encoding="utf-8",
    )
    declared = changed_files or ["source/kernel.py"]
    manifest = {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "baseline": {
            "resolution_hash": "a" * 64,
            "file_hashes": spec["baseline"]["file_hashes"],
        },
        "changed_files": declared,
        "candidate_file_hashes": {
            "source/kernel.py": apex_launcher._sha256_bytes(
                f"value = {after}\n".encode("utf-8")
            )
        },
        "patches": [
            {
                "path": "candidate.patch",
                "sha256": apex_launcher._sha256_file(patch),
            }
        ],
        "delivery": {"mode": "bundle", "applied": False},
    }
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = apex_launcher._bundle_digest(manifest, [patch])
    result = {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "status": "candidate_ready",
        "reason_code": "candidate_verified",
        "applied": False,
        "external_verification_required": True,
        "bundle_path": str(bundle),
        "bundle_digest": digest,
        "changed_files": declared,
        # Deliberately untrusted fields: the adapter must not use them.
        "score": 999999,
        "safety_certified": True,
    }
    return bundle, digest, result


def test_agent_registry_loads_apex() -> None:
    assert AgentType.from_string("apex") is AgentType.APEX
    assert (
        load_agent_launcher(AgentType.APEX, logging.getLogger(__name__))
        is apex_launcher.launch_agent
    )


def test_task_spec_maps_caller_contract_without_arena_scoring(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    workspace, artifact_root, spec = _spec(tmp_path)

    assert spec["schema_version"] == 1
    assert spec["task_id"] == "triton2triton.vllm.sample_kernel"
    assert spec["workspace"] == str(workspace)
    assert spec["results_dir"] == str(artifact_root)
    assert spec["instructions"] == "BASE PROMPT"
    adaptation = spec["instruction_adaptation"]
    assert adaptation["strategy"] == "verbatim"
    assert adaptation["boundary_marker"] is None
    assert adaptation["original"] == adaptation["adapted"]
    assert adaptation["adapted"]["characters"] == len("BASE PROMPT")
    assert adaptation["adapted"]["sha256"] == hashlib.sha256(
        b"BASE PROMPT"
    ).hexdigest()
    assert spec["gpu_arch"] == "gfx950"
    assert spec["language"] == "triton"
    assert spec["editable_files"] == ["source/kernel.py"]
    assert spec["target_functions"] == ["sample_kernel"]
    assert spec["commands"]["compile"] == {
        "argv": ["python3", "scripts/task_runner.py", "compile"],
        "timeout_seconds": 10,
    }
    assert spec["commands"]["correctness"]["timeout_seconds"] == 20
    assert spec["commands"]["performance"]["timeout_seconds"] == 30
    assert len(spec["baseline"]["file_hashes"]["source/kernel.py"]) == 64
    assert spec["recipe"]["provenance"] == "external_evaluator"
    assert spec["agent_backend"] == "codex"
    assert spec["agent_options"] == {
        "model": "gpt-5.5",
        "effort": "xhigh",
        "runtime_closure_sha256": None,
    }
    assert spec["budget"] == {
        "max_iterations": 3,
        "max_turns": 50,
        "timeout_seconds": 120,
    }
    assert spec["caller_run_control"] is None
    assert "score" not in spec
    assert "arena" not in json.dumps(spec).lower()


def test_oversized_prompt_omits_only_known_generic_context(tmp_path) -> None:
    task_contract = "TASK CONTRACT\n" + "x" * 4_000
    generic_context = (
        apex_launcher._APEX_GENERIC_CONTEXT_MARKER
        + "\n"
        + "generic architecture and Triton advice\n" * 300
    )
    prompt = task_contract + generic_context

    instructions, provenance = apex_launcher._apex_task_instructions(
        prompt,
        workspace=tmp_path,
        sources=["source/kernel.py"],
        symbols=["sample_kernel"],
    )

    assert instructions.startswith(task_contract)
    assert generic_context not in instructions
    assert "Structured Apex handoff" in instructions
    assert f"Scored workspace: `{tmp_path}`" in instructions
    assert "- `source/kernel.py`" in instructions
    assert "- `sample_kernel`" in instructions
    assert len(instructions) <= apex_launcher._APEX_INSTRUCTION_LIMIT
    assert provenance["strategy"] == "omit_known_generic_mi355x_triton_context_v1"
    assert provenance["boundary_marker"] == (
        "# AMD MI355X (CDNA 4) Kernel Optimization Context & Directives"
    )
    assert provenance["original"]["characters"] == len(prompt)
    assert provenance["original"]["sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert provenance["adapted"]["characters"] == len(instructions)
    assert provenance["adapted"]["sha256"] == hashlib.sha256(
        instructions.encode("utf-8")
    ).hexdigest()


def test_oversized_prompt_without_known_boundary_fails_closed(tmp_path) -> None:
    prompt = "unrecognized prompt layout " + "x" * apex_launcher._APEX_INSTRUCTION_LIMIT

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="exactly one known generic context boundary",
    ):
        apex_launcher._apex_task_instructions(
            prompt,
            workspace=tmp_path,
            sources=["source/kernel.py"],
            symbols=["sample_kernel"],
        )


def test_oversized_prompt_with_ambiguous_boundary_fails_closed(tmp_path) -> None:
    marker = apex_launcher._APEX_GENERIC_CONTEXT_MARKER
    prompt = "task" + marker + ("x" * apex_launcher._APEX_INSTRUCTION_LIMIT) + marker

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="exactly one known generic context boundary; count=2",
    ):
        apex_launcher._apex_task_instructions(
            prompt,
            workspace=tmp_path,
            sources=["source/kernel.py"],
            symbols=["sample_kernel"],
        )


def test_oversized_prompt_that_remains_over_limit_fails_closed(tmp_path) -> None:
    prompt = (
        "x" * apex_launcher._APEX_INSTRUCTION_LIMIT
        + apex_launcher._APEX_GENERIC_CONTEXT_MARKER
        + "\ngeneric"
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="exceed the ContextPacket text limit after adaptation",
    ):
        apex_launcher._apex_task_instructions(
            prompt,
            workspace=tmp_path,
            sources=["source/kernel.py"],
            symbols=["sample_kernel"],
        )


def test_formal_mi355x_cohort_preserves_task_contract_and_omits_cheatsheets(
    monkeypatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)
    run_config = yaml.safe_load(
        (repository / "example_configs/benchmark_apex_mi355x_10.yaml").read_text(
            encoding="utf-8"
        )
    )
    workspace = Path("/arena/scored-workspace")

    assert len(run_config["tasks"]) == 10
    for task_name in run_config["tasks"]:
        config_path = repository / "tasks" / task_name / "config.yaml"
        task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        prompt = render_task_prompt(
            str(config_path),
            workspace,
            run_config,
            logging.getLogger(__name__),
        )
        marker = apex_launcher._APEX_GENERIC_CONTEXT_MARKER
        assert prompt.count(marker) == 1
        task_contract = prompt[: prompt.index(marker)].rstrip()
        task_instructions = task_config["prompt"]["instructions"]
        assert task_instructions in task_contract

        commands = {
            phase: apex_launcher._command_specs(task_config, phase, 3600)
            for phase in ("compile", "correctness", "performance")
        }
        run_control = apex_launcher._caller_run_control(
            formal_campaign=True,
            commands=commands,
            max_turns=50,
            max_iterations=1,
        )
        instructions, provenance = apex_launcher._apex_task_instructions(
            prompt,
            workspace=workspace,
            sources=task_config["source_file_path"],
            symbols=task_config["target_kernel_functions"],
            caller_run_control=run_control,
        )

        assert instructions.startswith(task_contract + "\n\n")
        assert task_instructions in instructions
        assert marker.lstrip("\n") not in instructions
        assert "MI355X has 304 Compute Units" not in instructions
        assert f"Scored workspace: `{workspace}`" in instructions
        for source in task_config["source_file_path"]:
            assert f"- `{source}`" in instructions
        for symbol in task_config["target_kernel_functions"]:
            assert f"- `{symbol}`" in instructions
        assert len(instructions) <= apex_launcher._APEX_INSTRUCTION_LIMIT
        assert "Produce one final source version" in instructions
        assert "Each assistant message and each tool-call start counts once" in instructions
        assert sys.executable in instructions
        assert run_control["structured_turn_budget"] == {
                "policy": "structured_agent_turn_checkpoint_v2",
            "max_turns": 50,
            "counting": "assistant_message_and_tool_call_start_each_count_once",
        }
        assert all(
            argv[0] == sys.executable
            for argv in run_control["verifier_argv"].values()
        )
        assert provenance["strategy"] == (
            "omit_known_generic_mi355x_triton_context_v1_"
            "and_append_formal_run_control_v1"
        )
        assert provenance["original"]["sha256"] == hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()


def test_task_spec_rejects_unsupported_type_before_subprocess(tmp_path) -> None:
    workspace, config_path = _task(tmp_path, task_type="hip2hip")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with pytest.raises(apex_launcher.ApexAdapterError, match="does not support task_type"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=config,
            eval_config={"target_gpu_model": "MI355X"},
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=tmp_path / "artifacts",
            prompt="prompt",
        )


def test_matched_campaign_forces_one_inner_apex_iteration(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    spec = apex_launcher._build_task_spec(
        task_config_path=config_path,
        task_config=task_config,
        eval_config={
            "target_gpu_model": "MI355X",
            "campaign": {"comparison": "apex_vs_codex"},
        },
        agent_config=_agent_config(),
        workspace=workspace,
        artifact_root=artifact_root,
        prompt="BASE PROMPT",
    )

    assert spec["budget"]["max_iterations"] == 1
    assert spec["budget"]["max_turns"] == 50


def test_formal_task_spec_binds_run_control_and_exact_python(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)
    spec = apex_launcher._build_task_spec(
        task_config_path=config_path,
        task_config=task_config,
        eval_config={
            "target_gpu_model": "MI355X",
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "backend_runtime_closure_sha256": (
                    _BACKEND_RUNTIME_CLOSURE_SHA256
                ),
            },
        },
        agent_config=_agent_config(),
        workspace=workspace,
        artifact_root=artifact_root,
        prompt="BASE PROMPT",
        campaign_binding=_campaign_binding_stub(),
    )

    control = spec["caller_run_control"]
    assert control["schema"] == "aka.apex-caller-run-control/v1"
    assert control["deliverable_versions"] == 1
    assert control["python_interpreter"]["path"] == sys.executable
    assert control["structured_turn_budget"]["policy"] == (
        "structured_agent_turn_checkpoint_v2"
    )
    assert spec["agent_options"]["runtime_closure_sha256"] == (
        _BACKEND_RUNTIME_CLOSURE_SHA256
    )
    assert all(
        spec["commands"][phase]["argv"][0] == sys.executable
        == control["verifier_argv"][phase][0]
        for phase in ("compile", "correctness", "performance")
    )
    assert "leave the best source found" in spec["instructions"]
    assert len(spec["instructions"]) <= apex_launcher._APEX_INSTRUCTION_LIMIT


def test_formal_task_spec_requires_caller_python(tmp_path, monkeypatch) -> None:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    monkeypatch.delenv("AGENT_KERNEL_ARENA_PYTHON", raising=False)

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="formal Apex requires AGENT_KERNEL_ARENA_PYTHON",
    ):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=task_config,
            eval_config={
                "target_gpu_model": "MI355X",
                "campaign": {"comparison": "apex_vs_codex"},
                "campaign_attempt": {
                    "fresh_session": True,
                    "backend_runtime_closure_sha256": (
                        _BACKEND_RUNTIME_CLOSURE_SHA256
                    ),
                },
            },
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=artifact_root,
            prompt="BASE PROMPT",
            campaign_binding=_campaign_binding_stub(),
        )


def test_formal_apex_campaign_binding_rejects_cross_task_transplant_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, config_path = _task(tmp_path)
    receipt_path = (
        workspace.parent
        / ".campaign_attempts"
        / _FORMAL_TASK_COMPONENT
        / "attempt_01/session_receipt.json"
    )
    eval_config: dict[str, object] = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
        },
    }
    _install_campaign_binding(
        eval_config=eval_config,
        task_config_path=config_path,
        manifest_root=workspace.parent,
        monkeypatch=monkeypatch,
    )
    attempt = eval_config["campaign_attempt"]
    assert isinstance(attempt, dict)
    attempt["task_name"] = "other-task"

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="task or GPU mapping differs",
    ):
        apex_launcher._campaign_binding(
            eval_config, config_path, formal_campaign=True
        )


def test_formal_apex_campaign_binding_rejects_mutable_manifest_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, config_path = _task(tmp_path)
    receipt_path = (
        workspace.parent
        / ".campaign_attempts"
        / _FORMAL_TASK_COMPONENT
        / "attempt_01/session_receipt.json"
    )
    eval_config: dict[str, object] = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
        },
    }
    binding = _install_campaign_binding(
        eval_config=eval_config,
        task_config_path=config_path,
        manifest_root=workspace.parent,
        monkeypatch=monkeypatch,
    )
    Path(str(binding["campaign_manifest_path"])).chmod(0o644)

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="single-link read-only regular file",
    ):
        apex_launcher._campaign_binding(
            eval_config, config_path, formal_campaign=True
        )


def test_formal_python_binding_rewrites_pytest_entrypoint() -> None:
    bound = apex_launcher._bind_formal_python(
        {
            "argv": ["pytest", "-vv", "test_kernel.py"],
            "timeout_seconds": 30,
        },
        {"path": sys.executable},
    )

    assert bound == {
        "argv": [sys.executable, "-m", "pytest", "-vv", "test_kernel.py"],
        "timeout_seconds": 30,
    }


def test_matched_campaign_rejects_asymmetric_turn_budget(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    agent_config = _agent_config()
    agent_config["max_turns"] = 25

    with pytest.raises(apex_launcher.ApexAdapterError, match="max_turns=50"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=task_config,
            eval_config={
                "target_gpu_model": "MI355X",
                "campaign": {"comparison": "apex_vs_codex"},
            },
            agent_config=agent_config,
            workspace=workspace,
            artifact_root=artifact_root,
            prompt="BASE PROMPT",
        )


def test_nonformal_task_spec_preserves_legacy_turn_fallback(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_root = workspace.parent / ".artifacts"
    artifact_root.mkdir()
    agent_config = _agent_config()
    del agent_config["max_turns"]

    spec = apex_launcher._build_task_spec(
        task_config_path=config_path,
        task_config=task_config,
        eval_config={"target_gpu_model": "MI355X"},
        agent_config=agent_config,
        workspace=workspace,
        artifact_root=artifact_root,
        prompt="BASE PROMPT",
    )

    assert spec["budget"]["max_turns"] == 25


@pytest.mark.parametrize("source", ["../escape.py", "/absolute.py", "./source/kernel.py"])
def test_task_spec_rejects_non_normalized_source_paths(tmp_path, source) -> None:
    workspace, config_path = _task(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["source_file_path"] = [source]
    with pytest.raises(apex_launcher.ApexAdapterError, match="workspace-relative path"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=config,
            eval_config={"target_gpu_model": "MI355X"},
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=tmp_path / "artifacts",
            prompt="prompt",
        )


def test_task_spec_rejects_source_symlink(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    source = workspace / "source" / "kernel.py"
    target = workspace / "real.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    source.unlink()
    source.symlink_to(target)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with pytest.raises(apex_launcher.ApexAdapterError, match="symlink"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=config,
            eval_config={"target_gpu_model": "MI355X"},
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=tmp_path / "artifacts",
            prompt="prompt",
        )


def test_task_spec_rejects_shell_operator_commands(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["compile_command"] = ["python3 compile.py && touch escaped"]
    with pytest.raises(apex_launcher.ApexAdapterError, match="without shell operators"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=config,
            eval_config={"target_gpu_model": "MI355X"},
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=tmp_path / "artifacts",
            prompt="prompt",
        )


def test_task_spec_rejects_multiple_commands_per_phase(tmp_path) -> None:
    workspace, config_path = _task(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["compile_command"] = ["python3 compile_a.py", "python3 compile_b.py"]
    with pytest.raises(apex_launcher.ApexAdapterError, match="exactly one compile_command"):
        apex_launcher._build_task_spec(
            task_config_path=config_path,
            task_config=config,
            eval_config={"target_gpu_model": "MI355X"},
            agent_config=_agent_config(),
            workspace=workspace,
            artifact_root=tmp_path / "artifacts",
            prompt="prompt",
        )


def test_valid_bundle_is_applied_only_to_declared_source(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    _, _, result = _make_bundle(artifact_root, spec)

    changed = apex_launcher._validate_and_apply_bundle(
        result=result,
        task_spec=spec,
        workspace=workspace,
        artifact_root=artifact_root,
        max_result_bytes=1024 * 1024,
        max_bundle_bytes=1024 * 1024,
    )

    assert changed == ["source/kernel.py"]
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (workspace / "scripts/task_runner.py").read_text(encoding="utf-8") == "# trusted fixture\n"
    assert not (workspace / "task_result.yaml").exists()


def test_bundle_digest_tampering_is_rejected_without_workspace_change(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    _, _, result = _make_bundle(artifact_root, spec)
    result["bundle_digest"] = "0" * 64

    with pytest.raises(apex_launcher.ApexAdapterError, match="bundle digest mismatch"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 1\n"


def test_bundle_wrong_baseline_is_rejected(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline"]["file_hashes"]["source/kernel.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result["bundle_digest"] = apex_launcher._bundle_digest(
        manifest, [bundle / "candidate.patch"]
    )

    with pytest.raises(apex_launcher.ApexAdapterError, match="baseline file hashes"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )


def test_bundle_undeclared_file_is_rejected(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    _, _, result = _make_bundle(
        artifact_root,
        spec,
        changed_files=["source/kernel.py", "scripts/task_runner.py"],
    )
    with pytest.raises(apex_launcher.ApexAdapterError, match="undeclared files"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )


def test_bundle_extra_file_is_rejected(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    (bundle / "unbound.txt").write_text("not covered by digest contract\n", encoding="utf-8")
    with pytest.raises(apex_launcher.ApexAdapterError, match="undeclared files"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )


def test_bundle_symlink_is_rejected(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    target = bundle / "candidate.patch"
    alias = bundle / "alias.patch"
    alias.symlink_to(target)
    with pytest.raises(apex_launcher.ApexAdapterError, match="symlink"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )


def test_bundle_duplicate_patch_path_is_rejected(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patches"].append(dict(manifest["patches"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result["bundle_digest"] = apex_launcher._bundle_digest(
        manifest, [bundle / "candidate.patch", bundle / "candidate.patch"]
    )

    with pytest.raises(apex_launcher.ApexAdapterError, match="duplicate patch path"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )


def test_git_parser_prevents_mismatched_patch_header_escape(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    patch = bundle / "candidate.patch"
    patch.write_text(
        "--- a/scripts/task_runner.py\n"
        "+++ b/scripts/task_runner.py\n"
        "@@ -1 +1 @@\n"
        "-# trusted fixture\n"
        "+# compromised fixture\n",
        encoding="utf-8",
    )
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patches"][0]["sha256"] = apex_launcher._sha256_file(patch)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result["bundle_digest"] = apex_launcher._bundle_digest(manifest, [patch])

    with pytest.raises(apex_launcher.ApexAdapterError, match="Git-parsed patch targets"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (workspace / "scripts/task_runner.py").read_text(encoding="utf-8") == "# trusted fixture\n"


def test_candidate_hash_mismatch_restores_workspace(tmp_path) -> None:
    workspace, artifact_root, spec = _spec(tmp_path)
    bundle, _, result = _make_bundle(artifact_root, spec)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_file_hashes"]["source/kernel.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result["bundle_digest"] = apex_launcher._bundle_digest(
        manifest, [bundle / "candidate.patch"]
    )

    with pytest.raises(apex_launcher.ApexAdapterError, match="applied source hash"):
        apex_launcher._validate_and_apply_bundle(
            result=result,
            task_spec=spec,
            workspace=workspace,
            artifact_root=artifact_root,
            max_result_bytes=1024 * 1024,
            max_bundle_bytes=1024 * 1024,
        )
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 1\n"


def test_launch_agent_builds_spec_and_applies_candidate(tmp_path, monkeypatch) -> None:
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    monkeypatch.setenv("APEX_ROOT", str(apex_root))
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "MATCHED BASE PROMPT"),
    )
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, backend, timeout_seconds, output_limit, logger):
        captured["command"] = command
        captured["backend"] = backend
        spec_path = Path(command[command.index("--task-spec") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        captured["spec"] = spec
        _, _, result = _make_bundle(Path(spec["results_dir"]), spec)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0, "apex-output"

    monkeypatch.setattr(apex_launcher, "_run_apex", fake_run)
    output = apex_launcher.launch_agent(
        {"target_gpu_model": "MI355X"},
        str(config_path),
        str(workspace),
    )

    assert captured["backend"] == "codex"
    assert "--result-json" in captured["command"]
    assert captured["spec"]["instructions"] == "MATCHED BASE PROMPT"
    assert Path(captured["spec"]["results_dir"]).parent.parent == workspace.parent
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 2\n"
    assert "999999" not in output
    assert '"status": "candidate_ready"' in output
    assert not (workspace / "task_result.yaml").exists()


def test_launch_agent_rejects_nonzero_no_gain_and_leaves_baseline(tmp_path, monkeypatch) -> None:
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    monkeypatch.setenv("APEX_ROOT", str(apex_root))
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "prompt"),
    )

    def fake_run(command, *, cwd, backend, timeout_seconds, output_limit, logger):
        spec_path = Path(command[command.index("--task-spec") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": spec["task_id"],
                    "status": "no_gain",
                    "reason_code": "baseline_is_best",
                    "applied": False,
                    "external_verification_required": True,
                    "bundle_path": None,
                    "bundle_digest": None,
                    "changed_files": [],
                }
            ),
            encoding="utf-8",
        )
        return 4, "no gain"

    monkeypatch.setattr(apex_launcher, "_run_apex", fake_run)
    with pytest.raises(apex_launcher.ApexAdapterError, match="process exit code 4"):
        apex_launcher.launch_agent(
            {"target_gpu_model": "MI355X"}, str(config_path), str(workspace)
        )

    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 1\n"


def test_launch_agent_accepts_zero_exit_no_gain_outside_formal_campaign(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    monkeypatch.setenv("APEX_ROOT", str(apex_root))
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "prompt"),
    )

    def fake_run(command, *, cwd, backend, timeout_seconds, output_limit, logger):
        del cwd, backend, timeout_seconds, output_limit, logger
        spec_path = Path(command[command.index("--task-spec") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": spec["task_id"],
                    "status": "no_gain",
                    "reason_code": "baseline_is_best",
                    "applied": False,
                    "external_verification_required": True,
                    "bundle_path": None,
                    "bundle_digest": None,
                    "changed_files": [],
                }
            ),
            encoding="utf-8",
        )
        return 0, "no gain"

    monkeypatch.setattr(apex_launcher, "_run_apex", fake_run)
    output = apex_launcher.launch_agent(
        {"target_gpu_model": "MI355X"}, str(config_path), str(workspace)
    )

    assert '"status": "no_gain"' in output
    assert (workspace / "source/kernel.py").read_text(encoding="utf-8") == "value = 1\n"


def _formal_apex_launch_fixture(
    tmp_path,
    monkeypatch,
    *,
    mutate_workspace: bool,
    mutate_task_spec: bool = False,
    result_status: str = "no_gain",
    return_code: int = 0,
):
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    (apex_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    for arguments in (
        ["git", "init", "-q", str(apex_root)],
        ["git", "-C", str(apex_root), "add", "main.py", ".gitignore"],
        [
            "git",
            "-C",
            str(apex_root),
            "-c",
            "user.name=AKA Test",
            "-c",
            "user.email=aka@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
    ):
        apex_launcher.subprocess.run(arguments, check=True)
    apex_python = apex_root / ".venv/bin/python"
    apex_python.parent.mkdir(parents=True)
    (apex_root / ".venv/lib/python3.10/site-packages").mkdir(parents=True)
    (apex_root / ".venv/lib/python3.10/site-packages/fixture.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (apex_root / ".venv/pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    system_python = Path("/usr/bin/python3").resolve(strict=True)
    apex_python.symlink_to(system_python)
    apex_commit = apex_launcher.subprocess.run(
        ["git", "-C", str(apex_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attempt_root = workspace.parent
    receipt_path = (
        attempt_root
        / ".campaign_attempts"
        / _FORMAL_TASK_COMPONENT
        / "attempt_01"
        / "session_receipt.json"
    )
    eval_config = {
        "target_gpu_model": "MI355X",
        "campaign": {
            "comparison": "apex_vs_codex",
            "apex_internal_allowance_seconds": 3600,
        },
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
            "comparison_contract_sha256": "d" * 64,
            "backend_runtime_closure_sha256": (
                _BACKEND_RUNTIME_CLOSURE_SHA256
            ),
        },
    }
    captured: dict[str, object] = {}
    captured["campaign_binding"] = _install_campaign_binding(
        eval_config=eval_config,
        task_config_path=config_path,
        manifest_root=attempt_root,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_COMMIT", apex_commit)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_DIRTY", "false")
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_STATUS_SHA256", hashlib.sha256(b"").hexdigest()
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(attempt_root)
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_EXTERNAL_ROOTS_JSON", "[]"
    )
    runtime_plan = plan_runtime(
        apex_root, apex_python, declared_roots=[]
    )
    runtime_snapshot = materialize_runtime(
        runtime_plan, attempt_root / "apex-shared.runtime"
    )
    runtime_manifest = apex_runtime_contract.verify_runtime_snapshot(
        runtime_snapshot, runtime_plan.sha256
    )
    immutable_mount = {
        "mount_id": 91,
        "device": "0:91",
        "root": "/",
        "mount_point": str(runtime_snapshot),
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
    monkeypatch.setattr(
        apex_runtime_contract,
        "_observed_immutable_mount",
        lambda _root: immutable_mount,
    )
    runtime_service_evidence = _runtime_service_evidence(
        runtime_snapshot, runtime_manifest, immutable_mount
    )
    immutable_receipt = apex_runtime_contract.create_immutable_mount_receipt(
        runtime_snapshot,
        runtime_manifest,
        "f" * 64,
        runtime_service_evidence,
    )
    immutable_receipt_path = attempt_root / "apex-runtime-mount-receipt.json"
    immutable_receipt_path.write_text(
        json.dumps(immutable_receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT", str(apex_root))
    monkeypatch.setenv("APEX_ROOT", str(runtime_snapshot / "repo"))
    monkeypatch.setenv(
        "APEX_PYTHON", str(runtime_snapshot / "sealed-bin/python")
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256",
        runtime_plan.sha256,
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_SNAPSHOT_ROOT",
        str(runtime_snapshot),
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT",
        str(immutable_receipt_path),
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT_SHA256",
        immutable_receipt["sha256"],
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT_FILE_SHA256",
        apex_launcher._sha256_file(immutable_receipt_path),
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "formal prompt"),
    )
    monkeypatch.setattr(apex_launcher, "formal_gpu_evidence", lambda config: {})

    def fake_home(config, *, backend):
        del config, backend
        home = receipt_path.parent / ".agent-home"
        home.mkdir(parents=True)
        return home

    def fake_wrap(
        command,
        *,
        eval_config,
        writable_roots,
        read_only_roots=(),
        trusted_read_only_roots=(),
        mount_roles=None,
        private_proc=True,
    ):
        del eval_config
        writable = [Path(path) for path in writable_roots]
        read_only = [Path(path) for path in read_only_roots]
        trusted = [Path(path) for path in trusted_read_only_roots]
        captured["writable_roots"] = tuple(writable)
        captured["read_only_roots"] = tuple(read_only)
        captured["trusted_read_only_roots"] = tuple(trusted)
        captured["mount_roles"] = mount_roles
        captured["private_proc"] = private_proc
        table = campaign_isolation._mountinfo_table()
        identities = {}
        descriptors = []
        data_descriptor, data_identity = campaign_isolation._open_mount_root(
            attempt_root, table=table
        )
        descriptors.append(data_descriptor)
        for root in (*writable, *read_only, *trusted):
            descriptor, identity = campaign_isolation._open_mount_root(
                root, table=table
            )
            descriptors.append(descriptor)
            identities[root] = identity
        mount_receipt = campaign_isolation._build_attempt_mount_receipt(
            data_root=attempt_root,
            writable=writable,
            read_only=read_only,
            trusted_read_only=trusted,
            identities=identities,
            roles=mount_roles,
            data_identity=data_identity,
        )
        for descriptor in descriptors:
            os.close(descriptor)
        captured["attempt_mount_receipt"] = mount_receipt
        return WrappedAttemptCommand(command, mount_receipt=mount_receipt)

    def fake_run(command, **kwargs):
        del kwargs
        mount_receipt = captured["attempt_mount_receipt"]
        receipt_roles = mount_receipt["roles"]
        observed_roles = {"persistent_writable": {}, "read_only": {}}
        next_visible_mount_id = iter(range(10_001, 10_006))
        for group in ("persistent_writable", "read_only"):
            for role, identity in receipt_roles[group].items():
                source_mount = dict(identity["mount"])
                visible_mount = {
                    key: source_mount[key]
                    for key in (
                        "mount_id",
                        "parent_id",
                        "major_minor",
                        "root",
                        "mount_point",
                    )
                }
                # Every bwrap bind creates a distinct visible mount even when
                # several source paths originate on the same host mount.
                visible_mount["mount_id"] = next(next_visible_mount_id)
                observed_roles[group][role] = {
                    "source": {
                        key: identity[key] for key in ("path", "device", "inode")
                    }
                    | {"mount": source_mount},
                    "target": {
                        "path": identity["path"],
                        "device": identity["device"],
                        "inode": identity["inode"],
                        "access": (
                            "read_write"
                            if group == "persistent_writable"
                            else "read_only"
                        ),
                        "mount": visible_mount,
                        "mount_options": [
                            "rw" if group == "persistent_writable" else "ro"
                        ],
                        "covered_mount_ids": (
                            [source_mount["mount_id"]]
                            if role == "apex_runtime"
                            and source_mount["mount_point"] == identity["path"]
                            else []
                        ),
                    },
                }
        mount_receipt["namespace_mounts"] = {
            "policy": "blocked_namespace_mount_attestation_v2",
            "visible_mount_resolution_policy": (
                "proc_root_o_path_fdinfo_mnt_id_v1"
            ),
            "root": {
                "mount": {"mount_id": 20_000},
                "covered_mount_ids": [],
            },
            "campaign_data_root": {
                "mount": {"mount_id": 20_001},
                "covered_mount_ids": [],
            },
            "private_tmpfs": {
                "tmp": {
                    "mount": {"mount_id": 20_002},
                    "covered_mount_ids": [],
                },
                "dev_shm": {
                    "mount": {"mount_id": 20_003},
                    "covered_mount_ids": [],
                },
            },
            "roles": observed_roles,
            "closed_set": True,
        }
        mount_receipt.pop("sha256", None)
        mount_receipt["sha256"] = apex_launcher._canonical_digest(mount_receipt)
        spec_path = Path(command[command.index("--task-spec") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        captured["spec"] = spec
        captured["task_spec_path"] = spec_path
        if mutate_task_spec:
            changed_spec = dict(spec)
            changed_spec["instructions"] = "subprocess-forged instructions"
            spec_path.chmod(0o644)
            spec_path.write_text(json.dumps(changed_spec), encoding="utf-8")
            spec_path.chmod(0o444)
        if mutate_workspace:
            (workspace / "undeclared-agent-file.txt").write_text(
                "must never survive as a scoreable no_gain\n", encoding="utf-8"
            )
        reason = (
            "agent_turn_budget_overrun"
            if result_status == "budget_exhausted"
            else "baseline_is_best"
        )
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": spec["task_id"],
                    "status": result_status,
                    "reason_code": reason,
                    "applied": False,
                    "external_verification_required": True,
                    "bundle_path": None,
                    "bundle_digest": None,
                    "changed_files": [],
                    "internal_verdict": (
                        "reject" if result_status == "budget_exhausted" else "revert"
                    ),
                    "error": (
                        {"reason_code": reason}
                        if result_status == "budget_exhausted"
                        else None
                    ),
                }
            ),
            encoding="utf-8",
        )
        return apex_launcher.ApexProcessOutcome(
            exit_code=return_code,
            stdout=b"formal apex output",
            stderr=b"",
            timed_out=False,
            cleanup={
                "required": False,
                "reason": "normal_exit",
                "scope": "private_pid_namespace",
                "method": "namespace_init_pidfd_v1",
                "boundary": {
                    "schema": "aka.attempt-process-boundary/v1",
                    "policy": "private_pid_namespace_init_pidfd_v1",
                    "pid_namespace_unshared": True,
                    "procfs": "trusted_orchestrator_inherited_procfs",
                    "namespace_init_pid": 1234,
                    "namespace_init_starttime": 99,
                    "namespace_init_parent_pid": 1200,
                    "namespace_init_inner_pid": 1,
                    "pid_namespace_id": 1001,
                    "mount_namespace_id": 1002,
                    "ipc_namespace_id": 1003,
                    "pidfd_opened": True,
                    "identity_source": "pinned_bubblewrap_json_status_fd",
                },
                "verification_performed": True,
                "namespace_init_exit_verified": True,
                "namespace_membership_enumeration_completed": True,
                "namespace_membership_scan_complete": True,
                "namespace_membership_inaccessible_entries_count": 0,
                "live_visible_namespace_members_after": [],
                "verified_absent": True,
                "sigkill_sent": False,
                "outer_supervisor_force_killed": False,
                "outer_supervisor_exit_code": return_code,
                "kernel_semantics": (
                    "linux_pid_namespace_init_exit_sigkill_all_members"
                ),
                "bubblewrap_terminal_status_verified": True,
                "bubblewrap_terminal_status": {"exit-code": return_code},
                "bubblewrap_terminal_status_absent_after_sigkill": False,
                "bubblewrap_status_eof_verified": True,
                "teardown_mode": "natural_exit",
            },
            readers_completed=True,
            capture_errors=(),
        )

    def fake_lineage(**kwargs):
        del kwargs
        return {
            "codex": {},
            "invocation": {},
            "journal_path": attempt_root / "unused.sqlite",
            "journal_head_event_id": "event-1",
            "journal_head_checksum": "a" * 64,
            "event_count": 1,
            "transcript_bytes": b"{}",
            "transcript_digest": "b" * 64,
            "event_artifact_digests": [],
            "termination_kind": "completed",
            "termination_reason": None,
            "capture_status": "complete",
            "candidate_capture_allowed": True,
            "observed_turns": 1,
            "observer_stop_sent": False,
            "process_containment": {},
            "discarded_stdout_tail": {
                "lines": 0,
                "bytes": 0,
                "sha256": None,
            },
            "prompt_event": {
                "binding": "apex.prompt_sent_event_cas/v1",
                "event_id": "event-prompt",
                "sha256": "c" * 64,
                "size_bytes": 13,
                "artifact_path": str(attempt_root / "unused-prompt"),
                "stdin_transport_attested": False,
            },
        }

    def fake_receipt(**kwargs):
        captured["receipt"] = json.loads(json.dumps(kwargs["receipt"]))
        captured["receipt_task_spec_bytes"] = kwargs["task_spec_bytes"]

    monkeypatch.setattr(apex_launcher, "prepare_attempt_home", fake_home)
    monkeypatch.setattr(
        apex_launcher,
        "codex_cloud_config_bootstrap_receipt",
        lambda _home: {
            "schema": "aka.codex-cloud-config-bootstrap/v2",
            "policy": "campaign_refreshed_minimal_home_identity_bound_signed_cache_v2",
            "relative_path": ".codex/cloud-config-bundle-cache.json",
            "present": True,
            "sha256": "d" * 64,
            "size_bytes": 1024,
            "bundle_sha256": "e" * 64,
            "signed_envelope_shape_validated": True,
            "payload_recorded": False,
        },
    )
    monkeypatch.setattr(apex_launcher, "wrap_attempt_command", fake_wrap)
    monkeypatch.setattr(apex_launcher, "_run_apex", fake_run)
    monkeypatch.setattr(apex_launcher, "_validate_apex_lineage", fake_lineage)
    monkeypatch.setattr(apex_launcher, "_write_apex_attempt_receipt", fake_receipt)
    return workspace, config_path, receipt_path, eval_config, captured


def test_formal_apex_mount_contract_keeps_scored_workspace_read_only(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path, receipt_path, eval_config, captured = (
        _formal_apex_launch_fixture(tmp_path, monkeypatch, mutate_workspace=False)
    )

    output = apex_launcher.launch_agent(eval_config, str(config_path), str(workspace))

    spec = captured["spec"]
    artifact_root = Path(spec["results_dir"])
    attempt_home = receipt_path.parent / ".agent-home"
    contract_root = Path(captured["task_spec_path"]).parent
    runtime_mount = captured["receipt"]["apex_runtime_mount"]
    runtime_root = Path(runtime_mount["root"])
    assert captured["writable_roots"] == (artifact_root, attempt_home)
    assert captured["read_only_roots"] == (workspace, contract_root)
    assert captured["trusted_read_only_roots"] == (runtime_root,)
    assert captured["mount_roles"] == {
        "apex_artifacts": artifact_root,
        "backend_home": attempt_home,
        "scored_workspace": workspace,
        "sealed_task_contract": contract_root,
        "apex_runtime": runtime_root,
    }
    assert captured["private_proc"] is False
    assert contract_root.parent == artifact_root.parent
    assert contract_root != artifact_root
    assert contract_root.stat().st_mode & 0o777 == 0o555
    assert Path(captured["task_spec_path"]).stat().st_mode & 0o777 == 0o444
    assert receipt_path.parent not in captured["writable_roots"]
    assert workspace not in captured["writable_roots"]
    integrity = captured["receipt"]["workspace_integrity"]
    assert captured["receipt"]["comparison_contract_sha256"] == captured[
        "campaign_binding"
    ]["comparison_contract_sha256"]
    assert captured["receipt"]["campaign_binding"] == captured["campaign_binding"]
    assert spec["campaign_binding"] == captured["campaign_binding"]
    assert captured["receipt"]["lineage"][
        "campaign_binding_sha256"
    ] == apex_launcher._canonical_digest(captured["campaign_binding"])
    attempt_mounts = captured["receipt"]["attempt_mounts"]
    assert runtime_mount["policy_id"] == (
        campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
    )
    assert runtime_mount["attempt_mounts_sha256"] == attempt_mounts["sha256"]
    assert attempt_mounts["roles"]["read_only"]["apex_runtime"]["path"] == str(
        runtime_root
    )
    assert runtime_root.parent.name.endswith(".runtime")
    assert runtime_root.name == runtime_mount["runtime_manifest_sha256"]
    assert integrity["pre_apply_unchanged"] is True
    assert integrity["pre_apply_manifest_sha256"] == integrity["baseline_manifest_sha256"]
    assert '"status": "no_gain"' in output


def test_formal_apex_uses_sealed_snapshot_after_live_venv_replacement(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path, _, eval_config, _ = _formal_apex_launch_fixture(
        tmp_path, monkeypatch, mutate_workspace=False
    )
    apex_root = Path(os.environ["AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT"])
    shutil.rmtree(apex_root / ".venv")
    external = tmp_path / "external-venv/bin"
    external.mkdir(parents=True)
    (external / "python").symlink_to(Path("/usr/bin/python3").resolve(strict=True))
    (apex_root / ".venv").symlink_to(external.parent, target_is_directory=True)

    output = apex_launcher.launch_agent(
        eval_config, str(config_path), str(workspace)
    )
    assert '"status": "no_gain"' in output


def test_formal_apex_checkout_below_tmp_is_rebound_exactly_read_only(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign"
    artifact_root = data_root / "run/task/attempt_01/apex-artifacts"
    apex_root = tmp_path / "apex-runtime"
    artifact_root.mkdir(parents=True)
    apex_root.mkdir()
    entrypoint = apex_root / "main.py"
    entrypoint.write_text(
        "import pathlib\n"
        "root = pathlib.Path(__file__).parent\n"
        "try:\n"
        "    (root / 'forbidden').write_text('bad')\n"
        "except OSError:\n"
        "    print('apex-visible-read-only')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setattr(
        "src.campaign_isolation._codex_requirements_identity",
        lambda: (Path("/etc/codex/requirements.toml"), {"sha256": "f" * 64}),
    )
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "backend_runtime_closure_sha256": (
                _BACKEND_RUNTIME_CLOSURE_SHA256
            ),
        },
    }
    system_python = str(Path("/usr/bin/python3").resolve(strict=True))

    hidden_command = apex_launcher.wrap_attempt_command(
        [system_python, str(entrypoint)],
        eval_config=eval_config,
        writable_roots=(artifact_root,),
        private_proc=False,
    )
    hidden_outcome = apex_launcher._run_apex(
        hidden_command,
        cwd=artifact_root,
        backend="codex",
        timeout_seconds=10,
        output_limit=1024 * 1024,
        logger=logging.getLogger(__name__),
    )
    assert hidden_outcome.exit_code != 0
    assert "No such file or directory" in hidden_outcome.output

    visible_command = apex_launcher.wrap_attempt_command(
        [system_python, str(entrypoint)],
        eval_config=eval_config,
        writable_roots=(artifact_root,),
        trusted_read_only_roots=(apex_root,),
        private_proc=False,
    )
    mount_receipt = campaign_isolation.attempt_mount_receipt(visible_command)
    visible_outcome = apex_launcher._run_apex(
        visible_command,
        cwd=artifact_root,
        backend="codex",
        timeout_seconds=10,
        output_limit=1024 * 1024,
        logger=logging.getLogger(__name__),
    )

    assert visible_outcome.exit_code == 0
    assert visible_outcome.stdout == b"apex-visible-read-only\n"
    assert not (apex_root / "forbidden").exists()
    assert mount_receipt["trusted_external_read_only_roots"] == [str(apex_root)]


def test_formal_apex_outer_boundary_reaps_env_i_double_fork_late_writer(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign"
    artifact_root = data_root / "run/task/attempt_01/apex-artifacts"
    artifact_root.mkdir(parents=True)
    marker = artifact_root / "escaped-writer"
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setattr(
        "src.campaign_isolation._codex_requirements_identity",
        lambda: (Path("/etc/codex/requirements.toml"), {"sha256": "f" * 64}),
    )
    late_code = (
        "import os,signal,sys,time\n"
        "for fd in (0,1,2):\n"
        "    try: os.close(fd)\n"
        "    except OSError: pass\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(0.5)\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write('escaped')\n"
    )
    launcher_code = (
        "import os,sys,time\n"
        "child=os.fork()\n"
        "if child==0:\n"
        " os.setsid(); grand=os.fork()\n"
        " if grand: os._exit(0)\n"
        f" os.execve(sys.executable,[sys.executable,'-c',{late_code!r},{str(marker)!r}],{{}})\n"
        "time.sleep(0.05)\n"
    )
    command = apex_launcher.wrap_attempt_command(
        [sys.executable, "-c", launcher_code],
        eval_config={
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "backend_runtime_closure_sha256": (
                    _BACKEND_RUNTIME_CLOSURE_SHA256
                ),
            },
        },
        writable_roots=(artifact_root,),
        private_proc=False,
    )
    outcome = apex_launcher._run_apex(
        command,
        cwd=artifact_root,
        backend="codex",
        timeout_seconds=10,
        output_limit=1024 * 1024,
        logger=logging.getLogger(__name__),
    )

    assert outcome.exit_code == 0
    assert outcome.cleanup["boundary"]["procfs"] == (
        "trusted_orchestrator_inherited_procfs"
    )
    assert outcome.cleanup["namespace_init_exit_verified"] is True
    assert outcome.cleanup["namespace_membership_enumeration_completed"] is True
    assert outcome.cleanup["namespace_membership_inaccessible_entries_count"] >= 0
    assert outcome.cleanup["namespace_membership_scan_complete"] is (
        outcome.cleanup["namespace_membership_inaccessible_entries_count"] == 0
    )
    assert outcome.cleanup["live_visible_namespace_members_after"] == []
    assert outcome.cleanup["verified_absent"] is True
    time.sleep(0.7)
    assert not marker.exists()


def test_formal_apex_three_layer_pid_topology_is_live_and_non_escaping(
    tmp_path, monkeypatch
) -> None:
    """Exercise AKA outer -> Apex inner -> managed-command PID nesting."""
    data_root = tmp_path / "campaign"
    artifact_root = data_root / "run/task/attempt_01/apex-artifacts"
    home = artifact_root / "home"
    credential = home / ".codex/auth.json"
    artifact_root.mkdir(parents=True)
    credential.parent.mkdir(parents=True)
    credential.write_text("secret", encoding="utf-8")
    marker = artifact_root / "escaped-third-layer-writer"
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setattr(
        "src.campaign_isolation._codex_requirements_identity",
        lambda: (Path("/etc/codex/requirements.toml"), {"sha256": "f" * 64}),
    )
    bwrap = str(Path(shutil.which("bwrap") or "/usr/bin/bwrap").resolve())
    outer_namespace = os.readlink("/proc/self/ns/pid")
    probe = (
        "import errno,json,os,pathlib,socket,stat,sys,time\n"
        "root=pathlib.Path(sys.argv[1]); cred=pathlib.Path(sys.argv[2]); marker=pathlib.Path(sys.argv[3]); outer=sys.argv[4]\n"
        "visible=set()\n"
        "for p in pathlib.Path('/proc').iterdir():\n"
        " if p.name.isdigit():\n"
        "  try: visible.add(os.readlink(p/'ns/pid'))\n"
        "  except OSError: pass\n"
        "def denied(p):\n"
        " fd=None\n"
        " try:\n"
        "  fd=os.open(p,os.O_RDONLY); os.read(fd,1); return False\n"
        " except OSError as e: return e.errno in {errno.ENOENT,errno.EACCES,errno.EPERM,errno.EIO}\n"
        " finally:\n"
        "  if fd is not None: os.close(fd)\n"
        "credential_denied=denied(cred)\n"
        "s=socket.socket(); network_denied=s.connect_ex(('1.1.1.1',443)) in {errno.ENETUNREACH,errno.EHOSTUNREACH,errno.EPERM,errno.EACCES}; s.close()\n"
        "masked_dirs=['/proc/acpi','/proc/asound','/proc/scsi','/sys/devices/virtual/powercap','/sys/firmware']; masked_files=['/proc/interrupts','/proc/kcore','/proc/keys','/proc/latency_stats','/proc/sched_debug','/proc/timer_list','/proc/timer_stats']; readonly=['/proc/bus','/proc/fs','/proc/irq','/proc/sys','/proc/sysrq-trigger']\n"
        "def ro(p):\n"
        " try: return bool(os.statvfs(p).f_flag & os.ST_RDONLY)\n"
        " except FileNotFoundError: return True\n"
        "def md(p):\n"
        " q=pathlib.Path(p)\n"
        " try: return (not q.exists()) or (q.is_dir() and not any(q.iterdir()) and ro(p))\n"
        " except OSError: return False\n"
        "def mf(p):\n"
        " try: return stat.S_ISCHR(os.stat(p).st_mode) and os.stat(p).st_rdev==os.stat('/dev/null').st_rdev and ro(p)\n"
        " except FileNotFoundError: return True\n"
        "result={'outer_pid_namespace_absent':outer not in visible,'credential_denied':credential_denied,'pid1_root_credential_denied':denied('/proc/1/root'+str(cred)),'pid1_environ_denied':denied('/proc/1/environ'),'pid1_mem_denied':denied('/proc/1/mem'),'network_denied':network_denied,'docker_system_paths_remasked':all(map(md,masked_dirs)) and all(map(mf,masked_files)) and all(map(ro,readonly)),'gpu_device_view_preserved':pathlib.Path('/dev/kfd').exists()==bool(int(sys.argv[5]))}\n"
        "print(json.dumps(result,sort_keys=True),flush=True)\n"
        "child=os.fork()\n"
        "if child==0:\n"
        " os.setsid(); grand=os.fork()\n"
        " if grand: os._exit(0)\n"
        " for fd in (0,1,2):\n"
        "  try: os.close(fd)\n"
        "  except OSError: pass\n"
        " time.sleep(.5); marker.write_text('escaped',encoding='utf-8'); os._exit(0)\n"
    )
    third = [
        bwrap,
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--bind",
        str(artifact_root),
        str(artifact_root),
        "--tmpfs",
        str(credential.parent),
        "--",
        sys.executable,
        "-c",
        probe,
        str(artifact_root),
        str(credential),
        str(marker),
        outer_namespace,
        "1" if Path("/dev/kfd").exists() else "0",
    ]
    inner = [
        bwrap,
        "--die-with-parent",
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--proc",
        "/proc",
    ]
    for path in campaign_isolation._DOCKER_MASKED_DIRECTORIES:
        if Path(path).is_dir():
            inner.extend(["--tmpfs", path, "--remount-ro", path])
    for path in campaign_isolation._DOCKER_MASKED_FILES:
        if Path(path).exists():
            inner.extend(["--ro-bind", "/dev/null", path])
    for path in campaign_isolation._DOCKER_READONLY_PATHS:
        if Path(path).exists():
            inner.extend(["--ro-bind", path, path])
    inner.extend(["--", *third])
    command = apex_launcher.wrap_attempt_command(
        inner,
        eval_config={
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "backend_runtime_closure_sha256": (
                    _BACKEND_RUNTIME_CLOSURE_SHA256
                ),
            },
        },
        writable_roots=(artifact_root,),
        private_proc=False,
    )
    outcome = apex_launcher._run_apex(
        command,
        cwd=artifact_root,
        backend="codex",
        timeout_seconds=10,
        output_limit=1024 * 1024,
        logger=logging.getLogger(__name__),
    )

    evidence = json.loads(outcome.stdout.decode().splitlines()[-1])
    assert outcome.exit_code == 0
    assert all(evidence.values())
    assert outcome.cleanup["verified_absent"] is True
    time.sleep(0.7)
    assert not marker.exists()


def test_formal_apex_rejects_task_spec_mutation_and_receipts_prelaunch_bytes(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path, _, eval_config, captured = _formal_apex_launch_fixture(
        tmp_path,
        monkeypatch,
        mutate_workspace=False,
        mutate_task_spec=True,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="TaskSpec contract changed during subprocess execution",
    ):
        apex_launcher.launch_agent(eval_config, str(config_path), str(workspace))

    receipt = captured["receipt"]
    assert receipt["schema"] == "agentkernelarena.apex-attempt-receipt/v7"
    assert receipt["session_succeeded"] is False
    assert receipt["task_spec_contract"]["postlaunch_unchanged"] is False
    received_spec = json.loads(captured["receipt_task_spec_bytes"])
    assert received_spec["instructions"].startswith("formal prompt\n\n")
    assert "### Formal run control" in received_spec["instructions"]
    on_disk_spec = json.loads(Path(captured["task_spec_path"]).read_text())
    assert on_disk_spec["instructions"] == "subprocess-forged instructions"


def _successful_process_outcome() -> apex_launcher.ApexProcessOutcome:
    return apex_launcher.ApexProcessOutcome(
        exit_code=0,
        stdout=b"stdout\n",
        stderr=b"",
        timed_out=False,
        cleanup={"verification_performed": True, "verified_absent": True},
        readers_completed=True,
        capture_errors=(),
    )


def test_apex_receipt_seals_original_arena_prompt(tmp_path) -> None:
    _, _, spec = _spec(tmp_path)
    task_spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
    receipt_path = tmp_path / "session_receipt.json"

    apex_launcher._write_apex_attempt_receipt(
        receipt_path=receipt_path,
        task_spec_bytes=task_spec_bytes,
        original_prompt_bytes=b"BASE PROMPT",
        result_path=tmp_path / "missing-result.json",
        outcome=_successful_process_outcome(),
        receipt={"schema": "fixture"},
        lineage=None,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact = receipt["artifacts"]["original_arena_prompt"]
    artifact_path = Path(artifact["path"])
    assert artifact_path.read_bytes() == b"BASE PROMPT"
    assert artifact["sha256"] == hashlib.sha256(b"BASE PROMPT").hexdigest()
    assert artifact["size_bytes"] == len(b"BASE PROMPT")
    assert artifact_path.stat().st_mode & 0o777 == 0o444
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert artifact_path.parent.stat().st_mode & 0o777 == 0o555


def test_apex_receipt_seals_event_bound_agent_prompt(tmp_path) -> None:
    _, _, spec = _spec(tmp_path)
    task_spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
    receipt_path = tmp_path / "session_receipt.json"
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    journal_path = tmp_path / "journal.sqlite"
    journal_path.write_bytes(b"journal")
    prompt_bytes = b"INNER EVENT-BOUND PROMPT"

    apex_launcher._write_apex_attempt_receipt(
        receipt_path=receipt_path,
        task_spec_bytes=task_spec_bytes,
        original_prompt_bytes=b"BASE PROMPT",
        result_path=result_path,
        outcome=_successful_process_outcome(),
        receipt={"schema": "fixture"},
        lineage={
            "journal_path": journal_path,
            "transcript_bytes": b"{}",
            "prompt_bytes": prompt_bytes,
            "prompt_event": {
                "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "size_bytes": len(prompt_bytes),
            },
        },
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact = receipt["artifacts"]["agent_prompt"]
    artifact_path = Path(artifact["path"])
    assert artifact_path.read_bytes() == prompt_bytes
    assert artifact["sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
    assert artifact_path.stat().st_mode & 0o777 == 0o444


def test_apex_receipt_rejects_original_prompt_digest_tampering(tmp_path) -> None:
    _, _, spec = _spec(tmp_path)
    spec["instruction_adaptation"]["original"]["sha256"] = "0" * 64
    task_spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
    receipt_path = tmp_path / "session_receipt.json"

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="original digest/size does not match Arena prompt",
    ):
        apex_launcher._write_apex_attempt_receipt(
            receipt_path=receipt_path,
            task_spec_bytes=task_spec_bytes,
            original_prompt_bytes=b"BASE PROMPT",
            result_path=tmp_path / "missing-result.json",
            outcome=_successful_process_outcome(),
            receipt={"schema": "fixture"},
            lineage=None,
        )

    assert not receipt_path.exists()
    assert not (tmp_path / ".session_receipt.artifacts").exists()


def _store_artifact(store: Path, content: bytes, media_type: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    relative = f"sha256/{digest[:2]}/{digest}"
    path = store / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "digest": digest,
        "media_type": media_type,
        "relative_path": relative,
        "size": len(content),
    }


def _context_packet_prompt(objective: str) -> bytes:
    identity_and_role = {
        "identity": {"context_packet_id": "context-fixture"},
        "role": {"kind": "kernel_optimizer", "objective": objective},
    }
    encoded = json.dumps(
        identity_and_role,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "# Apex ContextPacket\n\n"
        "This packet is the complete task-local observation for this invocation.\n\n"
        "## Identity and role\n\n"
        f"> {encoded}\n\n"
        "## Objective and target\n\n"
        "> {}\n"
    ).encode("utf-8")


def _lineage_fixture(
    artifact_root: Path,
    spec: dict[str, object],
    *,
    status: str,
    containment_verified: bool = True,
    intervening_prompt_event: bool = False,
    turn_count: int | None = None,
    budget_reason_override: str | None = None,
    inner_exit_code: int = 0,
    prompt_objective: str | None = None,
    containment_terminal_overlap: bool = False,
) -> tuple[dict[str, object], Path]:
    if not isinstance(spec.get("campaign_binding"), dict):
        spec["campaign_binding"] = _campaign_binding_stub()
    run_id = "run-fixture"
    attempt_id = "attempt-fixture"
    run_root = artifact_root / "runs" / run_id
    store = run_root / "artifacts"
    store.mkdir(parents=True)
    prompt_bytes = _context_packet_prompt(
        str(spec["instructions"])
        if prompt_objective is None
        else prompt_objective
    )
    prompt_receipt = _store_artifact(store, prompt_bytes, "text/plain")
    failure = status == "budget_exhausted"
    selected_turn_count = (
        spec["budget"]["max_turns"] + 1 if failure else 1
    ) if turn_count is None else turn_count
    semantic_events = (
        [
            {
                "kind": "agent_message",
                "role": "assistant",
                "source_event_index": index + 1,
            }
            for index in range(selected_turn_count)
        ]
    )
    observed_turns = len(semantic_events)
    budget_reason = (
        budget_reason_override or "max_turns_overrun" if failure else None
    )
    executable = Path(sys.executable).resolve(strict=True)
    bwrap = Path(shutil.which("bwrap") or "/usr/bin/bwrap").resolve(strict=True)
    forced_stop = failure
    effective_exit_code = (
        128 + signal.SIGKILL
        if failure and inner_exit_code == 0
        else inner_exit_code
    )
    process_containment = {
        "schema": "apex.agent-process-containment/v1",
        "policy_id": "private_pid_namespace_init_pidfd_v1",
        "launcher_path": str(bwrap),
        "launcher_sha256": apex_launcher._sha256_file(bwrap),
        "namespace_init_host_pid": 4321,
        "namespace_init_starttime": 100,
        "namespace_init_inner_pid": 1,
        "pid_namespace_inode": 2001,
        "mount_namespace_inode": 2002,
        "ipc_namespace_inode": 2003,
        "user_namespace_inode": 2004,
        "private_procfs_verified": True,
        "pidfd_opened": True,
        "termination_reason": (
            "stdout_budget_boundary" if forced_stop else "natural_exit"
        ),
        "teardown_mode": "pidfd_sigkill" if forced_stop else "natural_exit",
        "pidfd_sigkill_sent": forced_stop,
        "namespace_init_exit_verified": True,
        "wrapper_exit_verified": True,
        "wrapper_force_killed": False,
        "terminal_status_verified": not forced_stop or containment_terminal_overlap,
        "terminal_status_absent_after_sigkill": forced_stop,
        "status_eof_verified": True,
        "namespace_membership_scan_complete": True,
        "live_namespace_members_after": [],
        "namespace_empty_verified": containment_verified,
    }
    invocation = {
        "schema": "apex.agent-invocation/v3",
        "cli_name": "codex",
        "cli_version": "codex-cli fixture",
        "executable_path": str(executable),
        "entrypoint_sha256": apex_launcher._sha256_file(executable),
        "runtime_closure_sha256": spec["agent_options"][
            "runtime_closure_sha256"
        ],
        "resolved_executable_path": str(executable),
        "workspace": str(spec["workspace"]),
        "requested_allowed_files": list(spec["editable_files"]),
        "allowed_files_enforced_by_cli": False,
        "max_turns": spec["budget"]["max_turns"],
        "turn_policy": "structured_agent_turn_checkpoint_v2",
        "process_containment_policy_id": (
            "private_pid_namespace_init_pidfd_v1"
        ),
        "prompt_transport": "stdin",
        "argv": [
            str(executable),
            "exec",
            "--strict-config",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "-",
        ],
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "response_token_limit": "not_supported_context_advisory_only",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
        },
    }
    transcript = {
        "schema": "apex.agent-transcript/v3",
        "backend": spec["agent_backend"],
        "model": spec["agent_options"]["model"],
        "effort": spec["agent_options"]["effort"],
        "invocation": invocation,
        "semantic_events": semantic_events,
        "termination": {
            "kind": "turn_overrun" if failure else "completed",
            "reason": budget_reason,
            "capture_status": "complete",
            "candidate_capture_allowed": not failure,
            "observer_stop_sent": failure,
            "process_containment": process_containment,
            "discarded_stdout_tail": {
                "lines": 0,
                "bytes": 0,
                "sha256": None,
            },
            "turn_policy": "structured_agent_turn_checkpoint_v2",
            "max_turns": spec["budget"]["max_turns"],
            "observed_turns": observed_turns,
        },
    }
    transcript_bytes = json.dumps(transcript, sort_keys=True).encode()
    transcript_receipt = _store_artifact(
        store, transcript_bytes, "application/json"
    )
    agent_payload = {
        "attempt_id": attempt_id,
        "backend": spec["agent_backend"],
        "model": spec["agent_options"]["model"],
        "effort": spec["agent_options"]["effort"],
        "exit_code": effective_exit_code,
        "timed_out": False,
        "observed_turns": observed_turns,
        "message_event_count": observed_turns,
        "tool_call_event_count": 0,
        "semantic_event_count": observed_turns,
        "termination_kind": "turn_overrun" if failure else "completed",
        "termination_reason": budget_reason,
        "capture_status": "complete",
        "candidate_capture_allowed": not failure,
        "observer_stop_sent": failure,
        "process_containment_policy_id": (
            "private_pid_namespace_init_pidfd_v1"
        ),
        "process_containment": process_containment,
        "discarded_stdout_lines": 0,
        "discarded_stdout_bytes": 0,
        "discarded_stdout_sha256": None,
        "invocation": invocation,
        "artifacts": [
            {"role": "agent_transcript", "receipt": transcript_receipt}
        ],
    }
    candidate_ready = status == "candidate_ready"
    reason = (
        "agent_turn_budget_overrun"
        if failure
        else "candidate_ready" if candidate_ready else "baseline_is_best"
    )
    verdict = "reject" if failure else "keep" if candidate_ready else "revert"
    raw_events = [
        (
            "prompt_sent",
            {
                "attempt_id": attempt_id,
                "artifacts": [{"role": "prompt", "receipt": prompt_receipt}],
            },
        ),
    ]
    if intervening_prompt_event:
        raw_events.append(("knowledge_read", {"attempt_id": attempt_id}))
    raw_events.extend([
        ("agent_failed" if failure else "agent_completed", agent_payload),
        (
            "decision",
            {"attempt_id": attempt_id, "verdict": verdict, "reason": reason},
        ),
        ("run.failed" if failure else "run.succeeded", {"reason": reason}),
    ])
    transaction_id = "tx-fixture"
    events: list[dict[str, object]] = []
    parent = None
    for sequence, (event_type, payload) in enumerate(raw_events, start=1):
        event = {
            "sequence": sequence,
            "event_id": f"evt-{sequence}",
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload,
            "parent_event_id": parent,
            "idempotency_key": f"fixture.{sequence}",
            "transaction_id": transaction_id,
            "created_at_ns": sequence,
        }
        event["checksum"] = apex_launcher._canonical_json_digest(event)
        events.append(event)
        parent = event["event_id"]
    journal = run_root / "events" / "run.db"
    journal.parent.mkdir()
    connection = sqlite3.connect(journal)
    connection.executescript(
        """
        CREATE TABLE transactions (
          transaction_id TEXT PRIMARY KEY, first_sequence INTEGER NOT NULL,
          last_sequence INTEGER NOT NULL, event_count INTEGER NOT NULL,
          checksum TEXT NOT NULL
        );
        CREATE TABLE events (
          sequence INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
          run_id TEXT NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL,
          parent_event_id TEXT, idempotency_key TEXT NOT NULL,
          transaction_id TEXT NOT NULL, created_at_ns INTEGER NOT NULL,
          checksum TEXT NOT NULL
        );
        """
    )
    transaction_checksum = apex_launcher._canonical_json_digest(
        {
            "transaction_id": transaction_id,
            "event_checksums": [event["checksum"] for event in events],
        }
    )
    connection.execute(
        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
        (transaction_id, 1, len(events), len(events), transaction_checksum),
    )
    for event in events:
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["sequence"],
                event["event_id"],
                event["run_id"],
                event["event_type"],
                json.dumps(event["payload"], sort_keys=True),
                event["parent_event_id"],
                event["idempotency_key"],
                event["transaction_id"],
                event["created_at_ns"],
                event["checksum"],
            ),
        )
    connection.commit()
    connection.close()
    result = {
        "run_id": run_id,
        "task_id": spec["task_id"],
        "status": status,
        "reason_code": reason,
        "error": {"reason_code": reason} if failure else None,
        "bundle_path": None,
        "bundle_digest": None,
        "changed_files": [],
        "baseline_lock": {"file_hashes": spec["baseline"]["file_hashes"]},
        "internal_verdict": verdict,
        "internal_verdict_ref": next(
            event["event_id"]
            for event in events
            if event["event_type"] == "decision"
        ),
        "event_journal_ref": {
            "path": str(journal),
            "head_event_id": events[-1]["event_id"],
            "head_checksum": events[-1]["checksum"],
        },
        "artifact_store_ref": {
            "path": str(store),
            "receipt_digests": [transcript_receipt["digest"]],
        },
    }
    return result, store / prompt_receipt["relative_path"]


def test_formal_lineage_validates_success_and_event_bound_prompt(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(artifact_root, spec, status="candidate_ready")

    lineage = apex_launcher._validate_apex_lineage(
        result=result, task_spec=spec, artifact_root=artifact_root
    )

    assert lineage["prompt_event"]["event_id"] == "evt-1"
    expected_prompt = _context_packet_prompt(str(spec["instructions"]))
    assert lineage["prompt_event"]["sha256"] == hashlib.sha256(
        expected_prompt
    ).hexdigest()
    assert lineage["prompt_event"]["size_bytes"] == len(expected_prompt)
    assert lineage["prompt_event"]["stdin_transport_attested"] is False


def test_formal_lineage_rejects_prompt_with_different_role_objective(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="candidate_ready",
        prompt_objective="Objective without the caller run control.",
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="does not bind TaskSpec instructions",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


@pytest.mark.parametrize("status", ["candidate_ready", "no_gain"])
@pytest.mark.parametrize("turn_count", [0, 51])
def test_formal_success_lineage_rejects_turn_count_outside_budget(
    tmp_path, status, turn_count
) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status=status,
        turn_count=turn_count,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="transcript turn evidence is inconsistent",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_validates_budget_exhausted_agent_failure(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root, spec, status="budget_exhausted"
    )

    lineage = apex_launcher._validate_apex_lineage(
        result=result, task_spec=spec, artifact_root=artifact_root
    )

    assert lineage["invocation"]["turn_policy"] == (
        "structured_agent_turn_checkpoint_v2"
    )
    assert lineage["prompt_event"]["binding"] == "apex.prompt_sent_event_cas/v1"


def test_formal_lineage_rejects_non_pidfd_exit_on_budget_exhaustion(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="budget_exhausted",
        inner_exit_code=-15,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="agent_failed outcome/identity is inconsistent",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_validates_exceeded_budget_reason_at_turn_51(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="budget_exhausted",
        turn_count=51,
    )

    lineage = apex_launcher._validate_apex_lineage(
        result=result, task_spec=spec, artifact_root=artifact_root
    )

    assert lineage["invocation"]["turn_policy"] == (
        "structured_agent_turn_checkpoint_v2"
    )


@pytest.mark.parametrize(
    ("turn_count", "budget_reason"),
    [
        (50, "max_turns_overrun"),
        (51, "max_turns_exact_boundary"),
    ],
)
def test_formal_lineage_rejects_budget_reason_count_mismatch(
    tmp_path, turn_count, budget_reason
) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="budget_exhausted",
        turn_count=turn_count,
        budget_reason_override=budget_reason,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="agent_failed outcome/identity is inconsistent",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_rejects_containment_claim_tampering(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="budget_exhausted",
        containment_verified=False,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="agent_failed outcome/identity is inconsistent",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_rejects_overlapping_terminal_evidence(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="budget_exhausted",
        containment_terminal_overlap=True,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="agent_failed outcome/identity is inconsistent",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_rejects_prompt_cas_tampering(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, prompt_path = _lineage_fixture(
        artifact_root, spec, status="candidate_ready"
    )
    prompt_path.write_bytes(b"FORGED INNER PROMPT")

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="agent prompt artifact receipt does not match stored bytes",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_rejects_unbound_result_artifact(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, prompt_path = _lineage_fixture(
        artifact_root, spec, status="candidate_ready"
    )
    unbound = _store_artifact(
        prompt_path.parents[2], b"UNBOUND", "application/octet-stream"
    )
    result["artifact_store_ref"]["receipt_digests"].append(unbound["digest"])

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="artifact receipt set is not event-bound",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_lineage_rejects_noncausal_prompt_edge(tmp_path) -> None:
    _, artifact_root, spec = _spec(tmp_path)
    result, _ = _lineage_fixture(
        artifact_root,
        spec,
        status="candidate_ready",
        intervening_prompt_event=True,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="prompt event is not uniquely bound to agent invocation",
    ):
        apex_launcher._validate_apex_lineage(
            result=result, task_spec=spec, artifact_root=artifact_root
        )


def test_formal_apex_rejects_missing_comparison_contract_digest() -> None:
    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="lacks a valid comparison contract digest",
    ):
        apex_launcher._comparison_contract_sha256({}, formal_campaign=True)


def test_formal_budget_failure_validates_lineage_before_exit_code(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path, _, eval_config, captured = _formal_apex_launch_fixture(
        tmp_path,
        monkeypatch,
        mutate_workspace=False,
        result_status="budget_exhausted",
        return_code=1,
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="budget_exhausted with process exit code 1",
    ):
        apex_launcher.launch_agent(eval_config, str(config_path), str(workspace))

    receipt = captured["receipt"]
    assert receipt["session_succeeded"] is False
    assert receipt["terminal_status"] == "budget_exhausted"
    assert receipt["lineage"]["prompt_event"]["event_id"] == "event-prompt"
    assert receipt["invocation"] == {}


def test_formal_apex_rejects_no_gain_after_direct_workspace_mutation(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path, _, eval_config, captured = _formal_apex_launch_fixture(
        tmp_path, monkeypatch, mutate_workspace=True
    )

    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="scored workspace changed before adapter-owned bundle apply",
    ):
        apex_launcher.launch_agent(eval_config, str(config_path), str(workspace))

    assert (workspace / "undeclared-agent-file.txt").is_file()
    assert captured["receipt"]["session_succeeded"] is False
    assert captured["receipt"]["workspace_integrity"]["pre_apply_unchanged"] is False


def test_launch_agent_does_not_turn_infrastructure_error_into_baseline_score(
    tmp_path, monkeypatch
) -> None:
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    monkeypatch.setenv("APEX_ROOT", str(apex_root))
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "prompt"),
    )

    def fake_run(command, *, cwd, backend, timeout_seconds, output_limit, logger):
        spec_path = Path(command[command.index("--task-spec") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": spec["task_id"],
                    "status": "infrastructure_error",
                    "reason_code": "backend_unavailable",
                    "applied": False,
                    "external_verification_required": True,
                    "changed_files": [],
                }
            ),
            encoding="utf-8",
        )
        return 5, "backend failed"

    monkeypatch.setattr(apex_launcher, "_run_apex", fake_run)
    with pytest.raises(apex_launcher.ApexAdapterError, match="infrastructure_error"):
        apex_launcher.launch_agent(
            {"target_gpu_model": "MI355X"}, str(config_path), str(workspace)
        )
    assert not (workspace / "task_result.yaml").exists()


def test_matched_benchmark_configs_have_identical_ten_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    apex = yaml.safe_load(
        (root / "example_configs/benchmark_apex_mi355x_10.yaml").read_text()
    )
    codex = yaml.safe_load(
        (root / "example_configs/benchmark_codex_mi355x_10.yaml").read_text()
    )
    assert apex["agent"]["template"] == "apex"
    assert codex["agent"]["template"] == "codex"
    for agent_name in ("apex", "codex"):
        agent_config = yaml.safe_load(
            (root / "agents" / agent_name / "agent_config.yaml").read_text()
        )
        assert agent_config["max_turns"] == FORMAL_MATCHED_MAX_TURNS
    assert {key: value for key, value in apex.items() if key != "agent"} == {
        key: value for key, value in codex.items() if key != "agent"
    }
    assert len(apex["tasks"]) == 10
    assert len(set(apex["tasks"])) == 10
    assert all(task.startswith("triton2triton/vllm/") for task in apex["tasks"])
    assert apex["workspace_directory_prefix"] == "/data/viouyang/apex/aka/workspace"
    assert Path(
        f"{apex['workspace_directory_prefix']}_{apex['target_gpu_model']}_apex"
    ).is_absolute()
    assert Path(
        f"{codex['workspace_directory_prefix']}_{codex['target_gpu_model']}_codex"
    ).is_absolute()
    for task in apex["tasks"]:
        task_root = root / "tasks" / task
        task_config = yaml.safe_load(
            (task_root / "config.yaml").read_text(encoding="utf-8")
        )
        assert task_config["task_type"] == "triton2triton"
        for phase in ("compile", "correctness", "performance"):
            assert len(task_config[f"{phase}_command"]) == 1
        sources = task_config["source_file_path"]
        assert sources and all(path.startswith("source/") for path in sources)
        assert all((task_root / path).is_file() for path in sources)

    apex_agent = yaml.safe_load(
        (root / "agents/apex/agent_config.yaml").read_text(encoding="utf-8")
    )
    codex_agent = yaml.safe_load(
        (root / "agents/codex/agent_config.yaml").read_text(encoding="utf-8")
    )
    assert apex_agent["model"] == codex_agent["model"] == "gpt-5.5"
    assert apex_agent["effort"] == codex_agent["effort"] == "xhigh"
    assert (
        apex_agent["permission_mode"]
        == codex_agent["permission_mode"]
        == "workspace_write_isolated"
    )
    assert apex_agent["timeout_seconds"] == codex_agent["timeout_seconds"] == 3600
    assert (
        apex_agent["structured_stream_output_limit_bytes"]
        == codex_agent["structured_stream_output_limit_bytes"]
        == 16 * 1024 * 1024
    )
    assert apex_agent["campaign_max_iterations"] == 1
    assert codex_agent["campaign_max_iterations"] == 1
    assert apex["campaign"] == codex["campaign"]
    assert apex["campaign"] == {
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


def test_backend_environment_drops_unselected_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("CURSOR_TOKEN", "cursor-secret")
    monkeypatch.setenv("CODEX_HOME", "/codex")

    codex = apex_launcher._subprocess_environment("codex")
    assert codex["OPENAI_API_KEY"] == "openai-secret"
    assert codex["CODEX_HOME"] == "/codex"
    assert "ANTHROPIC_API_KEY" not in codex
    assert "CURSOR_TOKEN" not in codex

    claude = apex_launcher._subprocess_environment("claude")
    assert claude["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert "OPENAI_API_KEY" not in claude
    assert "CODEX_HOME" not in claude
    assert "CURSOR_TOKEN" not in claude
