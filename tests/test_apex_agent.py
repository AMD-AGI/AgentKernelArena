"""Offline contract tests for the Apex AgentKernelArena adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

import pytest
import yaml

from src.agent_turn_budget import FORMAL_MATCHED_MAX_TURNS
from src.campaign_isolation import WrappedAttemptCommand
from src.module_registration import AgentType, load_agent_launcher
from src.prompt_builder import prompt_builder as render_task_prompt

apex_launcher = importlib.import_module("agents.apex.launch_agent")


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
    assert spec["agent_options"] == {"model": "gpt-5.5", "effort": "xhigh"}
    assert spec["budget"] == {
        "max_iterations": 3,
        "max_turns": 50,
        "timeout_seconds": 120,
    }
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


def test_formal_mi355x_cohort_preserves_task_contract_and_omits_cheatsheets() -> None:
    repository = Path(__file__).resolve().parents[1]
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

        instructions, provenance = apex_launcher._apex_task_instructions(
            prompt,
            workspace=workspace,
            sources=task_config["source_file_path"],
            symbols=task_config["target_kernel_functions"],
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
        assert provenance["strategy"] == (
            "omit_known_generic_mi355x_triton_context_v1"
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
):
    workspace, config_path = _task(tmp_path)
    apex_root = tmp_path / "apex"
    apex_root.mkdir()
    (apex_root / "main.py").write_text("# fake entrypoint\n", encoding="utf-8")
    attempt_root = workspace.parent
    receipt_path = attempt_root / "session_receipt.json"
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
        },
    }
    captured: dict[str, object] = {}
    monkeypatch.setenv("APEX_ROOT", str(apex_root))
    monkeypatch.setattr(
        apex_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "formal prompt"),
    )
    monkeypatch.setattr(apex_launcher, "formal_gpu_evidence", lambda config: {})

    def fake_home(config, *, backend):
        del config, backend
        home = attempt_root / ".agent-home"
        home.mkdir()
        return home

    def fake_wrap(
        command, *, eval_config, writable_roots, read_only_roots=()
    ):
        del eval_config
        captured["writable_roots"] = tuple(Path(path) for path in writable_roots)
        captured["read_only_roots"] = tuple(Path(path) for path in read_only_roots)
        return command

    def fake_run(command, **kwargs):
        del kwargs
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
        return 0, "formal apex output"

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
        }

    def fake_receipt(**kwargs):
        captured["receipt"] = json.loads(json.dumps(kwargs["receipt"]))
        captured["receipt_task_spec_bytes"] = kwargs["task_spec_bytes"]

    monkeypatch.setattr(apex_launcher, "prepare_attempt_home", fake_home)
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
    assert captured["writable_roots"] == (artifact_root, attempt_home)
    assert captured["read_only_roots"] == (workspace, contract_root)
    assert contract_root.parent == artifact_root.parent
    assert contract_root != artifact_root
    assert contract_root.stat().st_mode & 0o777 == 0o555
    assert Path(captured["task_spec_path"]).stat().st_mode & 0o777 == 0o444
    assert receipt_path.parent not in captured["writable_roots"]
    assert workspace not in captured["writable_roots"]
    integrity = captured["receipt"]["workspace_integrity"]
    assert captured["receipt"]["comparison_contract_sha256"] == "d" * 64
    assert integrity["pre_apply_unchanged"] is True
    assert integrity["pre_apply_manifest_sha256"] == integrity["baseline_manifest_sha256"]
    assert '"status": "no_gain"' in output


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
    assert receipt["session_succeeded"] is False
    assert receipt["task_spec_contract"]["postlaunch_unchanged"] is False
    received_spec = json.loads(captured["receipt_task_spec_bytes"])
    assert received_spec["instructions"] == "formal prompt"
    on_disk_spec = json.loads(Path(captured["task_spec_path"]).read_text())
    assert on_disk_spec["instructions"] == "subprocess-forged instructions"


def test_formal_apex_rejects_missing_comparison_contract_digest() -> None:
    with pytest.raises(
        apex_launcher.ApexAdapterError,
        match="lacks a valid comparison contract digest",
    ):
        apex_launcher._comparison_contract_sha256({}, formal_campaign=True)


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
