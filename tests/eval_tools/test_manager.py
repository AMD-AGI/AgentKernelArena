from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.eval_tools.config import EvalToolsConfig
from src.eval_tools.contracts import (
    CapabilityCheck,
    CapabilityState,
    ExecutionRecord,
    ExecutionStatus,
    Finding,
    FindingSeverity,
    FindingStatus,
    ToolInvocation,
    ToolRunResult,
)
from src.eval_tools.manager import EvalToolManager
from src.eval_tools.registry import ToolRegistry
from src.eval_tools.task_profile import resolve_builtin_capability


@dataclass
class FakePlugin:
    name: str = "gpu_asan"
    version: str = "1.0"
    parse_mode: str = "clean"

    def assess(self, context, runtime):
        return resolve_builtin_capability(self.name, context.profile, runtime)

    def build_invocation(self, context):
        return ToolInvocation(
            tool=self.name,
            command=("fake-tool", context.workspace),
            cwd=context.workspace,
            timeout_s=1,
        )

    def parse(self, context, execution):
        if self.parse_mode == "raise":
            raise ValueError("bad parser")
        if self.parse_mode == "finding":
            return ToolRunResult(
                tool=self.name,
                execution=ExecutionStatus.COMPLETED,
                finding=FindingStatus.FOUND,
                findings=(
                    Finding(
                        kind="heap_buffer_overflow",
                        severity=FindingSeverity.ERROR,
                        message="known OOB",
                    ),
                ),
            )
        return ToolRunResult(
            tool=self.name,
            execution=ExecutionStatus.COMPLETED,
            finding=FindingStatus.CLEAN,
        )


class FakeRuntime:
    def __init__(self, *, runtime=None, probe_error=None, execution_metadata=None):
        self.runtime = runtime or CapabilityCheck.ready(image_digest="sha256:abc")
        self.probe_error = probe_error
        self.probes = []
        self.invocations = []
        self.execution_metadata = execution_metadata or {}

    def probe(self, tool, context):
        self.probes.append((tool, context))
        if self.probe_error:
            raise self.probe_error
        return self.runtime

    def execute(self, invocation, context):
        self.invocations.append((invocation, context))
        return ExecutionRecord(
            command=invocation.command,
            returncode=0,
            stdout="tool completed",
            duration_s=0.25,
            metadata=self.execution_metadata,
        )


def _config(policy="advisory", *, runtime_ref="image@sha256:abc"):
    return EvalToolsConfig.from_mapping(
        {
            "evaluation_tools": {
                "policy": policy,
                "enabled": ["gpu_asan"],
                "tools": {
                    "gpu_asan": {
                        "runtime_ref": runtime_ref,
                        "timeout_s": 17,
                    }
                },
            }
        }
    )


def _task():
    return {
        "task_type": "triton2triton",
        "source_file_path": ["kernel.py"],
        "target_kernel_functions": ["kernel"],
    }


def _manager(plugin, runtime):
    registry = ToolRegistry()
    registry.register(plugin)
    return EvalToolManager(registry, runtime)


def test_manager_executes_ready_plugin_with_configured_isolation(tmp_path):
    plugin = FakePlugin()
    runtime = FakeRuntime()
    report = _manager(plugin, runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
        gpu_arch="gfx950",
        candidate_fingerprint="candidate-1",
        original_fingerprint="original-1",
    )

    assert report.overall_status == "clean"
    assert report.decision.allowed
    assert report.decision.policy_satisfied
    assert len(runtime.invocations) == 1
    invocation, context = runtime.invocations[0]
    assert invocation.timeout_s == 17
    artifact_dir = Path(invocation.artifact_dir)
    assert artifact_dir.parent.name == report.plan.fingerprint
    assert artifact_dir.parent.parent == tmp_path / "tool_reports" / "gpu_asan"
    assert artifact_dir.is_dir()
    assert context.runtime_ref == "image@sha256:abc"
    assert context.source_evidence.candidate_fingerprint == "candidate-1"
    result = report.evaluations[0].result
    assert result.execution_record.stdout == "tool completed"


def test_repeated_evaluation_cannot_reuse_stale_tool_artifacts(tmp_path):
    runtime = FakeRuntime()
    manager = _manager(FakePlugin(), runtime)
    first = manager.evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
        candidate_fingerprint="same-candidate",
    )
    first_dir = Path(runtime.invocations[-1][0].artifact_dir)
    (first_dir / "build_attestation.json").write_text("stale", encoding="utf-8")

    second = manager.evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
        candidate_fingerprint="same-candidate",
    )
    second_dir = Path(runtime.invocations[-1][0].artifact_dir)

    assert first.plan.fingerprint == second.plan.fingerprint
    assert first_dir != second_dir
    assert not (second_dir / "build_attestation.json").exists()


def test_runtime_evidence_cannot_be_shadowed_by_config_options(tmp_path):
    runtime = FakeRuntime(
        runtime=CapabilityCheck.ready(asan_runtime_dir="/trusted/sidecar/path")
    )
    # Configuration parsing rejects this reserved key. Construct the immutable
    # object directly to retain a defense-in-depth assertion on the manager.
    base = _config()
    config = replace(
        base,
        tools=(
            replace(
                base.tools[0],
                options={
                    **dict(base.tools[0].options),
                    "asan_runtime_dir": "/candidate/path",
                },
            ),
        ),
    )
    _manager(FakePlugin(), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=config,
    )
    assert runtime.invocations[0][1].options["asan_runtime_dir"] == "/trusted/sidecar/path"


def test_capsule_content_is_bound_into_plan_fingerprint(tmp_path):
    capsule = tmp_path / "capsule.json"
    capsule.write_text('{"case": 1}', encoding="utf-8")
    config = EvalToolsConfig.from_mapping(
        {
            "evaluation_tools": {
                "enabled": ["gpu_asan"],
                "positive_control": "optional",
                "tools": {"gpu_asan": {"options": {"capsule": "capsule.json"}}},
            }
        }
    )
    manager = _manager(FakePlugin(), FakeRuntime())
    first = manager.evaluate(workspace=tmp_path, task_config=_task(), config=config)
    record = first.plan.source_evidence.metadata["option_artifacts"]["gpu_asan"][
        "capsule"
    ]
    assert record["status"] == "captured"
    assert record["workspace_relative_path"] == "capsule.json"

    capsule.write_text('{"case": 2}', encoding="utf-8")
    second = manager.evaluate(workspace=tmp_path, task_config=_task(), config=config)
    assert second.plan.fingerprint != first.plan.fingerprint


def test_scoring_image_identity_is_recorded_in_plan_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AKA_SCORING_IMAGE_RUNTIME_REF", "sha256:scoring")
    monkeypatch.setenv("AKA_SCORING_IMAGE_REFERENCE", "example.invalid/scoring:pinned")

    report = _manager(FakePlugin(), FakeRuntime()).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
    )

    assert report.plan.source_evidence.metadata["scoring_runtime"] == {
        "image_id": "sha256:scoring",
        "reference": "example.invalid/scoring:pinned",
    }


def test_advisory_finding_is_reported_but_does_not_gate(tmp_path):
    report = _manager(FakePlugin(parse_mode="finding"), FakeRuntime()).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config("advisory"),
    )
    assert report.overall_status == "finding"
    assert report.decision.allowed
    assert not report.decision.policy_satisfied
    assert report.decision.reasons == ("gpu_asan:finding:found",)


def test_truncated_output_can_never_be_accepted_as_clean(tmp_path):
    runtime = FakeRuntime(
        execution_metadata={
            "execution": {
                "stdout": {"truncated": True},
                "stderr": {"truncated": False},
            }
        }
    )
    report = _manager(FakePlugin(), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
    )

    result = report.evaluations[0].result
    assert result.execution == ExecutionStatus.COMPLETED
    assert result.finding == FindingStatus.INCONCLUSIVE
    assert result.metadata["reason_code"] == "OUTPUT_TRUNCATED"
    assert result.metadata["truncated_streams"] == ["stdout"]
    assert report.overall_status == "incomplete"


def test_truncated_output_preserves_a_detected_finding(tmp_path):
    runtime = FakeRuntime(
        execution_metadata={
            "execution": {
                "stdout": {"truncated": False},
                "stderr": {"truncated": True},
            }
        }
    )
    report = _manager(FakePlugin(parse_mode="finding"), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
    )

    assert report.evaluations[0].result.finding == FindingStatus.FOUND


def test_required_finding_gates_performance(tmp_path):
    report = _manager(FakePlugin(parse_mode="finding"), FakeRuntime()).execute(
        workspace=tmp_path,
        task_config=_task(),
        config=_config("required"),
    )
    assert not report.decision.allowed
    assert not report.decision.policy_satisfied


def test_unavailable_runtime_remains_capability_evidence_and_is_not_executed(tmp_path):
    runtime = FakeRuntime(
        runtime=CapabilityCheck.blocked(
            CapabilityState.UNAVAILABLE_RUNTIME, "SIDECAR_OFFLINE"
        )
    )
    report = _manager(FakePlugin(), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config("required"),
    )
    evaluation = report.evaluations[0]
    assert evaluation.capability.engine.state == CapabilityState.READY
    assert evaluation.capability.runtime.state == CapabilityState.UNAVAILABLE_RUNTIME
    assert evaluation.result is None
    assert runtime.invocations == []
    assert not report.decision.allowed


def test_runtime_probe_exception_is_contained_as_unavailable(tmp_path):
    runtime = FakeRuntime(probe_error=ConnectionError("socket missing"))
    report = _manager(FakePlugin(), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
    )
    capability = report.evaluations[0].capability
    assert capability.runtime.reason_code == "RUNTIME_PROBE_ERROR"
    assert report.decision.allowed  # advisory
    assert report.overall_status == "incomplete"


def test_parser_failure_is_tool_error_not_a_clean_kernel(tmp_path):
    report = _manager(FakePlugin(parse_mode="raise"), FakeRuntime()).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=_config(),
    )
    result = report.evaluations[0].result
    assert result.execution == ExecutionStatus.COMPLETED
    assert result.finding == FindingStatus.INCONCLUSIVE
    assert "failed to parse" in result.summary
    assert report.overall_status == "incomplete"


def test_plan_fingerprint_covers_plugin_image_config_profile_and_source(tmp_path):
    plugin = FakePlugin(version="1.0")
    manager = _manager(plugin, FakeRuntime())
    profile_task = _task()

    base = manager.evaluate(
        workspace=tmp_path,
        task_config=profile_task,
        config=_config(runtime_ref="image@sha256:one"),
        candidate_fingerprint="candidate-one",
    ).plan.fingerprint
    changed_source = manager.evaluate(
        workspace=tmp_path,
        task_config=profile_task,
        config=_config(runtime_ref="image@sha256:one"),
        candidate_fingerprint="candidate-two",
    ).plan.fingerprint
    changed_image = manager.evaluate(
        workspace=tmp_path,
        task_config=profile_task,
        config=_config(runtime_ref="image@sha256:two"),
        candidate_fingerprint="candidate-one",
    ).plan.fingerprint

    plugin.version = "2.0"
    changed_plugin = manager.evaluate(
        workspace=tmp_path,
        task_config=profile_task,
        config=_config(runtime_ref="image@sha256:one"),
        candidate_fingerprint="candidate-one",
    ).plan.fingerprint
    assert len({base, changed_source, changed_image, changed_plugin}) == 4


def test_disabled_configuration_is_a_valid_noop(tmp_path):
    registry = ToolRegistry()
    report = EvalToolManager(registry, FakeRuntime()).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=EvalToolsConfig.disabled(),
    )
    assert report.evaluations == ()
    assert report.overall_status == "not_applicable"
    assert report.decision.allowed
    assert report.decision.policy_satisfied


def test_runtime_evidence_enriches_plugin_context_without_overriding_config(tmp_path):
    class EvidencePlugin(FakePlugin):
        def assess(self, context, runtime):
            assert context.options["asan_runtime_dir"] == "/opt/rocm/lib/asan"
            assert context.options["configured"] == "run-value"
            return super().assess(context, runtime)

        def build_invocation(self, context):
            assert context.options["asan_runtime_dir"] == "/opt/rocm/lib/asan"
            return super().build_invocation(context)

        def parse(self, context, execution):
            assert context.options["asan_runtime_dir"] == "/opt/rocm/lib/asan"
            return super().parse(context, execution)

    runtime = FakeRuntime(
        runtime=CapabilityCheck.ready(
            asan_runtime_dir="/opt/rocm/lib/asan",
            configured="probe-value",
        )
    )
    config = EvalToolsConfig.from_mapping(
        {
            "evaluation_tools": {
                "enabled": ["gpu_asan"],
                "tools": {
                    "gpu_asan": {
                        "options": {"configured": "run-value"},
                    }
                },
            }
        }
    )
    report = _manager(EvidencePlugin(), runtime).evaluate(
        workspace=tmp_path,
        task_config=_task(),
        config=config,
    )

    assert report.overall_status == "clean"
    assert runtime.invocations[0][1].options["asan_runtime_dir"] == "/opt/rocm/lib/asan"
    assert runtime.invocations[0][1].options["configured"] == "run-value"
