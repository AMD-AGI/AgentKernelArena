"""Policy-aware orchestration for registered evaluation tools."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from .config import EvalToolsConfig, RUNTIME_OPTION_KEYS, build_evaluation_plan
from .contracts import (
    CapabilityCheck,
    CapabilityState,
    EvaluationPlan,
    EvaluationPolicy,
    EvaluationReport,
    ExecutionRecord,
    ExecutionStatus,
    FindingStatus,
    PolicyDecision,
    RuntimeClient,
    SourceEvidence,
    TaskProfile,
    ToolCapability,
    ToolContext,
    ToolEvaluation,
    ToolInvocation,
    ToolRunResult,
)
from .registry import ToolRegistry
from .task_profile import resolve_task_profile


def evaluate_policy(
    policy: EvaluationPolicy,
    evaluations: tuple[ToolEvaluation, ...],
) -> PolicyDecision:
    """Resolve whether performance may proceed under advisory/required policy."""

    reasons: list[str] = []
    for evaluation in evaluations:
        capability = evaluation.capability
        assert capability.effective is not None
        if capability.effective.state == CapabilityState.NOT_APPLICABLE:
            continue
        if not capability.ready:
            reasons.append(
                f"{capability.tool}:capability:{capability.effective.state.value}"
            )
            continue
        result = evaluation.result
        if result is None:
            reasons.append(f"{capability.tool}:missing_result")
            continue
        if result.execution != ExecutionStatus.COMPLETED:
            reasons.append(f"{capability.tool}:execution:{result.execution.value}")
        if result.finding != FindingStatus.CLEAN:
            reasons.append(f"{capability.tool}:finding:{result.finding.value}")

    satisfied = not reasons
    allowed = True if policy == EvaluationPolicy.ADVISORY else satisfied
    return PolicyDecision(
        policy=policy,
        allowed=allowed,
        policy_satisfied=satisfied,
        reasons=tuple(reasons),
    )


def _source_evidence(
    original_evidence: SourceEvidence | Mapping[str, Any] | None,
    *,
    original_root: str | Path | None,
    original_fingerprint: str | None,
    candidate_fingerprint: str | None,
) -> SourceEvidence:
    base = SourceEvidence.from_value(original_evidence)
    metadata = dict(base.metadata)
    scoring_runtime_ref = os.environ.get("AKA_SCORING_IMAGE_RUNTIME_REF")
    if scoring_runtime_ref:
        metadata["scoring_runtime"] = {
            "image_id": scoring_runtime_ref,
            "reference": os.environ.get("AKA_SCORING_IMAGE_REFERENCE"),
        }
    return SourceEvidence(
        original_root=(str(original_root) if original_root is not None else base.original_root),
        original_fingerprint=original_fingerprint or base.original_fingerprint,
        candidate_fingerprint=candidate_fingerprint or base.candidate_fingerprint,
        metadata=metadata,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_option_artifact_evidence(
    evidence: SourceEvidence,
    *,
    config: EvalToolsConfig,
    workspace: Path,
) -> SourceEvidence:
    """Bind pre-existing replay inputs to the plan fingerprint.

    A replay capsule is generated after the optimization agent has run, so its
    content is not necessarily present in the pre-agent submission snapshot.
    Hash the JSON here; its validated manifest hashes bind the HSACO and blobs.
    Invalid/missing paths are recorded rather than raised so capability
    assessment can produce the normal fail-closed adapter reason.
    """

    records: dict[str, Any] = {}
    workspace = workspace.resolve(strict=False)
    for tool in config.tools:
        raw = tool.options.get("capsule")
        if not raw:
            continue
        record: dict[str, Any] = {"configured_path": str(raw)}
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = candidate.resolve(strict=False)
        try:
            relative = candidate.relative_to(workspace)
        except ValueError:
            record["status"] = "outside_workspace"
        else:
            record["workspace_relative_path"] = relative.as_posix()
            if candidate.is_file():
                record.update(
                    status="captured",
                    sha256=_sha256_file(candidate),
                    size=candidate.stat().st_size,
                )
            else:
                record["status"] = "missing"
        records[tool.name] = {"capsule": record}
    if not records:
        return evidence
    metadata = dict(evidence.metadata)
    metadata["option_artifacts"] = records
    return SourceEvidence(
        original_root=evidence.original_root,
        original_fingerprint=evidence.original_fingerprint,
        candidate_fingerprint=evidence.candidate_fingerprint,
        metadata=metadata,
    )


def _exception_result(
    tool: str,
    message: str,
    *,
    execution_record: ExecutionRecord | None = None,
) -> ToolRunResult:
    return ToolRunResult(
        tool=tool,
        execution=(
            execution_record.status
            if execution_record is not None
            else ExecutionStatus.TOOL_ERROR
        ),
        finding=FindingStatus.INCONCLUSIVE,
        summary=message,
        execution_record=execution_record,
        metadata={"manager_error": message},
    )


def _fail_closed_on_truncated_output(result: ToolRunResult) -> ToolRunResult:
    """Prevent a bounded log prefix from being accepted as clean evidence.

    A finding seen in the retained prefix remains valid.  By contrast, a clean
    parser result is only meaningful when both streams were observed in full;
    a sanitizer report may have appeared after the worker's byte limit.
    """

    if result.finding != FindingStatus.CLEAN or result.execution_record is None:
        return result
    runtime_execution = result.execution_record.metadata.get("execution")
    if not isinstance(runtime_execution, Mapping):
        return result
    truncated_streams = tuple(
        name
        for name in ("stdout", "stderr")
        if isinstance(runtime_execution.get(name), Mapping)
        and runtime_execution[name].get("truncated") is True
    )
    if not truncated_streams:
        return result
    metadata = dict(result.metadata)
    metadata.update(
        reason_code="OUTPUT_TRUNCATED",
        truncated_streams=list(truncated_streams),
    )
    return replace(
        result,
        finding=FindingStatus.INCONCLUSIVE,
        summary=(
            "sanitizer output was truncated; absence of a finding cannot be trusted "
            f"({', '.join(truncated_streams)})"
        ),
        metadata=metadata,
    )


class EvalToolManager:
    """Run tool plugins using an injected runtime transport.

    The manager never imports Docker or mutates the parent process environment.
    A local test client, Unix-socket sidecar client, or future remote worker can
    implement the same :class:`RuntimeClient` protocol.
    """

    def __init__(self, registry: ToolRegistry, runtime_client: RuntimeClient) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        for method in ("probe", "execute"):
            if not callable(getattr(runtime_client, method, None)):
                raise TypeError(f"runtime_client must provide callable {method}()")
        self.registry = registry
        self.runtime_client = runtime_client

    def build_plan(
        self,
        *,
        config: EvalToolsConfig | Mapping[str, Any],
        task_config: Mapping[str, Any],
        profile: TaskProfile | None = None,
        source_evidence: SourceEvidence | Mapping[str, Any] | None = None,
    ) -> EvaluationPlan:
        parsed = config if isinstance(config, EvalToolsConfig) else EvalToolsConfig.from_mapping(config)
        resolved_profile = profile or resolve_task_profile(task_config)
        versions = self.registry.versions(list(parsed.enabled))
        return build_evaluation_plan(
            config=parsed,
            profile=resolved_profile,
            plugin_versions=versions,
            source_evidence=source_evidence,
        )

    def evaluate(
        self,
        *,
        workspace: str | Path,
        task_config: Mapping[str, Any],
        config: EvalToolsConfig | Mapping[str, Any],
        profile: TaskProfile | None = None,
        gpu_arch: str | None = None,
        artifact_root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        original_evidence: SourceEvidence | Mapping[str, Any] | None = None,
        original_root: str | Path | None = None,
        original_fingerprint: str | None = None,
        candidate_fingerprint: str | None = None,
    ) -> EvaluationReport:
        parsed = config if isinstance(config, EvalToolsConfig) else EvalToolsConfig.from_mapping(config)
        resolved_profile = profile or resolve_task_profile(task_config)
        workspace_path = Path(workspace).absolute()
        evidence = _source_evidence(
            original_evidence,
            original_root=original_root,
            original_fingerprint=original_fingerprint,
            candidate_fingerprint=candidate_fingerprint,
        )
        evidence = _with_option_artifact_evidence(
            evidence,
            config=parsed,
            workspace=workspace_path,
        )
        plan = self.build_plan(
            config=parsed,
            task_config=task_config,
            profile=resolved_profile,
            source_evidence=evidence,
        )

        root = Path(artifact_root).absolute() if artifact_root else workspace_path / "tool_reports"
        evaluations: list[ToolEvaluation] = []

        for tool_plan in plan.tools:
            plugin = self.registry.get(tool_plan.tool)
            artifact_dir = root / tool_plan.tool
            artifact_dir.mkdir(parents=True, exist_ok=True)
            context = ToolContext(
                workspace=str(workspace_path),
                task_config=task_config,
                profile=resolved_profile,
                artifact_dir=str(artifact_dir),
                gpu_arch=gpu_arch,
                runtime_ref=tool_plan.runtime_ref,
                env=env or {},
                options=tool_plan.options,
                source_evidence=evidence,
            )

            try:
                runtime = self.runtime_client.probe(tool_plan.tool, context)
                if not isinstance(runtime, CapabilityCheck):
                    raise TypeError(
                        f"runtime probe returned {type(runtime).__name__}, expected CapabilityCheck"
                    )
            except Exception as exc:  # runtime unavailability is reportable, not fatal
                runtime = CapabilityCheck.blocked(
                    CapabilityState.UNAVAILABLE_RUNTIME,
                    "RUNTIME_PROBE_ERROR",
                    str(exc),
                )

            # Runtime probes execute inside the tool image and are the authority
            # for container-internal paths (binaries, architecture configs,
            # headers, sanitizer runtimes). Only a narrow allow-list crosses
            # into plugin options so unrelated evidence cannot collide with an
            # adapter option; the selected runtime keys remain authoritative.
            runtime_options = {
                key: value
                for key, value in runtime.evidence.items()
                if key in RUNTIME_OPTION_KEYS.get(tool_plan.tool, frozenset())
            }
            context = replace(
                context,
                options={**dict(context.options), **runtime_options},
            )

            try:
                capability = plugin.assess(context, runtime)
                if not isinstance(capability, ToolCapability):
                    raise TypeError(
                        f"plugin assess returned {type(capability).__name__}, expected ToolCapability"
                    )
                if capability.tool != tool_plan.tool:
                    raise ValueError(
                        f"plugin returned capability for {capability.tool}, expected {tool_plan.tool}"
                    )
            except Exception as exc:
                capability = ToolCapability(
                    tool=tool_plan.tool,
                    engine=CapabilityCheck.blocked(
                        CapabilityState.UNSUPPORTED,
                        "PLUGIN_ASSESS_ERROR",
                        str(exc),
                    ),
                    adapter=CapabilityCheck.ready(),
                    runtime=runtime,
                )

            if not capability.ready:
                evaluations.append(ToolEvaluation(capability=capability))
                continue

            try:
                invocation = plugin.build_invocation(context)
                if not isinstance(invocation, ToolInvocation):
                    raise TypeError(
                        f"plugin invocation is {type(invocation).__name__}, expected ToolInvocation"
                    )
                if invocation.tool != tool_plan.tool:
                    raise ValueError(
                        f"plugin built invocation for {invocation.tool}, expected {tool_plan.tool}"
                    )
                # Run configuration is authoritative for isolation/timeouts and
                # cannot be bypassed accidentally by a plugin default.
                invocation = replace(
                    invocation,
                    timeout_s=tool_plan.timeout_s,
                    artifact_dir=str(artifact_dir),
                )
            except Exception as exc:
                evaluations.append(
                    ToolEvaluation(
                        capability=capability,
                        result=_exception_result(
                            tool_plan.tool, f"failed to build invocation: {exc}"
                        ),
                    )
                )
                continue

            try:
                execution = self.runtime_client.execute(invocation, context)
                if not isinstance(execution, ExecutionRecord):
                    raise TypeError(
                        f"runtime returned {type(execution).__name__}, expected ExecutionRecord"
                    )
            except Exception as exc:
                execution = ExecutionRecord(
                    command=invocation.command,
                    returncode=None,
                    stderr=str(exc),
                    metadata={"runtime_execute_error": str(exc)},
                )
                evaluations.append(
                    ToolEvaluation(
                        capability=capability,
                        result=_exception_result(
                            tool_plan.tool,
                            f"runtime execution failed: {exc}",
                            execution_record=execution,
                        ),
                    )
                )
                continue

            try:
                result = plugin.parse(context, execution)
                if not isinstance(result, ToolRunResult):
                    raise TypeError(
                        f"plugin parse returned {type(result).__name__}, expected ToolRunResult"
                    )
                if result.tool != tool_plan.tool:
                    raise ValueError(
                        f"plugin parsed result for {result.tool}, expected {tool_plan.tool}"
                    )
                if result.execution_record is None:
                    result = replace(result, execution_record=execution)
                result = _fail_closed_on_truncated_output(result)
            except Exception as exc:
                result = _exception_result(
                    tool_plan.tool,
                    f"failed to parse tool output: {exc}",
                    execution_record=execution,
                )
            evaluations.append(ToolEvaluation(capability=capability, result=result))

        frozen_evaluations = tuple(evaluations)
        decision = evaluate_policy(plan.policy, frozen_evaluations)
        return EvaluationReport(
            plan=plan,
            evaluations=frozen_evaluations,
            decision=decision,
        )

    # ``execute`` reads naturally at evaluator integration sites and remains an
    # alias rather than a second orchestration implementation.
    execute = evaluate


ToolManager = EvalToolManager


__all__ = ["EvalToolManager", "ToolManager", "evaluate_policy"]
