"""Schema-v3 sanitizer lanes: defaults, gate/signal roles, baseline rechecks and v3 adapters."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from src.eval_tools import defaults
from src.eval_tools.adapters.aka_v3_capsules import build_capsule, code_object_target
from src.eval_tools.adapters.replay_capsule import CapsuleValidationError, ReplayCapsule
from src.eval_tools.config import PLAN_SCHEMA_VERSION, EvalToolsConfig, build_evaluation_plan
from src.eval_tools.contracts import (
    CapabilityCheck,
    CapabilityState,
    EvaluationPolicy,
    ExecutionRecord,
    ExecutionStatus,
    Finding,
    FindingSeverity,
    FindingStatus,
    ToolCapability,
    ToolContext,
    ToolEvaluation,
    ToolInvocation,
    ToolRole,
    ToolRunResult,
)
from src.eval_tools.manager import BASELINE_ALSO_FOUND, EvalToolManager, evaluate_policy
from src.eval_tools.plugins import aka_v3, get_plugin, v3_tools
from src.eval_tools.registry import ToolRegistry
from src.eval_tools.task_profile import resolve_task_profile

PREFIX = "ARENA_EVAL_RESULT="


def v3_config(target: str = "triton", baseline: str = "triton") -> dict:
    entry = "solution/kernel.hip" if target == "hip" else "solution/kernel.py"
    return {
        "schema_version": 2,
        "task_id": "activation/silu/f32",
        "candidate": {"language": target, "initial_state": "implemented",
                      "editable": [{"path": "solution", "scope": "tree"}],
                      "entrypoints": [{"file": entry, "kind": "function", "symbol": "run"}]},
        "baseline": {"kind": "provided", "language": baseline, "source_files": ["baseline/kernel.py"]},
        "evaluation": {"workloads": "workloads.json"},
        "contract": {"schema_version": 3, "task_file": "task.yaml", "target_language": target,
                     "baseline_language": baseline, "operator": {"name": "silu", "category": "activation"}},
    }


def v3_context(tmp_path: Path, target="triton", baseline="triton", options=None, gpu_arch="gfx950") -> ToolContext:
    config = v3_config(target, baseline)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    return ToolContext(workspace=str(tmp_path), task_config=config, profile=resolve_task_profile(config),
                       artifact_dir=str(tmp_path / "artifacts"), gpu_arch=gpu_arch, options=options or {})


def report(lane: str, status: str = "PASS", **fields) -> str:
    body = {"protocol": "arena-eval-v1", "role": "candidate", "action": "sanitize", "status": status,
            "cases": fields.pop("cases", []), "metadata": {"lane": lane, **fields.pop("metadata", {})}, **fields}
    return PREFIX + json.dumps(body)


def execution(stdout="", stderr="", returncode=0, timed_out=False) -> ExecutionRecord:
    return ExecutionRecord(command=("python",), returncode=returncode, stdout=stdout, stderr=stderr,
                           timed_out=timed_out)


# ------------------------------------------------------------------ defaults


def test_runner_starts_defaults_only_without_a_section_on_a_default_architecture():
    assert defaults.runner_selection({}, "gfx950") == defaults.DEFAULT_ENABLED
    assert defaults.runner_selection({"agent": {}}, "gfx942") == ()
    assert defaults.runner_selection({"evaluation_tools": False}, "gfx950") == ()
    assert defaults.runner_selection({"evaluation_tools": {"enabled": []}}, "gfx950") == ()
    assert defaults.runner_selection({"evaluation_tools": {"enabled": ["gpu_asan"]}}, "gfx950") == ("gpu_asan",)
    assert defaults.runner_selection({"evaluation_tools": {"enabled": True}}, "gfx950") == defaults.ALL_TOOLS
    assert "gpu_asan" not in defaults.DEFAULT_ENABLED


def test_defaults_name_every_registered_tool():
    from src.eval_tools.contracts import ToolName

    assert set(defaults.ALL_TOOLS) == {tool.value for tool in ToolName}
    assert set(defaults.DEFAULT_ENABLED) <= set(defaults.ALL_TOOLS)


def test_missing_section_plans_the_runner_started_set_under_the_gated_policy(monkeypatch):
    monkeypatch.delenv("AKA_EVAL_TOOLS_SELECTED", raising=False)
    assert EvalToolsConfig.from_mapping({"agent": {"template": "x"}}).enabled == ()
    monkeypatch.setenv("AKA_EVAL_TOOLS_SELECTED", ",".join(defaults.DEFAULT_ENABLED))
    parsed = EvalToolsConfig.from_mapping({"agent": {"template": "x"}})
    assert parsed.enabled == defaults.DEFAULT_ENABLED
    assert parsed.policy == EvaluationPolicy.GATED
    roles = {tool.name: tool.role for tool in parsed.tools}
    assert roles["triton_fpsan"] == ToolRole.SIGNAL
    assert roles["page_fault_guard"] == roles["rocjitsu"] == ToolRole.GATE


def test_role_is_configurable_validated_and_part_of_the_plan():
    parsed = EvalToolsConfig.from_mapping({"evaluation_tools": {
        "policy": "gated", "enabled": ["csan", "triton_fpsan"],
        "tools": {"csan": {"role": "signal"}}}})
    assert {tool.name: tool.role for tool in parsed.tools} == {"csan": ToolRole.SIGNAL,
                                                                "triton_fpsan": ToolRole.SIGNAL}
    plan = build_evaluation_plan(config=parsed, profile=resolve_task_profile(v3_config()),
                                 plugin_versions={"csan": "1", "triton_fpsan": "2"})
    assert plan.schema_version == PLAN_SCHEMA_VERSION == 3
    assert [item.to_dict()["role"] for item in plan.tools] == ["signal", "signal"]
    with pytest.raises(ValueError, match="role must be gate or signal"):
        EvalToolsConfig.from_mapping({"evaluation_tools": {"enabled": ["csan"], "tools": {"csan": {"role": "x"}}}})


# -------------------------------------------------------------------- policy


def _evaluation(tool: str, finding: FindingStatus, *, metadata=None, ready=True) -> ToolEvaluation:
    runtime = CapabilityCheck.ready() if ready else CapabilityCheck.blocked(
        CapabilityState.UNAVAILABLE_RUNTIME, "down", "down")
    capability = ToolCapability(tool, CapabilityCheck.ready(), CapabilityCheck.ready(), runtime)
    if not ready:
        return ToolEvaluation(capability=capability)
    findings = (Finding(kind="x", severity=FindingSeverity.ERROR, message="x"),) \
        if finding == FindingStatus.FOUND else ()
    execution_status = ExecutionStatus.COMPLETED if finding != FindingStatus.INCONCLUSIVE else ExecutionStatus.TOOL_ERROR
    return ToolEvaluation(capability=capability, result=ToolRunResult(
        tool=tool, execution=execution_status, finding=finding, findings=findings, metadata=metadata or {}))


def test_gated_policy_blocks_only_on_an_unshared_gate_finding():
    roles = {"page_fault_guard": ToolRole.GATE, "triton_fpsan": ToolRole.SIGNAL, "csan": ToolRole.GATE}
    clean = evaluate_policy(EvaluationPolicy.GATED, (_evaluation("page_fault_guard", FindingStatus.CLEAN),), roles)
    assert clean.allowed and clean.policy_satisfied
    found = evaluate_policy(EvaluationPolicy.GATED, (_evaluation("page_fault_guard", FindingStatus.FOUND),), roles)
    assert not found.allowed and found.blocking == ("page_fault_guard:finding:found",)
    shared = evaluate_policy(EvaluationPolicy.GATED, (_evaluation(
        "page_fault_guard", FindingStatus.FOUND, metadata={BASELINE_ALSO_FOUND: True}),), roles)
    assert shared.allowed and not shared.policy_satisfied
    signal = evaluate_policy(EvaluationPolicy.GATED, (_evaluation("triton_fpsan", FindingStatus.FOUND),), roles)
    assert signal.allowed and not signal.policy_satisfied
    unverified = evaluate_policy(EvaluationPolicy.GATED, (
        _evaluation("csan", FindingStatus.INCONCLUSIVE), _evaluation("page_fault_guard", FindingStatus.CLEAN,
                                                                      ready=False)), roles)
    assert unverified.allowed and not unverified.policy_satisfied and unverified.blocking == ()
    assert "csan:execution:tool_error" in unverified.reasons


@dataclass
class RolePlugin:
    """Finds a bug for each role in ``found_for``."""

    found_for: tuple = ("candidate",)
    baseline_kind: str = "fault"
    name: str = "page_fault_guard"
    version: str = "1"
    baseline_check: bool = True

    def assess(self, context, runtime):
        return ToolCapability(self.name, CapabilityCheck.ready(), CapabilityCheck.ready(), runtime)

    def build_invocation(self, context):
        return ToolInvocation(tool=self.name, command=("lane", aka_v3.role(context)), cwd=context.workspace,
                              timeout_s=5)

    def parse(self, context, execution):
        role = execution.command[-1]
        if role in self.found_for:
            kind = self.baseline_kind if role == "baseline" else "fault"
            return ToolRunResult(tool=self.name, execution=ExecutionStatus.COMPLETED, finding=FindingStatus.FOUND,
                                 findings=(Finding(kind=kind, severity=FindingSeverity.ERROR, message=role),))
        return ToolRunResult(tool=self.name, execution=ExecutionStatus.COMPLETED, finding=FindingStatus.CLEAN)


class EchoRuntime:
    def __init__(self):
        self.commands = []

    def probe(self, tool, context):
        return CapabilityCheck.ready()

    def execute(self, invocation, context):
        self.commands.append(invocation.command)
        return ExecutionRecord(command=invocation.command, returncode=0)


@pytest.mark.parametrize("found_for,baseline_kind,allowed,shared", [
    (("candidate",), "fault", False, False),
    (("candidate", "baseline"), "fault", True, True),
    # A baseline finding of another kind does not excuse the candidate's.
    (("candidate", "baseline"), "allocator-sensitive-output", False, False),
    ((), "fault", True, False),
])
def test_a_gate_finding_is_rechecked_on_the_baseline(tmp_path, found_for, baseline_kind, allowed, shared):
    registry = ToolRegistry()
    registry.register(RolePlugin(found_for=found_for, baseline_kind=baseline_kind))
    runtime = EchoRuntime()
    config = EvalToolsConfig.from_mapping({"evaluation_tools": {"policy": "gated", "enabled": ["page_fault_guard"]}})
    report = EvalToolManager(registry, runtime).evaluate(
        workspace=tmp_path, task_config=v3_config(), config=config, gpu_arch="gfx950",
        artifact_root=tmp_path / "reports")
    assert report.decision.allowed is allowed
    result = report.evaluations[0].result
    if found_for:
        assert runtime.commands == [("lane", "candidate"), ("lane", "baseline")]
        assert bool(result.metadata.get(BASELINE_ALSO_FOUND)) is shared
        assert result.metadata["baseline_check"]["finding"] == ("found" if "baseline" in found_for else "clean")
    else:
        assert runtime.commands == [("lane", "candidate")]


# ----------------------------------------------------------- v3 lane plugins


@pytest.mark.parametrize("tool,language,state", [
    ("page_fault_guard", "flydsl", CapabilityState.READY),
    ("page_fault_guard", "gluon", CapabilityState.READY),
    ("csan", "hip", CapabilityState.READY),
    ("tilelens", "triton", CapabilityState.READY),
    ("tilelens", "gluon", CapabilityState.UNSUPPORTED),
    ("tilelens", "hip", CapabilityState.NOT_APPLICABLE),
    ("tilelens", "flydsl", CapabilityState.NOT_APPLICABLE),
    ("csan", "tilelang", CapabilityState.UNSUPPORTED),
])
def test_v3_lane_capability_by_language(tmp_path, tool, language, state):
    capability = get_plugin(tool).assess(v3_context(tmp_path, target=language), CapabilityCheck.ready())
    assert capability.effective.state == state


def test_v3_lanes_do_not_apply_without_the_v3_runner(tmp_path):
    plain = replace(v3_context(tmp_path), task_config={"schema_version": 2})
    capability = get_plugin("page_fault_guard").assess(plain, CapabilityCheck.ready())
    assert capability.effective.state == CapabilityState.NOT_APPLICABLE


def test_lane_invocation_runs_the_workspace_runner_with_the_lane_environment(tmp_path):
    invocation = get_plugin("page_fault_guard").build_invocation(v3_context(tmp_path))
    assert invocation.command[1:] == ("-m", "_aka_eval.evaluate", "candidate", "sanitize", "page_fault_guard")
    assert invocation.cwd == str(tmp_path)
    assert invocation.env["PYTHONPATH"] == ""
    assert all(invocation.env[key] == value for key, value in aka_v3.NO_CACHING.items())
    baseline = get_plugin("page_fault_guard").build_invocation(v3_context(tmp_path, options={"aka_role": "baseline"}))
    assert baseline.command[3] == "baseline"
    with pytest.raises(ValueError):
        aka_v3.role(v3_context(tmp_path, options={"aka_role": "reference"}))


def _parse(tool, tmp_path, record):
    return get_plugin(tool).parse(v3_context(tmp_path), record)


def _mismatch_case(**output):
    metrics = {"passed": False, "peak_bound_multiple": 1.14, "violation_fraction": 3e-05, "nonfinite_mismatches": 0}
    metrics.update(output)
    return {"test_case_id": "m512", "status": "FAIL", "failure_kind": "numerical_mismatch",
            "metrics": {"calls": [{"outputs": {"out": {"passed": True}}}, {"outputs": {"out": metrics}}]}}


@pytest.mark.parametrize("output,finding", [
    ({}, FindingStatus.INCONCLUSIVE),  # one element 14% over a baseline-relative bound
    ({"peak_bound_multiple": 50.0}, FindingStatus.FOUND),
    ({"violation_fraction": 0.2}, FindingStatus.FOUND),
    ({"violation_fraction": 0.2, "mismatched_elements": 100}, FindingStatus.FOUND),
    # A one-row mean 11% over its bound: one element is a 100% fraction, not gross.
    ({"violation_fraction": 1.0, "mismatched_elements": 1, "peak_bound_multiple": 1.11}, FindingStatus.INCONCLUSIVE),
    ({"nonfinite_mismatches": 3}, FindingStatus.FOUND),
])
def test_page_fault_guard_flags_only_gross_output_changes(tmp_path, output, finding):
    result = _parse("page_fault_guard", tmp_path, execution(report(
        "page_fault_guard", "FAIL", failure_kind="numerical_mismatch", reason="1 case failed",
        cases=[_mismatch_case(**output)]), returncode=1))
    assert result.finding == finding


def test_page_fault_guard_verdicts(tmp_path):
    fault = _parse("page_fault_guard", tmp_path, execution(stderr="Memory access fault by GPU node-2 on address 0x7f",
                                                           returncode=-6))
    assert fault.finding == FindingStatus.FOUND and fault.findings[0].kind == "gpu-memory-fault"
    environment = dict(aka_v3.NO_CACHING)
    clean = _parse("page_fault_guard", tmp_path, execution(report("page_fault_guard",
                                                                  metadata={"environment": environment})))
    assert clean.finding == FindingStatus.CLEAN
    unattested = _parse("page_fault_guard", tmp_path, execution(report("page_fault_guard")))
    assert unattested.finding == FindingStatus.INCONCLUSIVE
    sensitive = _parse("page_fault_guard", tmp_path, execution(report(
        "page_fault_guard", "FAIL", failure_kind="numerical_mismatch", reason="1 case failed"), returncode=1))
    assert sensitive.finding == FindingStatus.FOUND and sensitive.findings[0].kind == "allocator-sensitive-output"
    crashed = _parse("page_fault_guard", tmp_path, execution(report(
        "page_fault_guard", "FAIL", failure_kind="execution_error", reason="x",
        cases=[{"metadata": {"error": "AcceleratorError: CUDA error: an illegal memory access was encountered"}}]),
        returncode=1))
    assert crashed.finding == FindingStatus.FOUND


@pytest.mark.parametrize("stdout", [
    "",
    report("page_fault_guard") + "\nprinted after the report",
    report("page_fault_guard") + "\n" + report("page_fault_guard"),
    PREFIX + "{not json",
])
def test_an_unusable_lane_report_is_never_clean(tmp_path, stdout):
    result = _parse("page_fault_guard", tmp_path, execution(stdout))
    assert result.finding == FindingStatus.INCONCLUSIVE
    assert result.execution == ExecutionStatus.TOOL_ERROR


def test_csan_verdicts(tmp_path):
    race = _parse("csan", tmp_path, execution(report("csan", "FAIL", failure_kind="data_race",
                                                     metadata={"races": ["CUDASanitizerErrors: detected 1 errors"]}),
                                              returncode=1))
    assert race.finding == FindingStatus.FOUND and race.findings[0].kind == "stream-data-race"
    assert _parse("csan", tmp_path, execution(report("csan", metadata={"enabled": True}))).finding == \
        FindingStatus.CLEAN
    assert _parse("csan", tmp_path, execution(report("csan", metadata={"enabled": False}))).finding == \
        FindingStatus.INCONCLUSIVE


def test_tilelens_verdicts(tmp_path):
    violation = {"op": "Load", "tensor": "x", "address": "0x10", "location": ["kernel.py:18"]}
    concrete = {"op": "store", "kernel": "k", "address": "0x2000", "bytes": 4, "lanes": 1,
                "location": ["solution/kernel.py:21 k: tl.store(out_ptrs + 1, o)"]}
    found = _parse("tilelens", tmp_path, execution(report("tilelens", "FAIL", failure_kind="out_of_bounds", metadata={
        "violations": [violation],
        "confirmation": {"status": "confirmed", "violation_count": 1, "violations": [concrete]}}), returncode=1))
    assert found.finding == FindingStatus.FOUND and "kernel.py:21" in found.findings[0].locations[0]
    # A symbolic violation that a concrete interpreter run does not reproduce
    # (TileLens over-approximates loop-carried pointers) is unverified.
    for status in ("reproduced_nothing", "error", "timeout"):
        unconfirmed = _parse("tilelens", tmp_path, execution(report("tilelens", "FAIL", failure_kind="unconfirmed",
                                                                    metadata={"violations": [violation],
                                                                              "confirmation": {"status": status}}),
                                                             returncode=1))
        assert unconfirmed.finding == FindingStatus.INCONCLUSIVE
        assert unconfirmed.metadata["reason_code"] == "tilelens_unconfirmed"
    unattested = _parse("tilelens", tmp_path, execution(report("tilelens", "FAIL", failure_kind="out_of_bounds",
                                                               metadata={"violations": [violation]}), returncode=1))
    assert unattested.finding == FindingStatus.INCONCLUSIVE
    unanalyzable = _parse("tilelens", tmp_path, execution(report("tilelens", "FAIL", failure_kind="unanalyzable"),
                                                          returncode=1))
    assert unanalyzable.finding == FindingStatus.INCONCLUSIVE
    approximated = _parse("tilelens", tmp_path, execution(report("tilelens", metadata={
        "approximations": ["Triton inline assembly is approximated"]})))
    assert approximated.finding == FindingStatus.INCONCLUSIVE
    assert _parse("tilelens", tmp_path, execution(report("tilelens"))).finding == FindingStatus.CLEAN


# -------------------------------------------------- original tools on v3


@pytest.mark.parametrize("target,baseline,state", [
    ("triton", "triton", CapabilityState.READY),
    ("gluon", "triton", CapabilityState.READY),
    ("triton", "hip", CapabilityState.NOT_APPLICABLE),
    ("hip", "hip", CapabilityState.NOT_APPLICABLE),
])
def test_fpsan_compares_only_triton_or_gluon_pairs(tmp_path, target, baseline, state):
    capability = get_plugin("triton_fpsan").assess(v3_context(tmp_path, target, baseline),
                                                   CapabilityCheck.ready(triton_fpsan=True))
    assert capability.effective.state == state


def test_fpsan_v3_invocation_is_the_fpsan_lane(tmp_path):
    invocation = get_plugin("triton_fpsan").build_invocation(v3_context(tmp_path, "gluon", "triton"))
    assert invocation.command[-3:] == ("candidate", "sanitize", "fpsan")
    assert invocation.env["TRITON_INSTRUMENTATION_MODE"] == "fpsan"
    assert invocation.env["AKA_BUILD_ATTESTATION_PATH"].endswith("build_attestation.json")


_ASAN_RUNTIME = {"host_asan_preload": "/opt/rocm-asan/lib/libclang_rt.asan-x86_64.so",
                 "hip_asan_runtime": "/opt/rocm-asan/lib/libamdhip64.so",
                 "hsa_asan_runtime": "/opt/rocm-asan/lib/libhsa-runtime64.so",
                 "asan_runtime_dir": "/opt/rocm-asan/lib"}


@pytest.mark.parametrize("language,state", [("hip", CapabilityState.READY), ("gluon", CapabilityState.READY),
                                            ("flydsl", CapabilityState.UNSUPPORTED)])
def test_gpu_asan_v3_languages(tmp_path, language, state):
    capability = get_plugin("gpu_asan").assess(v3_context(tmp_path, language, language),
                                               CapabilityCheck.ready(xnack_supported=True, triton_asan=True))
    assert capability.effective.state == state


def test_gpu_asan_v3_invocations(tmp_path):
    hip = get_plugin("gpu_asan").build_invocation(v3_context(tmp_path, "hip", "hip", options=_ASAN_RUNTIME))
    assert hip.command[-3:] == ("candidate", "sanitize", "gpu_asan")
    assert hip.env["AKA_HIP_SANITIZER_CFLAGS"] == "-fsanitize=address -shared-libsan"
    assert hip.env["AKA_HIP_OFFLOAD_ARCH"] == "gfx950:xnack+"
    assert hip.env["HSA_XNACK"] == "1" and "libamdhip64.so" in hip.env["LD_PRELOAD"]
    gluon = get_plugin("gpu_asan").build_invocation(v3_context(tmp_path, "gluon", "gluon", options=_ASAN_RUNTIME))
    assert gluon.env["TRITON_ENABLE_ASAN"] == "1" and gluon.env["AMDGCN_USE_BUFFER_OPS"] == "0"
    assert all(gluon.env[key] == value for key, value in aka_v3.NO_CACHING.items())


@pytest.mark.parametrize("language,state", [("triton", CapabilityState.READY), ("gluon", CapabilityState.READY),
                                            ("hip", CapabilityState.UNSUPPORTED),
                                            ("flydsl", CapabilityState.UNSUPPORTED)])
def test_rocjitsu_v3_replays_only_captured_triton_and_gluon_launches(tmp_path, language, state):
    assert get_plugin("rocjitsu").assess(v3_context(tmp_path, language, language),
                                         CapabilityCheck.ready()).effective.state == state


def test_waitcheck_v3_runs_on_every_language_and_only_on_gfx950(tmp_path):
    for language in ("triton", "gluon", "hip", "flydsl"):
        assert get_plugin("rocjitsu_waitcheck").assess(v3_context(tmp_path, language, language),
                                                       CapabilityCheck.ready()).ready
    other = get_plugin("rocjitsu_waitcheck").assess(v3_context(tmp_path, gpu_arch="gfx942"), CapabilityCheck.ready())
    assert other.effective.state == CapabilityState.UNSUPPORTED


def _waitcheck_payload(sha: str, kernel: str, diagnostics=(), discovered=1):
    payload = {"schema_version": 1, "code_object_sha256": sha, "target": "gfx950", "expected_kernel": kernel,
               "kernel_entry": 64, "inventory_attested": True, "api_status": 0, "analysis_complete": True,
               "diagnostics_truncated": False, "stopped_early": False, "kernels_discovered": discovered,
               "kernels_analyzed": 1, "instructions_analyzed": 10, "memory_events_tracked": 2,
               "diagnostics_observed": len(diagnostics), "diagnostics_reported": len(diagnostics),
               "diagnostics": list(diagnostics), "passed": not diagnostics}
    return "AKA_WAITCHECK_RESULT " + json.dumps(payload)


def _waitcheck_artifacts(tmp_path, entries, *, sha="a" * 64, inventory_size=None):
    context = v3_context(tmp_path)
    output = Path(context.artifact_dir) / "v3-waitcheck"
    (output / "capture").mkdir(parents=True)
    (output / "capture" / "capture.json").write_text(json.dumps({
        "schema_version": 1, "code_objects": [{"code_object": "k.hsaco", "sha256": sha, "kernels": ["k"]}],
        "launches": [], "skipped": []}))
    index = {"schema_version": 1, "entries": [], "skipped": []}
    for number, text in enumerate(entries):
        log = output / "kernels" / f"{number}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text)
        entry = {"code_object": "k.hsaco", "code_object_sha256": sha, "kernel_name": "k",
                 "kernel_entry": 64, "returncode": 0, "log": f"kernels/{number}.log"}
        if inventory_size is not None:
            entry["kernels_in_code_object"] = inventory_size
        index["entries"].append(entry)
    (output / "index.json").write_text(json.dumps(index))
    return context


def test_waitcheck_v3_aggregates_one_parse_per_kernel(tmp_path):
    sha = "a" * 64
    clean = get_plugin("rocjitsu_waitcheck").parse(
        _waitcheck_artifacts(tmp_path / "clean", [_waitcheck_payload(sha, "k")]), execution())
    assert clean.finding == FindingStatus.CLEAN
    hazard = {"kernel_name": "k", "kernel_entry": 64, "code": "missing_wait", "message": "s_waitcnt missing",
              "section_name": ".text", "section_offset": 32}
    found = get_plugin("rocjitsu_waitcheck").parse(
        _waitcheck_artifacts(tmp_path / "found", [_waitcheck_payload(sha, "k"),
                                                  _waitcheck_payload(sha, "k", [hazard])]), execution())
    assert found.finding == FindingStatus.FOUND and found.findings[0].kind == "wait-hazard-missing_wait"
    nothing = get_plugin("rocjitsu_waitcheck").parse(_waitcheck_artifacts(tmp_path / "none", []), execution())
    assert nothing.finding == FindingStatus.INCONCLUSIVE


def test_waitcheck_v3_accepts_one_analyzed_kernel_of_a_multi_kernel_code_object(tmp_path):
    # A HIP extension's code object holds every kernel of the source file; the
    # bridge analyzes the selected one and counts them all as discovered.
    sha = "a" * 64
    two = [_waitcheck_payload(sha, "k", discovered=2)]
    clean = get_plugin("rocjitsu_waitcheck").parse(_waitcheck_artifacts(tmp_path / "two", two, inventory_size=2),
                                                   execution())
    assert clean.finding == FindingStatus.CLEAN
    # Without the entry point's inventory size, the count must be exactly one.
    legacy = get_plugin("rocjitsu_waitcheck").parse(_waitcheck_artifacts(tmp_path / "legacy", two), execution())
    assert legacy.finding == FindingStatus.INCONCLUSIVE
    short = get_plugin("rocjitsu_waitcheck").parse(
        _waitcheck_artifacts(tmp_path / "short", two, inventory_size=3), execution())
    assert short.finding == FindingStatus.INCONCLUSIVE


def test_waitcheck_v3_rejects_a_code_object_that_was_not_captured(tmp_path):
    context = _waitcheck_artifacts(tmp_path, [_waitcheck_payload("b" * 64, "k")])
    index_path = Path(context.artifact_dir) / "v3-waitcheck" / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"][0]["code_object_sha256"] = "b" * 64
    index_path.write_text(json.dumps(index))
    assert get_plugin("rocjitsu_waitcheck").parse(context, execution()).finding == FindingStatus.INCONCLUSIVE


# ----------------------------------------------------------------- capsules


def _hsaco(abi_version: int = 3, xnack_on: bool = False) -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[8] = abi_version
    header[0x30:0x34] = (0x300 if xnack_on else 0x200).to_bytes(4, "little")
    return bytes(header)


def _launch_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "launch-00"
    directory.mkdir()
    hsaco = _hsaco()
    (directory / "copy_kernel.hsaco").write_bytes(hsaco)
    before, after = bytes(range(16)), bytes(range(16, 32))
    for key in ("a1", "a2"):
        (directory / f"{key}.before.bin").write_bytes(before)
        (directory / f"{key}.after.bin").write_bytes(after if key == "a2" else before)
    record = {
        "kernel_name": "copy_kernel", "hsaco": "copy_kernel.hsaco", "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
        "grid": [4, 1, 1], "metadata": {"num_warps": 4, "warp_size": 64, "shared": 0, "profile_scratch_size": 0,
                                        "profile_scratch_align": 1, "global_scratch_size": 0},
        "signature": {"source": "*fp32", "destination": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        "constants": {json.dumps(["BLOCK"]): 1024}, "arg_names": ["source", "destination", "n", "BLOCK"],
        "arguments": [{"index": 0, "kind": "tensor", "allocation": "a1", "byte_offset": 0},
                      {"index": 1, "kind": "tensor", "allocation": "a2", "byte_offset": 0},
                      {"index": 2, "kind": "value", "value": 4}, {"index": 3, "kind": "value", "value": 1024}],
        "allocations": [{"id": key, "byte_size": 16, "before_blob": f"{key}.before.bin",
                         "expected_blob": f"{key}.after.bin", "changed": key == "a2"} for key in ("a1", "a2")],
    }
    (directory / "launch.json").write_text(json.dumps(record))
    return directory


def test_a_captured_launch_becomes_a_validated_replay_capsule(tmp_path):
    directory = _launch_dir(tmp_path)
    path = build_capsule(directory, manifest={"case": "t1", "seed": 7, "role": "candidate"}, task_id="t",
                         image_digest="sha256:image", rocm_version="10.0.0", framework_version="3.8.0")
    capsule = ReplayCapsule.load(path, verify_files=True)
    assert capsule.code_object.kernel_name == "copy_kernel"
    assert capsule.launch.grid == (4, 1, 1) and capsule.launch.block == (256, 1, 1)
    assert [arg.kind for arg in capsule.abi] == ["pointer", "pointer", "scalar", "implicit", "implicit"]
    assert capsule.target.code_object_version == 5 and capsule.target.xnack is False


def test_a_changed_captured_hsaco_is_rejected(tmp_path):
    directory = _launch_dir(tmp_path)
    (directory / "copy_kernel.hsaco").write_bytes(_hsaco(abi_version=4))
    with pytest.raises(CapsuleValidationError, match="hash"):
        build_capsule(directory, manifest={}, task_id="t", image_digest="d", rocm_version="r", framework_version="f")


_RACE_REPORT = "RACE type=LDS reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown\n  write ds_write_b32\nEND_RACE\n"


def _rocjitsu_artifacts(tmp_path: Path, *, replay_passed: bool):
    context = v3_context(tmp_path)
    output = Path(context.artifact_dir) / "v3-rocjitsu"
    capture = output / "capture"
    capture.mkdir(parents=True)
    directory = _launch_dir(capture)
    capsule = build_capsule(directory, manifest={"case": "t1", "seed": 7, "role": "candidate"}, task_id="t",
                            image_digest="sha256:image", rocm_version="10.0.0", framework_version="3.8.0")
    loaded = ReplayCapsule.load(capsule, verify_files=True)
    capsule_sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    (capture / "capture.json").write_text(json.dumps({
        "schema_version": 1, "code_objects": [],
        "launches": [{"directory": "launch-00", "kernel_name": "copy_kernel",
                      "hsaco_sha256": loaded.code_object.sha256}], "skipped": []}))
    log_lines = ['[rocjitsu] Kernel dispatch: "copy_kernel"',
                 f"AKA_REPLAY_CAPSULE sha256={capsule_sha} code_sha256={loaded.code_object.sha256} "
                 "adapter=triton_aot arch=gfx950 kernel=copy_kernel",
                 "AKA_REPLAY_RESULT pass" if replay_passed
                 else "AKA_REPLAY_RESULT fail: output allocation a2 mismatched golden blob"]
    (output / "replay.log").write_text("\n".join(log_lines) + "\n")
    (output / "race.log").write_text(_RACE_REPORT)
    (output / "index.json").write_text(json.dumps({"schema_version": 1, "skipped": [], "entries": [{
        "directory": "launch-00", "kernel_name": "copy_kernel", "capsule": capsule.relative_to(output).as_posix(),
        "capsule_sha256": capsule_sha, "returncode": 1, "log": "replay.log", "race_log": "race.log"}]}))
    return context


def test_rocjitsu_v3_reports_a_race_only_from_a_replay_that_reproduced_the_gpu_output(tmp_path):
    faithful = get_plugin("rocjitsu").parse(_rocjitsu_artifacts(tmp_path / "pass", replay_passed=True), execution())
    assert faithful.finding == FindingStatus.FOUND and faithful.findings[0].kind == "lds-race"
    # A replay whose outputs differ from the captured GPU outputs is not the
    # launch the GPU ran: its race report is recorded as inconclusive.
    unfaithful = get_plugin("rocjitsu").parse(_rocjitsu_artifacts(tmp_path / "fail", replay_passed=False),
                                              execution())
    assert unfaithful.finding == FindingStatus.INCONCLUSIVE
    assert "rocjitsu_v3_race_on_unattested_replay" in unfaithful.summary


def test_code_object_target_reads_the_elf_header():
    assert code_object_target(_hsaco(2, True)) == (4, True)
    with pytest.raises(CapsuleValidationError):
        code_object_target(b"\x00" * 64)


def test_v3_aggregate_is_partial_when_any_kernel_is_unverified():
    from src.eval_tools.plugins.base import PASS, ParseResult

    clean = ParseResult(PASS, reason_code="ok", attested=True)
    assert v3_tools.aggregate("x", [clean], []).status == PASS
    assert v3_tools.aggregate("x", [clean], ["launch-01: capsule: tensor descriptor"]).status == "inconclusive"
