"""Sanitizer lanes that exist only for schema-v3 tasks.

``page_fault_guard``, ``csan`` and ``tilelens`` run the generated workspace
runner's ``sanitize`` action (see ``src/tools/arena_eval/sanitize.py``) in
their own sidecar. Each judges the lane process from its exit status, stderr
and single report together; a clean verdict needs a completed report whose
environment evidence matches what this plugin set.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    CapabilityCheck,
    CapabilityState,
    ToolCapability,
    ToolContext,
    ToolInvocation,
    ToolName,
    ToolRunResult,
)
from . import aka_v3
from .base import (
    FINDING,
    INCONCLUSIVE,
    PASS,
    TOOL_ERROR,
    FindingRecord,
    ParseResult,
    blocked_check,
    parsed_to_run_result,
    ready_check,
)


def _v3_engine(context: ToolContext, tool: str, languages: frozenset[str]) -> CapabilityCheck:
    if not aka_v3.is_v3(context):
        return blocked_check(
            CapabilityState.NOT_APPLICABLE,
            f"{tool}_requires_v3_runner",
            f"{tool} runs the generated schema-v3 workspace runner; this task has none.",
        )
    language = aka_v3.role_language(context)
    if language not in aka_v3.V3_LANGUAGES:
        return blocked_check(
            CapabilityState.UNSUPPORTED,
            f"{tool}_language_unsupported",
            f"{tool} supports triton, gluon, hip and flydsl, not {language!r}.",
        )
    if language not in languages:
        return blocked_check(
            CapabilityState.NOT_APPLICABLE,
            f"{tool}_language_not_applicable",
            f"{tool} does not apply to {language} kernels.",
        )
    return ready_check(adapter=aka_v3.ADAPTER, language=language)


def _runtime(runtime: CapabilityCheck) -> CapabilityCheck:
    control = runtime.evidence.get("positive_control") if runtime.evidence else None
    if runtime.state == CapabilityState.READY and isinstance(control, dict) and control.get("passed") is False:
        return blocked_check(
            CapabilityState.UNAVAILABLE_RUNTIME,
            "positive_control_failed",
            f"the sidecar's startup control failed: {control.get('detail')}",
            **dict(runtime.evidence),
        )
    return runtime


def _report_or_error(execution, lane: str) -> tuple[dict | None, ParseResult | None]:
    if execution.timed_out:
        return None, ParseResult(TOOL_ERROR, reason_code=f"{lane}_timeout", details=execution.stderr[-4000:])
    report, problem = aka_v3.lane_report(execution.stdout, lane=lane)
    if report is None:
        return None, ParseResult(TOOL_ERROR, reason_code=f"{lane}_{problem}", details=execution.stderr[-4000:])
    return report, None


def _failure_text(report: dict) -> str:
    rows = report.get("cases") or []
    errors = [str((row.get("metadata") or {}).get("error", "")) for row in rows]
    return "\n".join([str(report.get("reason", ""))] + errors)


class _LanePlugin:
    lane = ""
    languages: frozenset[str] = aka_v3.V3_LANGUAGES
    timeout_s = 1800
    # The manager may rerun this gate on the baseline after a finding.
    baseline_check = True

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        engine = _v3_engine(context, self.name, self.languages)
        adapter = ready_check(adapter=aka_v3.ADAPTER) if engine.state == CapabilityState.READY else ready_check()
        return ToolCapability(self.name, engine, adapter, _runtime(runtime))

    def lane_env(self, context: ToolContext) -> dict[str, str]:
        return {}

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        return aka_v3.lane_invocation(
            self.name, context, self.lane, env=self.lane_env(context),
            timeout_s=int(context.options.get("timeout_s", self.timeout_s)),
        )

    def judge(self, context: ToolContext, execution) -> ParseResult:
        raise NotImplementedError

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        parsed = self.judge(context, execution)
        return parsed_to_run_result(self.name, parsed, execution, metadata={"adapter": aka_v3.ADAPTER,
                                                                             "role": aka_v3.role(context)})


class PageFaultGuardPlugin(_LanePlugin):
    """Correctness cases with the caching allocators off.

    With caching on, an access past a tensor's end usually lands silently in a
    neighbouring allocation. Off, it reaches an unmapped page and the GPU
    faults; a result that changes with the allocator also means the role
    reads memory it does not own. Accesses inside an allocation's rounding
    slack, and buffer instructions whose descriptor carries the real size
    (they clamp instead of faulting), are outside this check.
    """

    name = ToolName.PAGE_FAULT_GUARD.value
    version = "1"
    lane = "page_fault_guard"

    def lane_env(self, context: ToolContext) -> dict[str, str]:
        return dict(aka_v3.NO_CACHING)

    def judge(self, context: ToolContext, execution) -> ParseResult:
        fault = aka_v3.fault_excerpt(execution.stderr) or aka_v3.fault_excerpt(execution.stdout)
        if fault:
            finding = FindingRecord(kind="gpu-memory-fault", message="the GPU faulted on an address outside every "
                                    "allocation while the caching allocators were off", raw=fault)
            return ParseResult(FINDING, (finding,), "page_fault_guard_fault", attested=True)
        report, error = _report_or_error(execution, self.lane)
        if error is not None:
            return error
        environment = (report.get("metadata") or {}).get("environment") or {}
        if report.get("status") == "PASS":
            if any(environment.get(key) != value for key, value in aka_v3.NO_CACHING.items()):
                return ParseResult(INCONCLUSIVE, reason_code="page_fault_guard_environment_not_attested")
            return ParseResult(PASS, reason_code="page_fault_guard_clean", attested=True)
        kind = report.get("failure_kind")
        text = _failure_text(report)
        if aka_v3.FAULT_RE.search(text):
            finding = FindingRecord(kind="gpu-memory-fault", message="an illegal device memory access was reported "
                                    "while the caching allocators were off", raw=text[:4000])
            return ParseResult(FINDING, (finding,), "page_fault_guard_fault", attested=True)
        if kind in ("numerical_mismatch", "input_mutation", "mixed"):
            finding = FindingRecord(kind="allocator-sensitive-output", message="outputs that passed correctness "
                                    "changed when the caching allocators were off: the role reads memory it does "
                                    "not own", raw=text[:4000])
            return ParseResult(FINDING, (finding,), "page_fault_guard_allocator_sensitive", attested=True)
        return ParseResult(TOOL_ERROR, reason_code=f"page_fault_guard_{kind or 'failed'}", details=text[:4000])


class CsanPlugin(_LanePlugin):
    """Correctness cases under PyTorch's CUDA Sanitizer.

    CSAN tracks PyTorch dispatches on each stream, including a HIP
    candidate's custom op; raw Triton, Gluon and FlyDSL launches are not
    visible to it, so on those paths it checks the PyTorch work around them.
    """

    name = ToolName.CSAN.value
    version = "1"
    lane = "csan"

    def judge(self, context: ToolContext, execution) -> ParseResult:
        report, error = _report_or_error(execution, self.lane)
        if error is not None:
            return error
        metadata = report.get("metadata") or {}
        if report.get("failure_kind") == "data_race":
            finding = FindingRecord(kind="stream-data-race", message="CSAN reported an unsynchronized access "
                                    "between streams", raw="\n".join(metadata.get("races") or [])[:4000])
            return ParseResult(FINDING, (finding,), "csan_race", attested=True)
        if report.get("status") == "PASS":
            if metadata.get("enabled") is not True:
                return ParseResult(INCONCLUSIVE, reason_code="csan_not_enabled")
            return ParseResult(PASS, reason_code="csan_clean", attested=True)
        return ParseResult(TOOL_ERROR, reason_code=f"csan_{report.get('failure_kind') or 'failed'}",
                           details=_failure_text(report)[:4000])


class TileLensPlugin(_LanePlugin):
    """TileLens's symbolic out-of-bounds check of each Triton/Gluon kernel.

    The smallest correctness case runs in Triton's CPU interpreter, where
    every load and store address is checked against its tensor (z3 over the
    symbolic program id, mask and loop bounds). No GPU time is used.
    """

    name = ToolName.TILELENS.value
    version = "1"
    lane = "tilelens"
    languages = frozenset({"triton", "gluon"})
    timeout_s = 3600
    TILELENS_ROOT = "/opt/tilelens/site-packages"

    def lane_env(self, context: ToolContext) -> dict[str, str]:
        return {"PYTHONPATH": self.TILELENS_ROOT, "TILELENS_NUM_SMS": str(context.options.get("num_sms", 4))}

    def judge(self, context: ToolContext, execution) -> ParseResult:
        report, error = _report_or_error(execution, self.lane)
        if error is not None:
            return error
        metadata = report.get("metadata") or {}
        if report.get("failure_kind") == "out_of_bounds":
            findings = tuple(
                FindingRecord(kind="out-of-bounds", message=f"{item.get('op')} on {item.get('tensor')} at "
                              f"{item.get('address')}", location="; ".join(item.get("location") or [])[:600] or None)
                for item in (metadata.get("violations") or [])[:16]
            ) or (FindingRecord(kind="out-of-bounds", message="TileLens reported an out-of-bounds access"),)
            return ParseResult(FINDING, findings, "tilelens_out_of_bounds", attested=True)
        if report.get("failure_kind") == "unanalyzable":
            return ParseResult(INCONCLUSIVE, reason_code="tilelens_unanalyzable", details=_failure_text(report)[:4000])
        if report.get("status") == "PASS":
            if metadata.get("approximations"):
                return ParseResult(INCONCLUSIVE, reason_code="tilelens_approximated",
                                   details="; ".join(metadata["approximations"])[:4000])
            return ParseResult(PASS, reason_code="tilelens_clean", attested=True)
        return ParseResult(TOOL_ERROR, reason_code=f"tilelens_{report.get('failure_kind') or 'failed'}",
                           details=_failure_text(report)[:4000])


__all__ = ["CsanPlugin", "PageFaultGuardPlugin", "TileLensPlugin"]
