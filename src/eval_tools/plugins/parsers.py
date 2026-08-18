"""Parsers for sanitizer output.

Parsers distinguish a clean, *attested* run from an inconclusive run.  This is
essential for GPU tools: an uninstrumented HSACO normally exits zero even when
it contains the exact bug the tool was expected to catch.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from .base import FINDING, INCONCLUSIVE, PASS, TOOL_ERROR, FindingRecord, ParseResult


_ASAN_HEAD_RE = re.compile(r"(?:ERROR:\s*)?(?:AddressSanitizer|GPU AddressSanitizer)", re.I)
_ASAN_KIND_RE = re.compile(
    r"\b(heap-buffer-overflow|global-buffer-overflow|stack-buffer-overflow|"
    r"use-after-free|use-after-return|double-free|invalid-free|"
    r"container-overflow|unknown-crash)\b",
    re.I,
)
_ASAN_LOCATION_RE = re.compile(r"(?:pc|address)\s+(0x[0-9a-f]+)", re.I)


def parse_gpu_asan(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
    *,
    attested: bool,
    timed_out: bool = False,
) -> ParseResult:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    asan_marker = _ASAN_HEAD_RE.search(combined) or "Begin function __asan_report" in combined
    if asan_marker:
        kind_match = _ASAN_KIND_RE.search(combined)
        location_match = _ASAN_LOCATION_RE.search(combined)
        kind = kind_match.group(1).lower() if kind_match else "gpu-memory-error"
        finding = FindingRecord(
            kind=kind,
            message="GPU AddressSanitizer reported an invalid memory access",
            location=location_match.group(1) if location_match else None,
            raw=combined,
        )
        return ParseResult(FINDING, (finding,), "gpu_asan_finding", attested=attested)
    if timed_out:
        return ParseResult(TOOL_ERROR, reason_code="gpu_asan_timeout", details=combined, attested=attested)
    if not attested:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="gpu_asan_instrumentation_not_attested",
            details="A clean exit cannot prove safety for an uninstrumented code object.",
        )
    if returncode != 0:
        return ParseResult(
            TOOL_ERROR,
            reason_code="gpu_asan_process_failed_without_report",
            details=combined,
            attested=True,
        )
    return ParseResult(PASS, reason_code="gpu_asan_clean", attested=True)


_RACE_START_RE = re.compile(
    r"^RACE\s+type=(?P<type>\S+)\s+reg=(?P<reg>\d+)\s+wave=(?P<wave>\d+)\s+"
    r"lane=(?P<lane>\d+)\s+wg=(?P<wg>[^\s]+)(?:\s+conflict=(?P<conflict>\S+))?",
    re.M,
)
_KERNEL_RE = re.compile(r'^\[rocjitsu\]\s+Kernel dispatch:\s+"(?P<kernel>[^"]+)"', re.M)


def _race_blocks(text: str) -> Iterable[tuple[re.Match[str], str]]:
    for match in _RACE_START_RE.finditer(text):
        end = text.find("END_RACE", match.end())
        yield match, text[match.start() : (end + len("END_RACE") if end >= 0 else len(text))]


def parse_rocjitsu(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
    *,
    attested: bool,
    report_text: str = "",
    timed_out: bool = False,
) -> ParseResult:
    combined = "\n".join(part for part in (stdout, stderr, report_text) if part)
    kernel_matches = list(_KERNEL_RE.finditer(combined))
    findings = []
    seen_findings: set[tuple[object, ...]] = set()
    for match, raw in _race_blocks(combined):
        kernel = None
        for candidate in kernel_matches:
            if candidate.start() < match.start():
                kernel = candidate.group("kernel")
            else:
                break
        finding_key = (
            match.group("type").lower(),
            int(match.group("reg")),
            int(match.group("wave")),
            int(match.group("lane")),
            match.group("wg"),
            match.group("conflict"),
            kernel,
        )
        if finding_key in seen_findings:
            continue
        seen_findings.add(finding_key)
        findings.append(
            FindingRecord(
                kind=f"{match.group('type').lower()}-race",
                message=(
                    f"rocJITsu reported a {match.group('type')} race at register/byte "
                    f"{match.group('reg')} in workgroup {match.group('wg')}"
                ),
                kernel=kernel,
                location=f"wave={match.group('wave')},lane={match.group('lane')}",
                raw=raw,
                metadata={
                    "memory": match.group("type"),
                    "register_or_byte": int(match.group("reg")),
                    "wave": int(match.group("wave")),
                    "lane": int(match.group("lane")),
                    "workgroup": match.group("wg"),
                    "conflict": match.group("conflict"),
                },
            )
        )
    if findings:
        return ParseResult(FINDING, tuple(findings), "rocjitsu_race", attested=attested)
    if timed_out:
        return ParseResult(TOOL_ERROR, reason_code="rocjitsu_timeout", details=combined, attested=attested)
    if not attested:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="rocjitsu_dispatch_not_attested",
            details="No matching simulated kernel dispatch was attested.",
        )
    if returncode != 0:
        return ParseResult(
            TOOL_ERROR,
            reason_code="rocjitsu_process_failed_without_race_report",
            details=combined,
            attested=True,
        )
    if not kernel_matches and "rocjitsu" not in combined.lower():
        return ParseResult(
            INCONCLUSIVE,
            reason_code="rocjitsu_no_dispatch_observed",
            details="The launcher exited cleanly but rocJITsu logged no kernel dispatch.",
            attested=True,
        )
    return ParseResult(PASS, reason_code="rocjitsu_clean", attested=True)


_FPSAN_PREFIX = "AKA_FPSAN_RESULT "


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_fpsan_comparison(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
    *,
    attested: bool,
    timed_out: bool = False,
) -> ParseResult:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    payloads: list[object] = []
    for line in combined.splitlines():
        if line.startswith(_FPSAN_PREFIX):
            try:
                payloads.append(
                    json.loads(
                        line[len(_FPSAN_PREFIX) :],
                        parse_constant=_reject_json_constant,
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                return ParseResult(
                    TOOL_ERROR,
                    reason_code="fpsan_invalid_result_json",
                    details=str(exc),
                    attested=attested,
                )
    if timed_out:
        return ParseResult(TOOL_ERROR, reason_code="fpsan_timeout", details=combined, attested=attested)
    if returncode != 0:
        return ParseResult(
            TOOL_ERROR,
            reason_code="fpsan_process_failed",
            details=combined,
            attested=attested,
        )
    if not attested:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="fpsan_instrumentation_not_attested",
            details="FPSan outputs are meaningful only when both compared kernels were instrumented.",
        )
    if len(payloads) > 1:
        return ParseResult(
            TOOL_ERROR,
            reason_code="fpsan_multiple_results",
            details=combined,
            attested=True,
        )
    if not payloads:
        return ParseResult(
            TOOL_ERROR,
            reason_code="fpsan_comparison_missing",
            details="Expected an AKA_FPSAN_RESULT JSON line from the comparison harness.",
            attested=True,
        )
    payload = payloads[0]
    if not isinstance(payload, dict):
        return ParseResult(
            TOOL_ERROR,
            reason_code="fpsan_result_not_an_object",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    if payload.get("instrumented") is not True:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="fpsan_harness_did_not_attest_instrumentation",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    reference = payload.get("reference_digest")
    candidate = payload.get("candidate_digest")
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return ParseResult(
            TOOL_ERROR,
            reason_code="fpsan_digest_missing",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    if reference != candidate:
        finding = FindingRecord(
            kind="floating-point-semantic-mismatch",
            message="FPSan payloads differ between reference and optimized kernels",
            raw=json.dumps(payload, sort_keys=True),
            metadata={"reference_digest": reference, "candidate_digest": candidate},
        )
        return ParseResult(FINDING, (finding,), "fpsan_semantic_mismatch", attested=True)
    return ParseResult(PASS, reason_code="fpsan_equivalent", attested=True)


_WAITCHECK_PREFIX = "AKA_WAITCHECK_RESULT "
_CONSAN_RUN_PREFIX = "AKA_CONSAN_RUN "
_CONSAN_PREFIX = "ConSan "
_UNSIGNED_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_PAIR_RE = re.compile(r"(0|[1-9][0-9]*)/(0|[1-9][0-9]*)\Z")
_CONSAN_SITE_KINDS = ("access", "barrier", "atomic", "fence")
_CONSAN_RECORD_FIRST_FIELD = {
    "patch end": "reader",
    "coverage": "reader",
    "analysis verdict": "applicable",
    "MOI auto report": "reader",
    "MOI auto replay diagnostic": "reader",
}


def _json_payloads(text: str, prefix: str) -> list[object]:
    payloads: list[object] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        payloads.append(
            json.loads(line[len(prefix) :], parse_constant=_reject_json_constant)
        )
    return payloads


def parse_waitcheck(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
    *,
    expected_sha256: str,
    expected_target: str,
    expected_kernel: str,
    expected_entry: int,
    timed_out: bool = False,
) -> ParseResult:
    """Parse the single JSON record emitted by the image-owned C API bridge."""

    combined = "\n".join(part for part in (stdout, stderr) if part)
    if timed_out:
        return ParseResult(
            TOOL_ERROR, reason_code="waitcheck_timeout", details=combined
        )
    try:
        payloads = _json_payloads(combined, _WAITCHECK_PREFIX)
    except (json.JSONDecodeError, ValueError) as error:
        return ParseResult(
            TOOL_ERROR,
            reason_code="waitcheck_invalid_result_json",
            details=str(error),
        )
    if len(payloads) != 1 or not isinstance(payloads[0], dict):
        return ParseResult(
            TOOL_ERROR,
            reason_code="waitcheck_result_record_invalid",
            details=f"expected one result object, observed {len(payloads)}",
        )
    payload = payloads[0]
    identity_matches = (
        payload.get("schema_version") == 1
        and payload.get("code_object_sha256") == expected_sha256
        and payload.get("target") == expected_target
        and payload.get("expected_kernel") == expected_kernel
        and payload.get("kernel_entry") == expected_entry
        and payload.get("inventory_attested") is True
    )
    if not identity_matches:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="waitcheck_artifact_identity_not_attested",
            details=json.dumps(payload, sort_keys=True),
        )
    if returncode != 0 or payload.get("api_status") != 0:
        return ParseResult(
            TOOL_ERROR,
            reason_code="waitcheck_analysis_failed",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    counts = (
        "instructions_analyzed",
        "memory_events_tracked",
        "kernels_discovered",
        "kernels_analyzed",
        "diagnostics_observed",
        "diagnostics_reported",
    )
    if any(
        isinstance(payload.get(name), bool)
        or not isinstance(payload.get(name), int)
        or int(payload[name]) < 0
        for name in counts
    ):
        return ParseResult(
            TOOL_ERROR,
            reason_code="waitcheck_invalid_result_counts",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    if (
        payload.get("analysis_complete") is not True
        or payload.get("diagnostics_truncated") is not False
        or payload.get("stopped_early") is not False
        or payload.get("kernels_discovered") != 1
        or payload.get("kernels_analyzed") != 1
    ):
        return ParseResult(
            INCONCLUSIVE,
            reason_code="waitcheck_analysis_incomplete",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != payload.get(
        "diagnostics_reported"
    ):
        return ParseResult(
            TOOL_ERROR,
            reason_code="waitcheck_diagnostic_count_mismatch",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    if payload.get("diagnostics_observed") != len(diagnostics):
        return ParseResult(
            INCONCLUSIVE,
            reason_code="waitcheck_diagnostics_not_lossless",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    findings: list[FindingRecord] = []
    for item in diagnostics:
        if not isinstance(item, dict):
            return ParseResult(
                TOOL_ERROR,
                reason_code="waitcheck_diagnostic_invalid",
                details=json.dumps(payload, sort_keys=True),
                attested=True,
            )
        kernel = item.get("kernel_name")
        entry = item.get("kernel_entry")
        if kernel != expected_kernel or entry != expected_entry:
            return ParseResult(
                INCONCLUSIVE,
                reason_code="waitcheck_diagnostic_kernel_mismatch",
                details=json.dumps(item, sort_keys=True),
                attested=True,
            )
        code = str(item.get("code") or "unknown")
        findings.append(
            FindingRecord(
                kind=f"wait-hazard-{code}",
                message=str(item.get("message") or "Waitcheck reported an AMDGPU wait hazard"),
                kernel=expected_kernel,
                location=(
                    f"{item.get('section_name', '.text')}+0x{int(item.get('section_offset', 0)):x}"
                ),
                raw=json.dumps(item, sort_keys=True),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"message", "kernel_name"}
                },
            )
        )
    if findings:
        if payload.get("passed") is not False:
            return ParseResult(
                TOOL_ERROR,
                reason_code="waitcheck_verdict_disagrees_with_diagnostics",
                details=json.dumps(payload, sort_keys=True),
                attested=True,
            )
        return ParseResult(
            FINDING,
            tuple(findings),
            "waitcheck_wait_hazard",
            attested=True,
        )
    if payload.get("passed") is not True:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="waitcheck_clean_verdict_missing",
            details=json.dumps(payload, sort_keys=True),
            attested=True,
        )
    return ParseResult(PASS, reason_code="waitcheck_clean", attested=True)


def _key_value_fields(payload: str, context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise ValueError(f"{context}: malformed field {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise ValueError(f"{context}: invalid or duplicate field {token!r}")
        result[key] = value
    return result


def _uint(fields: dict[str, str], name: str, context: str) -> int:
    value = fields.get(name, "")
    if not _UNSIGNED_RE.fullmatch(value):
        raise ValueError(f"{context}: {name} is not an unsigned count")
    return int(value)


def _boolean(fields: dict[str, str], name: str, context: str) -> bool:
    value = fields.get(name)
    if value not in {"true", "false"}:
        raise ValueError(f"{context}: {name} is not a boolean")
    return value == "true"


def _consan_records(text: str, kind: str) -> list[dict[str, str]]:
    first_field = _CONSAN_RECORD_FIRST_FIELD[kind]
    marker = f"{_CONSAN_PREFIX}{kind} {first_field}="
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        offset = line.find(marker)
        if offset >= 0:
            records.append(
                _key_value_fields(
                    f"{first_field}={line[offset + len(marker) :]}",
                    f"ConSan {kind} line {line_number}",
                )
            )
    return records


def _validate_coverage(records: list[dict[str, str]]) -> tuple[set[int], bool]:
    if not records:
        raise ValueError("missing ConSan coverage record")
    applicable_readers: set[int] = set()
    all_complete = True
    for index, fields in enumerate(records, 1):
        context = f"ConSan coverage record {index}"
        reader = _uint(fields, "reader", context)
        if fields.get("flavor") != "moi" or fields.get("engine") != "record_replay":
            raise ValueError(f"{context}: unexpected flavor or engine")
        derived_complete = True
        discovered_total = 0
        for kind in _CONSAN_SITE_KINDS:
            discovered = _uint(fields, f"{kind}_discovered", context)
            supported = _uint(fields, f"{kind}_supported", context)
            selected = _uint(fields, f"{kind}_selected", context)
            patched = _uint(fields, f"{kind}_patched", context)
            unsupported = _uint(fields, f"{kind}_unsupported", context)
            resource_failed = _uint(fields, f"{kind}_resource_failed", context)
            lowering_failed = _uint(
                fields, f"{kind}_placement_or_lowering_failed", context
            )
            expert_omitted = _uint(fields, f"{kind}_expert_limit_omitted", context)
            if discovered != supported + unsupported:
                raise ValueError(f"{context}: {kind} discovery accounting mismatch")
            if supported != selected + expert_omitted:
                raise ValueError(f"{context}: {kind} selection accounting mismatch")
            if selected != patched + resource_failed + lowering_failed:
                raise ValueError(f"{context}: {kind} patch accounting mismatch")
            derived_complete &= not any(
                (unsupported, resource_failed, lowering_failed, expert_omitted)
            )
            discovered_total += discovered
        reported_complete = _boolean(fields, "analysis_complete", context)
        if reported_complete != derived_complete:
            raise ValueError(f"{context}: completeness disagrees with counters")
        if _boolean(fields, "expert_limit", context):
            derived_complete = False
        if discovered_total:
            applicable_readers.add(reader)
            all_complete &= derived_complete
    return applicable_readers, all_complete


def _validate_verdict(records: list[dict[str, str]], applicable_count: int) -> bool:
    if len(records) != 1:
        raise ValueError(
            f"expected one ConSan process verdict, observed {len(records)}"
        )
    fields = records[0]
    context = "ConSan analysis verdict"
    for name in (
        "applicable",
        "analysis_complete",
        "static_complete",
        "dynamic_complete",
    ):
        if not _boolean(fields, name, context):
            return False
    if _uint(fields, "applicable_code_objects", context) != applicable_count:
        raise ValueError("ConSan applicable-code-object count mismatch")
    if _uint(fields, "incomplete_code_objects", context) != 0:
        return False
    for name in (
        "dynamic_incomplete",
        "replay_unsupported_access",
        "replay_unsupported_atomics",
        "replay_unsupported_fences",
        "replay_metadata_full",
    ):
        if _uint(fields, name, context) != 0:
            return False
    for kind in _CONSAN_SITE_KINDS:
        pair = fields.get(kind, "")
        match = _PAIR_RE.fullmatch(pair)
        if match is None or match.group(1) != match.group(2):
            return False
    return True


def parse_consan(
    stdout: str,
    stderr: str,
    returncode: Optional[int],
    *,
    expected_sha256: str,
    expected_fingerprint: str,
    timed_out: bool = False,
) -> ParseResult:
    """Parse strict record/replay evidence for one evaluator-selected HSACO."""

    combined = "\n".join(part for part in (stdout, stderr) if part)
    if timed_out:
        return ParseResult(TOOL_ERROR, reason_code="consan_timeout", details=combined)
    try:
        run_payloads = _json_payloads(combined, _CONSAN_RUN_PREFIX)
    except (json.JSONDecodeError, ValueError) as error:
        return ParseResult(
            TOOL_ERROR, reason_code="consan_invalid_run_json", details=str(error)
        )
    if len(run_payloads) != 1 or not isinstance(run_payloads[0], dict):
        return ParseResult(
            TOOL_ERROR,
            reason_code="consan_run_record_invalid",
            details=f"expected one run object, observed {len(run_payloads)}",
        )
    run = run_payloads[0]
    attested = (
        run.get("schema_version") == 1
        and run.get("code_object_sha256") == expected_sha256
        and run.get("code_object_fingerprint") == expected_fingerprint
        and run.get("mode") == "record-replay"
    )
    if not attested:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="consan_artifact_identity_not_attested",
            details=json.dumps(run, sort_keys=True),
        )
    if (
        returncode != 0
        or run.get("instrumented_returncode") != 0
        or run.get("oracle_returncode") != 0
        or run.get("oracle_passed") is not True
    ):
        return ParseResult(
            TOOL_ERROR,
            reason_code="consan_execution_or_oracle_failed",
            details=combined,
            attested=True,
        )

    try:
        patch_records = _consan_records(combined, "patch end")
        coverage_records = _consan_records(combined, "coverage")
        verdict_records = _consan_records(combined, "analysis verdict")
        report_records = _consan_records(combined, "MOI auto report")
        diagnostic_records = _consan_records(
            combined, "MOI auto replay diagnostic"
        )
        applicable_readers, coverage_complete = _validate_coverage(coverage_records)
        expected_reports = [
            record
            for record in report_records
            if record.get("code_object") == expected_fingerprint
            and _uint(record, "reader", "ConSan MOI report") in applicable_readers
        ]
        if len(expected_reports) != 1:
            raise ValueError(
                "expected exactly one dynamic report for the selected code object"
            )
        expected_reader = _uint(expected_reports[0], "reader", "ConSan MOI report")
        dispatch_pair = expected_reports[0].get("dispatch_tokens", "")
        dispatch_match = _PAIR_RE.fullmatch(dispatch_pair)
        if dispatch_match is None or int(dispatch_match.group(1)) < 1:
            raise ValueError("selected code object has no attested dispatch token")
        matching_patches = [
            record
            for record in patch_records
            if _uint(record, "reader", "ConSan patch end") == expected_reader
            and record.get("outcome") == "modified-valid"
            and record.get("modified") == "true"
            and _uint(record, "patches", "ConSan patch end") > 0
        ]
        if len(matching_patches) != 1:
            raise ValueError("selected code object lacks one modified-valid patch record")
        verdict_complete = _validate_verdict(
            verdict_records, len(applicable_readers)
        )
    except ValueError as error:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="consan_evidence_incomplete_or_invalid",
            details=str(error),
            attested=True,
        )

    findings: list[FindingRecord] = []
    for record in diagnostic_records:
        if record.get("code_object") != expected_fingerprint:
            continue
        if _uint(record, "reader", "ConSan diagnostic") != expected_reader:
            continue
        index = _uint(record, "index", "ConSan diagnostic")
        findings.append(
            FindingRecord(
                kind="lds-concurrency-race",
                message="rocJITsu ConSan reported conflicting LDS accesses",
                location=(
                    f"first_inst={record.get('first_inst')},"
                    f"second_inst={record.get('second_inst')}"
                ),
                raw=" ".join(f"{key}={value}" for key, value in record.items()),
                metadata={**record, "diagnostic_index": index},
            )
        )
    if findings:
        return ParseResult(
            FINDING,
            tuple(findings),
            "consan_lds_race",
            attested=True,
        )
    if not coverage_complete or not verdict_complete:
        return ParseResult(
            INCONCLUSIVE,
            reason_code="consan_analysis_incomplete",
            details=combined,
            attested=True,
        )
    return ParseResult(PASS, reason_code="consan_clean", attested=True)
