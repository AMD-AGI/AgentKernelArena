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
