import json

from src.eval_tools.plugins.base import FINDING, INCONCLUSIVE, PASS, TOOL_ERROR
from src.eval_tools.plugins.parsers import (
    parse_fpsan_comparison,
    parse_gpu_asan,
    parse_rocjitsu,
)


def test_gpu_asan_extracts_memory_finding():
    stderr = "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234"
    result = parse_gpu_asan("", stderr, 1, attested=True)
    assert result.status == FINDING
    assert result.findings[0].kind == "heap-buffer-overflow"


def test_gpu_asan_clean_requires_build_attestation():
    assert parse_gpu_asan("safe", "", 0, attested=False).status == INCONCLUSIVE
    assert parse_gpu_asan("safe", "", 0, attested=True).status == PASS


def test_rocjitsu_parses_structured_race_block():
    report = '''
[rocjitsu] Kernel dispatch: "lds_race_kernel_0"
RACE type=LDS reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508
  ==> ds_write_b32 v2, v0
  ==> ds_read_b32 v0, v0
END_RACE
'''
    result = parse_rocjitsu("", "", 0, attested=True, report_text=report)
    assert result.status == FINDING
    assert result.findings[0].kind == "lds-race"
    assert result.findings[0].kernel == "lds_race_kernel_0"
    assert result.findings[0].metadata["register_or_byte"] == 508


def test_rocjitsu_clean_requires_dispatch_attestation():
    assert parse_rocjitsu("done", "", 0, attested=False).status == INCONCLUSIVE
    result = parse_rocjitsu('[rocjitsu] Kernel dispatch: "safe"', "", 0, attested=True)
    assert result.status == PASS


def fpsan_line(reference="abc", candidate="abc", instrumented=True):
    return "AKA_FPSAN_RESULT " + json.dumps(
        {"instrumented": instrumented, "reference_digest": reference, "candidate_digest": candidate}
    )


def test_fpsan_parser_detects_semantic_mismatch():
    result = parse_fpsan_comparison(fpsan_line("abc", "def"), "", 0, attested=True)
    assert result.status == FINDING
    assert result.findings[0].kind == "floating-point-semantic-mismatch"


def test_fpsan_parser_requires_comparison_and_attestation():
    assert parse_fpsan_comparison(fpsan_line(), "", 0, attested=False).status == INCONCLUSIVE
    assert parse_fpsan_comparison("ordinary correctness pass", "", 0, attested=True).status == INCONCLUSIVE
    assert parse_fpsan_comparison(fpsan_line(), "", 0, attested=True).status == PASS
    assert parse_fpsan_comparison("", "crash", 2, attested=True).status == TOOL_ERROR
