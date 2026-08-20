import json

from src.eval_tools.plugins.base import FINDING, INCONCLUSIVE, PASS, TOOL_ERROR
from src.eval_tools.plugins.parsers import (
    parse_consan,
    parse_fpsan_comparison,
    parse_gpu_asan,
    parse_rocjitsu,
    parse_waitcheck,
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


def test_rocjitsu_parser_rejects_unstructured_dispatch_even_if_pre_attested():
    result = parse_rocjitsu(
        'Kernel dispatch: "safe"\nrocjitsu launcher output',
        "",
        0,
        attested=True,
    )
    assert result.status == INCONCLUSIVE
    assert result.reason_code == "rocjitsu_no_dispatch_observed"


def test_rocjitsu_deduplicates_races_repeated_across_sinks():
    race = '''
[rocjitsu] Kernel dispatch: "kernel"
RACE type=LDS reg=12 wave=0 lane=1 wg=0,0,0 conflict=unknown
END_RACE
'''
    result = parse_rocjitsu("", race, 0, attested=True, report_text=race)
    assert result.status == FINDING
    assert len(result.findings) == 1


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
    assert parse_fpsan_comparison("ordinary correctness pass", "", 0, attested=True).status == TOOL_ERROR
    assert parse_fpsan_comparison(fpsan_line(), "", 0, attested=True).status == PASS
    assert parse_fpsan_comparison("", "crash", 2, attested=True).status == TOOL_ERROR


def test_fpsan_parser_rejects_multiple_result_records():
    result = parse_fpsan_comparison(
        "\n".join((fpsan_line("abc", "def"), fpsan_line("abc", "abc"))),
        "",
        0,
        attested=True,
    )
    assert result.status == TOOL_ERROR
    assert result.reason_code == "fpsan_multiple_results"


def test_fpsan_parser_never_reports_clean_after_process_failure():
    for returncode in (139, None):
        result = parse_fpsan_comparison(
            fpsan_line(), "", returncode, attested=True
        )
        assert result.status == TOOL_ERROR
        assert result.reason_code == "fpsan_process_failed"


def waitcheck_line(*, diagnostics=None, passed=True, complete=True):
    diagnostics = diagnostics or []
    payload = {
        "schema_version": 1,
        "code_object_sha256": "a" * 64,
        "target": "gfx950",
        "expected_kernel": "kernel",
        "kernel_entry": 64,
        "inventory_attested": True,
        "api_status": 0,
        "analysis_complete": complete,
        "instructions_analyzed": 8,
        "memory_events_tracked": 1,
        "kernels_discovered": 1,
        "kernels_analyzed": 1,
        "diagnostics_observed": len(diagnostics),
        "diagnostics_reported": len(diagnostics),
        "diagnostics_truncated": False,
        "stopped_early": False,
        "passed": passed,
        "diagnostics": diagnostics,
    }
    return "AKA_WAITCHECK_RESULT " + json.dumps(payload)


def parse_waitcheck_fixture(text, returncode=0):
    return parse_waitcheck(
        text,
        "",
        returncode,
        expected_sha256="a" * 64,
        expected_target="gfx950",
        expected_kernel="kernel",
        expected_entry=64,
    )


def test_waitcheck_clean_requires_lossless_complete_exact_kernel_analysis():
    assert parse_waitcheck_fixture(waitcheck_line()).status == PASS
    incomplete = parse_waitcheck_fixture(waitcheck_line(complete=False))
    assert incomplete.status == INCONCLUSIVE
    assert incomplete.reason_code == "waitcheck_analysis_incomplete"


def test_waitcheck_returns_structured_hazard():
    diagnostic = {
        "code": "wait-counter",
        "kernel_name": "kernel",
        "kernel_entry": 64,
        "section_name": ".text",
        "section_offset": 80,
        "message": "missing s_wait_kmcnt <= 0",
    }
    result = parse_waitcheck_fixture(
        waitcheck_line(diagnostics=[diagnostic], passed=False)
    )
    assert result.status == FINDING
    assert result.findings[0].kind == "wait-hazard-wait-counter"
    assert result.findings[0].kernel == "kernel"


def _coverage_line():
    values = {
        "reader": "7",
        "flavor": "moi",
        "engine": "record_replay",
        "analysis_complete": "true",
        "expert_limit": "false",
    }
    for kind in ("access", "barrier", "atomic", "fence"):
        discovered = 1 if kind == "access" else 0
        values.update(
            {
                f"{kind}_discovered": str(discovered),
                f"{kind}_supported": str(discovered),
                f"{kind}_selected": str(discovered),
                f"{kind}_patched": str(discovered),
                f"{kind}_unsupported": "0",
                f"{kind}_resource_failed": "0",
                f"{kind}_placement_or_lowering_failed": "0",
                f"{kind}_expert_limit_omitted": "0",
            }
        )
    values["load"] = "1"
    return "[rocjitsu-dbi-hooks] ConSan coverage " + " ".join(
        f"{key}={value}" for key, value in values.items()
    )


def consan_log(*, diagnostic=False, complete=True, oracle=True):
    fingerprint = "fnv1a64:0123456789abcdef"
    run = "AKA_CONSAN_RUN " + json.dumps(
        {
            "schema_version": 1,
            "code_object_sha256": "b" * 64,
            "code_object_fingerprint": fingerprint,
            "mode": "record-replay",
            "instrumented_returncode": 0,
            "oracle_returncode": 0 if oracle else 1,
            "oracle_passed": oracle,
        }
    )
    patch = (
        "[rocjitsu-dbi-hooks] ConSan patch end reader=7 visited=true "
        "modified=true outcome=modified-valid errors=0 warnings=0 patches=3 patch_ms=1.0"
    )
    report = (
        "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=7 "
        f"code_object={fingerprint} dispatch_tokens=1/2048"
    )
    true_or_false = "true" if complete else "false"
    incomplete = "0" if complete else "1"
    verdict = (
        "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
        f"analysis_complete={true_or_false} static_complete={true_or_false} "
        f"dynamic_complete={true_or_false} applicable_code_objects=1 "
        f"incomplete_code_objects={incomplete} access=1/1 barrier=0/0 "
        "atomic=0/0 fence=0/0 visible_evidence=1 "
        f"dynamic_incomplete={incomplete} record_replay_bank_saturation=0 "
        "record_replay_invalid_site_tokens=0 replay_unsupported_access=0 "
        "replay_unsupported_atomics=0 replay_unsupported_fences=0 replay_metadata_full=0"
    )
    report_setup_noise = (
        "[rocjitsu-dbi-hooks] ConSan MOI auto report plan reader=7 "
        "outcome=complete required_bytes=4096"
    )
    report_buffer_noise = (
        "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=7 "
        "bytes=4096 allocation_outcome=allocated"
    )
    lines = [
        patch,
        _coverage_line(),
        report_setup_noise,
        report_buffer_noise,
        report,
    ]
    if diagnostic:
        lines.append(
            "[rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=7 "
            f"index=0 kind=1 code_object={fingerprint} first_inst=0x40 "
            "second_inst=0x48 first_lds=[0,4) second_lds=[0,4)"
        )
    lines.extend((verdict, run))
    return "\n".join(lines)


def parse_consan_fixture(text, returncode=0):
    return parse_consan(
        text,
        "",
        returncode,
        expected_sha256="b" * 64,
        expected_fingerprint="fnv1a64:0123456789abcdef",
    )


def test_consan_clean_requires_complete_coverage_dispatch_and_oracle():
    assert parse_consan_fixture(consan_log()).status == PASS
    assert parse_consan_fixture(consan_log(complete=False)).status == INCONCLUSIVE
    failed_oracle = parse_consan_fixture(consan_log(oracle=False), returncode=1)
    assert failed_oracle.status == TOOL_ERROR


def test_consan_reports_only_exact_code_object_diagnostics():
    result = parse_consan_fixture(consan_log(diagnostic=True))
    assert result.status == FINDING
    assert result.findings[0].kind == "lds-concurrency-race"
    unrelated = consan_log(diagnostic=True).replace(
        "code_object=fnv1a64:0123456789abcdef first_inst",
        "code_object=fnv1a64:ffffffffffffffff first_inst",
    )
    assert parse_consan_fixture(unrelated).status == PASS
