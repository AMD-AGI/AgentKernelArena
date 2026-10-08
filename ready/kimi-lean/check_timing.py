#!/usr/bin/env python3
"""Validate the complete frozen Lean comparison and assess its paired timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.tools import trusted_task_eval as trusted
from src.task_contract import canonical, fingerprint, require, strict_json, validate_report
from src.native_baseline import validate_paired_measurements
from quality_policy import assess_comparison

TASK = "tasks/headkernel/kimi-k3__lean_attention_decode"
COMMIT = "20c1a949b07e0d2530315ee666cae8a178edad69"
SOURCE = "source/decode_attention.py"
REFERENCE_SHA = "34ca0066e2716e1ffad048ec7cbb3a25af13eeb61e99034a663fed6277bf691a"
IMAGE = "docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96"
CONTRACT = trusted.package_contract(ROOT / TASK)
MANIFEST, CONFIG = CONTRACT["manifest"], CONTRACT["config"]
CASE_IDS = {row["case_id"] for row in MANIFEST["cases"]}
POLICY = trusted.scoring_policy(CONFIG)
PROVENANCE_PATH = ROOT / TASK / POLICY["native_source_manifest"]
PROVENANCE_BYTES = PROVENANCE_PATH.read_bytes()
PROVENANCE = strict_json(PROVENANCE_BYTES.decode())


def close(actual, expected, message):
    require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-12), message)


def validate_and_assess(measurement, reports):
    require(measurement["status"] == "measured" and measurement["full_case_coverage"] is True, "Incomplete trusted measurement")
    require(measurement["trusted_commit"] == COMMIT and measurement["task_path"] == TASK and measurement["image"] == IMAGE,
            "Wrong task, qualified commit or image")
    require(measurement["manifest_sha256"] == fingerprint(MANIFEST), "Case manifest differs")
    sources = measurement["source_sha256"]
    require(set(sources) == {"reference", "candidate"} and sources["reference"] == {SOURCE: REFERENCE_SHA}
            and all(set(source) == {SOURCE} for source in sources.values()), "Executable source boundary differs")
    require(set(reports) == {"reference", "candidate"}, "Both trusted outer legs are required")
    requests, seeds, packages, ordinary, paired, scored, qualities = set(), set(), {}, {}, {}, {}, {}
    for leg in ("reference", "candidate"):
        require(set(reports[leg]) == {"compile", "correctness", "performance"}
                and set(measurement["reports"][leg]) == set(reports[leg]), "All six trusted phases are required")
        for phase in ("compile", "correctness", "performance"):
            report = reports[leg][phase]; request = report["request"]
            require(request["phase"] == phase and request["source_sha256"] == sources[leg]
                    and request["gpu"] == measurement["gpu"], "Phase, source or physical GPU binding differs")
            measured = validate_report(report, MANIFEST, request)
            require(request["request_id"] not in requests, "Reused phase request")
            requests.add(request["request_id"]); seeds.add(request["challenge_seed"])
            packages.setdefault(leg, request["package_sha256"])
            require(packages[leg] == request["package_sha256"], "Task package changed between phases")
            compiled = report["compiled_specializations"]
            require(len(compiled) == len(CASE_IDS) and {row["case_id"] for row in compiled} == CASE_IDS, "Native specialization coverage differs")
            for row in compiled:
                require(row["source_sha256"] == sources[leg] and row["candidate_invoked"] is True
                        and row["independent_reference_invoked"] is True and row["captured_cpu_golden_parity"] is True,
                        "Native invocation or captured parity proof missing")
                require(row["candidate_callable"] != row["reference_callable"], "Private reference isolation missing")
            if phase == "performance":
                ordinary[leg] = {row["test_case_id"]: row for row in measured}
                # The frozen consumer validates every realized work schedule,
                # both 100-sample legs, all110 pairs and reference-output clears.
                # The trusted measurement supplies the submitted source hash.
                # The handoff contains the original source, so pass that bound
                # hash to the unchanged validator rather than treating the
                # handoff's original GPU body as the future candidate's body.
                require(PROVENANCE["reference_source_sha256"] == {SOURCE: REFERENCE_SHA}, "Protected source provenance differs")
                paired[leg] = validate_paired_measurements(report, MANIFEST, request, sources[leg],
                    hashlib.sha256(PROVENANCE_BYTES).hexdigest(), POLICY, PROVENANCE, task_root=ROOT / TASK)
                require(paired[leg]["comparison_protocol_version"] == 2 and paired[leg]["baseline_kind"] == "native_production",
                        "Paired native scoring policy differs")
                scored[leg] = trusted.as_test_cases(paired[leg], is_baseline=leg == "reference")
                comparison = []
                authoritative = {row["case_id"]: row for row in report["paired_reference_comparison"]["cases"]}
                for outer in report["cases"]:
                    pair = outer["paired_reference"]
                    require(canonical(pair) == canonical(authoritative[outer["case"]["case_id"]]),
                            "Duplicated paired evidence differs from the validated comparison")
                    lengths = [str(row["work_value"]) for row in pair["input_schedule"]["measured_inputs"]]
                    comparison.append({"case_id": pair["case_id"], "work_kind": "paired_variable",
                        "reference_samples_ms": pair["legs"]["protected_reference"]["samples_ms"],
                        "candidate_samples_ms": pair["legs"]["candidate_port"]["samples_ms"],
                        "reference_work_ids": lengths, "candidate_work_ids": lengths})
                qualities[leg] = assess_comparison(comparison, reference_source=sources["reference"], candidate_source=sources[leg])
    require(len(requests) == 6 and len(seeds) == 1, "Matched six-phase challenge coverage differs")
    summary = trusted.metric_summary(scored["reference"], scored["candidate"])
    for key, value in summary.items():
        require(canonical(measurement.get(key)) == canonical(value), "Native scoring summary differs: " + key)
    close(measurement["arithmetic_mean_speedup"], summary["native_speedup_ratio"], "Primary score replaced by an outer port ratio")
    require(len(measurement["cases"]) == len(CASE_IDS) and {row["test_case_id"] for row in measurement["cases"]} == CASE_IDS,
            "Summary cases differ")
    for row in measurement["cases"]:
        name = row["test_case_id"]
        native = next(item["execution_time_ms"] for item in paired["candidate"]["native"] if item["test_case_id"] == name)
        candidate = next(item["execution_time_ms"] for item in paired["candidate"]["candidate"] if item["test_case_id"] == name)
        require(row["case_sha256"] == ordinary["reference"][name]["case_sha256"] == ordinary["candidate"][name]["case_sha256"], "Case fingerprint differs")
        for key, expected in (("reference_ms", native), ("candidate_ms", candidate), ("speedup", native / candidate),
                              ("port_reference_ms", ordinary["reference"][name]["execution_time_ms"]),
                              ("port_candidate_ms", ordinary["candidate"][name]["execution_time_ms"])):
            close(row[key], expected, "Primary/outer means or ratio differ: " + key)
    quality = qualities["candidate"]
    quality["reference_source_control"] = qualities["reference"]
    if qualities["reference"]["status"] != "pass":
        quality.update(status="reject", comparison_status="rejected_timing_quality", gain_eligible=False,
                       accepted_gain=False, accepted_arithmetic_mean_speedup=None)
    quality.update(qualified_task=TASK, qualified_commit=COMMIT, all_six_reports_validated=True,
                   all_400_inner_samples_retained=True, quality_module_commit="47aaa88342ba07ef67f95ba7b3c348eb20f945e1",
                   quality_policy_scope="Post-evaluation analysis only; frozen20c task and measurement unchanged",
                   stability_limit="Work classes are verified sequence lengths. Singleton classes cannot establish repeated timing stability; a pass is not statistical proof of gain.")
    return quality


def read_and_assess(directory):
    directory = Path(directory)
    measurement = strict_json((directory / "trusted_measurement.json").read_text())
    reports = {}
    for leg in ("reference", "candidate"):
        reports[leg] = {}
        for phase in ("compile", "correctness", "performance"):
            item = measurement["reports"][leg][phase]
            require(item["file"] == leg + "_" + phase + ".json", "Unexpected phase path")
            path = directory / item["file"]
            require(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "Phase report hash differs")
            reports[leg][phase] = strict_json(path.read_text())
    path = directory / "fixtures_receipt.json"
    require(measurement["fixtures"] == {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, "Fixture receipt hash differs")
    receipt = strict_json(path.read_text()); descriptor_path = ROOT / TASK / "fixtures/EXTERNAL-MANIFEST.json"
    descriptor = strict_json(descriptor_path.read_text())
    require(receipt["trusted_commit"] == COMMIT and receipt["task_path"] == TASK and receipt["runtime_image"] == IMAGE
            and receipt["source"] == "verified_oci_download" and receipt["assets"] == descriptor["assets"]
            and receipt["manifest_sha256"] == hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
            and receipt["case_manifest_fingerprint"] == fingerprint(MANIFEST), "Fixture receipt identity or coverage differs")
    return validate_and_assess(measurement, reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); quality = read_and_assess(args.measurement_dir)
    quality["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    quality["quality_module_sha256"] = hashlib.sha256((HERE / "quality_policy.py").read_bytes()).hexdigest()
    quality["measurement_sha256"] = hashlib.sha256((Path(args.measurement_dir) / "trusted_measurement.json").read_bytes()).hexdigest()
    with Path(args.output).open("x") as stream:
        json.dump(quality, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({key: quality[key] for key in ("status", "comparison_status", "gain_eligible", "accepted_gain", "raw_arithmetic_mean_speedup")}))
    raise SystemExit(0 if quality["status"] == "pass" else 2)
