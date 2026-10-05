"""Validate captured and supplemental native-replay provenance without changing source captures."""
from copy import deepcopy
from pathlib import Path

from evaluation_contract import canonical, fingerprint, require, strict_json


def normalized_schema(schema):
    result = deepcopy(schema)
    for launch in result["controls"]["launches"]:
        launch["compiled"].pop("num_warps", None)
        launch["compiled"].pop("num_stages", None)
    return result


def validate_replay(row, schema, representative_key, states):
    require(row.get("case_key") == schema["family"]+"-"+fingerprint(schema)[:24]
            and row.get("representative_case_key") == representative_key
            and row.get("provenance") == "native replay with captured operands"
            and row.get("source_sha256") == schema["source"]
            and row.get("launch_contract") == schema["controls"]["launches"]
            and row.get("tolerance") == 0.02 and row.get("all_recorded_states_passed") is True
            and row.get("timing_performed") is False, "supplemental native replay identity differs")
    proof = row["actual_native_launch"]
    launches = schema["controls"]["launches"]
    require(proof.get("source_sha256") == schema["source"]
            and proof.get("production_namespace_rebound") is False
            and proof.get("engaged_kernels") == [x["kernel"] for x in launches]
            and len(proof.get("last_launches", [])) == len(launches), "supplemental native source proof differs")
    for actual, expected in zip(proof["last_launches"], launches):
        require(all(actual.get(k) == expected[k] for k in ("kernel", "grid", "constexpr"))
                and all(actual.get(k) == expected["compiled"][k] for k in ("num_warps", "num_stages", "shared")),
                "supplemental actual launch differs")
    recorded = row.get("recorded_states", [])
    require(len(recorded) == len(states), "supplemental first/min/max state coverage differs")
    for actual, expected in zip(recorded, states):
        require(actual.get("fixture_sha256") == expected["fixture_sha256"]
                and actual.get("labels") == expected["representative_labels"]
                and actual.get("actual_tensor_controls") == expected["tensor_controls"]
                and all(actual.get(k) is True for k in ("captured_output_parity", "rank0_native_config_parity",
                    "independent_math_parity", "target_native_config_parity")), "supplemental state parity proof differs")


def validate_coverage(manifest, definition, root):
    from minimax_fixtures import file_hash, path_inside
    evidence = manifest["capture"]
    reference = evidence.get("coverage_certificate", {})
    path = path_inside(root, reference.get("path"))
    require(file_hash(path) == reference.get("sha256"), "coverage certificate changed")
    certificate = strict_json(path.read_text())
    require(certificate.get("schema") == "minimax-supplemented-coverage-v1"
            and certificate.get("status") == "PASS"
            and certificate.get("run_id") == evidence["run_id"]
            and certificate.get("runtime_image") == definition["runtime_image"]
            and certificate.get("exact_capture_fixture_coverage") is False
            and certificate.get("task_validator_status") == "not run at coverage sealing"
            and certificate.get("all_rank_metadata_verified") is True
            and certificate.get("exact_audits_and_tensor_controls_verified") is True,
            "supplemented coverage certificate is incomplete")
    variants = certificate["variants"]
    require(set(variants) == {c["source_capture_key"] for c in manifest["cases"]}, "coverage case set differs")
    for case in manifest["cases"]:
        row = variants[case["source_capture_key"]]
        schema = row["schema"]
        require(fingerprint(schema) == case["capture_schema_sha256"]
                and schema["controls"]["launches"] == case["launch_contract"]
                and schema["source"] == definition["source_sha256"]
                and row["occurrences"] == case["occurrences"]
                and row["work_distribution_sha256"] == fingerprint(case["work_distribution"])
                and row["states"] == case["states"], "coverage schema/work/state identity differs")
        transfer = case.get("fixture_transfer")
        if transfer:
            require(row["provenance"] == "native replay with captured operands"
                    and transfer.get("native_replay_validated") is True
                    and transfer.get("native_replay_required") is True
                    and transfer.get("captured_tensor_rank") == 0
                    and transfer.get("supplemental_report_sha256") == row["supplemental_report_sha256"],
                    "supplemental transfer provenance is missing")
            require(canonical(normalized_schema(schema)) == canonical(normalized_schema(row["representative_schema"]))
                    and schema != row["representative_schema"], "transfer changes more than recorded compiler choices")
            validate_replay(row["native_replay"], schema, transfer["representative_case_key"], case["states"])
        else:
            require(row["provenance"] == "direct captured native configuration", "direct fixture provenance differs")
    return certificate
