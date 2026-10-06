"""Build a new portable FP4 fixture bundle from verified full-workload receipts."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from admission import POLICY, case_from_fixture, oracle_policy
from evaluation_contract import canonical, require, strict_json, validate_manifest
from fixture_codec import file_sha, safe_file, validate_fixture

COMMON_SHA256 = "8cff65ca8a74c4ee7565de5fddfbab92f4a5bb564c6348b6449eb2086da741ea"


def import_capture(common, receipt_path, policy_path, output):
    pins = strict_json((ROOT / "SOURCE-PROVENANCE.json").read_text())
    receipt_path, output = Path(receipt_path).resolve(), Path(output)
    receipt = strict_json(receipt_path.read_text())
    require(receipt.get("schema") == "minimax-fp4-full-workload-receipt-v1"
            and receipt.get("complete") is True and receipt.get("scope") == "full_served_workload"
            and receipt.get("image") == pins["runtime_image"], "verified full-workload owner receipt required")
    require(type(receipt.get("expected_requests")) is int and receipt["expected_requests"] > 0
            and receipt.get("successful_requests") == receipt["expected_requests"]
            and receipt.get("draining_completed") is True, "full request completion/draining proof missing")
    require(set(receipt["ranks"]) == {str(rank) for rank in range(8)}, "all eight TP rank receipts required")
    policy = oracle_policy(strict_json(Path(policy_path).read_text()))
    common = Path(common) / "runtime_capture.py"
    require(file_sha(common) == COMMON_SHA256, "wrong shared capture verifier")
    spec = importlib.util.spec_from_file_location("runtime_capture", common)
    cap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cap
    spec.loader.exec_module(cap)
    paths, rows, required, counts, schemas = [], {}, {}, defaultdict(dict), {}
    for rank_text, ref in receipt["ranks"].items():
        rank = int(rank_text)
        path = safe_file(receipt_path.parent, ref["path"])
        require(file_sha(path) == ref["sha256"], "rank receipt changed")
        row = strict_json(path.read_text())
        require(row["provenance"]["tp_rank"] == rank and row["provenance"]["image"] == pins["runtime_image"],
                "wrong rank/image receipt")
        require(row.get("sealed") is True and row.get("required_cases_supplied") is True
                and not row["failures"] and not row.get("additional_failure_count")
                and not row.get("unrepresented_metadata_case_notifications")
                and row.get("metadata_counts_include_all_notified_served_calls") is True,
                "rank capture is incomplete")
        selected = {key: schema for key, schema in row["case_schemas"].items() if schema["family"] == "minimax_fp4_gemm"}
        require({schema["stage"] for schema in selected.values()} == {"prefill", "decode"},
                "FP4 prefill/decode missing from a TP rank")
        for key, schema in selected.items():
            require(schema["source"] == pins["sources"]["source/kernel.py"]["sha256"], "wrong native FP4 source")
            count = row["runtime_case_counts"].get(key)
            require(type(count) is int and count > 0, "actual served case frequency missing")
            require(key not in schemas or canonical(schemas[key]) == canonical(schema), "cross-rank ABI disagrees")
            schemas[key] = schema
            counts[key][rank_text] = count
        required[rank] = list(selected)
        rows[rank], paths = row, paths + [path]
    proof = cap.verify_rank_manifests(paths, run_id=receipt["run_id"], required_ranks=tuple(range(8)),
                                      required_cases_by_rank=required)
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    (output / "provenance/owner-receipt.json").write_bytes(receipt_path.read_bytes())
    for path in paths:
        rank = strict_json(path.read_text())["provenance"]["tp_rank"]
        (output / "provenance" / ("rank-" + str(rank) + ".json")).write_bytes(path.read_bytes())
    fixture_dir = output / "fixtures"
    fixture_dir.mkdir()
    cases, copied = [], set()
    for key in sorted(schemas):
        reference = proof["fixture_representatives"][key]
        path = Path(reference["path"])
        require(file_sha(path) == reference["sha256"], "selected fixture changed")
        fixture = strict_json(path.read_text())
        validate_fixture(path.parent, fixture)
        require(fixture["source_sha256"] == schemas[key]["source"] and fixture["controls"] == schemas[key]["controls"],
                "fixture source/controls do not match observed cases")
        for phase in ("inputs", "outputs"):
            geometry = {name: None if meta is None else {k: v for k, v in meta.items() if k != "device"}
                        for name, meta in fixture[phase].items()}
            require(geometry == schemas[key][phase], "fixture geometry differs from full-workload metadata")
            for group in fixture["payload"][phase].values():
                for segment in group["segments"]:
                    relative = segment["blob"]
                    source = safe_file(path.parent, relative)
                    if relative not in copied:
                        target = fixture_dir / relative
                        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                                "blob path escapes portable fixture bundle")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        subprocess.run(["rclone", "copyto", str(source), str(target), "--transfers", "64000",
                                        "--progress", "--buffer-size", "0", "--config", "/dev/null"], check=True, timeout=1800)
                        copied.add(relative)
                    require(file_sha(fixture_dir / relative) == segment["sha256"], "portable blob copy differs")
        name = reference["sha256"] + ".json"
        (fixture_dir / name).write_bytes(path.read_bytes())
        cases.append(case_from_fixture(fixture, counts[key], {"path": "fixtures/" + name, "sha256": reference["sha256"]}))
    manifest = validate_manifest({"schema_version": 1, "runtime_image": pins["runtime_image"], "cases": cases,
                                  "measurement": POLICY, "oracle_policy": policy, "run_id": receipt["run_id"],
                                  "capture_scope": "full_served_workload", "owner_receipt_sha256": file_sha(receipt_path)})
    (output / "cases.json").write_text(canonical(manifest) + "\n")
    (output / "CAPTURE-ADMISSION.json").write_text(canonical({"status": "captured_not_qualified", "scoreable": False,
        "cases_sha256": file_sha(output / "cases.json"), "owner_receipt_sha256": file_sha(receipt_path),
        "case_count": len(cases), "occurrences": sum(case["occurrences"] for case in cases),
        "framework_task_validator_status": "not_run"}) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("common", "receipt", "oracle-policy", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import_capture(args.common, args.receipt, args.oracle_policy, args.output)


if __name__ == "__main__":
    main()
