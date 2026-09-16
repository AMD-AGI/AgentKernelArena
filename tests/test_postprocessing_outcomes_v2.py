"""Synthetic reporting fixtures; these scores are not GPU validation evidence."""
import csv
import json

import pytest
import yaml

from src.postprocessing import general_post_processing
from src.visualization import build_data


@pytest.mark.parametrize("recover_from_text", [False, True])
def test_final_outcomes_survive_all_report_formats_without_changing_scores(
    tmp_path, monkeypatch, recover_from_text,
):
    run = tmp_path / "workspace_MI355X_codex" / "run_20260916_120000"
    # Legacy results have numerical evidence, but no evidence of final acceptance.
    fixtures = [
        ("SIKL-task/accepted", True, "COMPLETE", True, "ACCEPTED"),
        ("SIKL-task/export_failed", True, "INCOMPLETE", True, "INCOMPLETE"),
        ("SIKL-task/export_mutated", False, "INCOMPLETE", True, "NOT_ACCEPTED"),
        ("hip2hip/rejected", False, "NOT_ACCEPTED", False, "NOT_ACCEPTED"),
        ("hip2hip/legacy", None, None, True, "PASS"),
    ]
    workspaces = []
    for task_name, accepted, delivery, correct, _ in fixtures:
        workspace = run / task_name.replace("/", "_")
        workspace.mkdir(parents=True)
        result = {
            "task_name": task_name, "pass_compilation": True,
            "pass_correctness": correct, "base_execution_time": 10.0,
            "best_optimized_execution_time": 10.0,
            "benchmark_method_consistent": True, "speedup_ratio": 1.0 if correct else 0.0,
        }
        if accepted is not None:
            result.update(candidate_accepted=accepted, delivery_status=delivery)
        (workspace / "task_result.yaml").write_text(yaml.safe_dump(result))
        workspaces.append(str(workspace))

    general_post_processing(workspaces, logger=None)
    reports = run / "reports"
    summary = json.loads((reports / "task_type_breakdown.json").read_text())
    overall = summary["overall"]
    assert overall["total_score"] == 900
    assert overall["correctness_pass_count"] == 4
    assert overall["valid_speedup_count"] == 4
    expected_counts = {
        "candidate_accepted_count": 2, "candidate_rejected_count": 2,
        "candidate_acceptance_unknown_count": 1, "delivery_complete_count": 1,
        "delivery_incomplete_count": 2, "delivery_not_accepted_count": 1,
        "delivery_unknown_count": 1,
    }
    for key, count in expected_counts.items():
        assert overall[key] == count
        assert sum(group[key] for group in summary["task_types"].values()) == count
    assert summary["task_types"]["hip2hip"]["candidate_acceptance_unknown_count"] == 1
    assert summary["task_types"]["SIKL-task"]["candidate_rejected_count"] == 1

    csv_path = reports / "overall_summary.csv"
    with csv_path.open() as handle:
        rows = {row["Task Name"]: row for row in csv.DictReader(handle)}
    text_rows = build_data.load_status_map(reports / "overall_report.txt")
    for name, accepted, delivery, correct, status in fixtures:
        row = rows[name]
        assert row["Status"] == text_rows[name]["status"] == status
        assert row["Candidate Accepted"] == ("N/A" if accepted is None else "YES" if accepted else "NO")
        assert row["Delivery Status"] == (delivery or "N/A")
        assert float(row["Score"]) == text_rows[name]["score_from_report"] == (220 if correct else 20)
        assert text_rows[name]["candidateAccepted"] is accepted
        assert text_rows[name]["deliveryStatus"] == delivery

    if recover_from_text:
        # The dashboard supports recovering detail rows missing from the CSV.
        csv_path.write_text(csv_path.read_text().splitlines()[0] + "\n")
    monkeypatch.setattr(build_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(build_data, "REPORTS_ROOT", tmp_path / ".visualization/reports")
    dataset = build_data.build_dataset(include_workspace_runs=True)
    report, = dataset["reports"]
    assert report["overall"]["candidate_accepted_count"] == 2
    tasks = {task["taskName"]: task for task in report["tasks"]}
    for name, accepted, delivery, correct, status in fixtures:
        assert tasks[name]["status"] == status
        assert tasks[name]["candidateAccepted"] is accepted
        assert tasks[name]["deliveryStatus"] == delivery
        assert tasks[name]["score"] == (220 if correct else 20)
