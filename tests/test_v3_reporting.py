"""Results are keyed by (task, target language) across the report pipeline."""
import csv
import json
from pathlib import Path
from unittest import mock

import yaml

from src.evaluator import write_task_result
from src.postprocessing import general_post_processing
from src.tools.compare_runs import generate_comparison_report
from src.visualization import build_data

TASK = "triton/normalization/rms_norm/bf16_h8192"


def _run(root: Path, languages=("triton", "gluon"), speedup=1.25) -> Path:
    run = root / "workspace_MI355X_agent" / "run_20261105_120000"
    workspaces = []
    for language in languages:
        workspace = run / f"task_{TASK.replace('/', '_')}_{language}"
        workspace.mkdir(parents=True)
        write_task_result(
            workspace,
            {"pass_compilation": True, "pass_correctness": True, "best_optimized_execution_time": 10.0 / speedup,
             "average_speedup": speedup, "benchmark_method_consistent": True},
            [], TASK, "agent", create_plots=False,
            target_language=language, operator="rms_norm", operator_category="normalization")
        workspaces.append(str(workspace))
    general_post_processing(workspaces, logger=None)
    return run


def test_write_task_result_records_language_and_operator(tmp_path):
    write_task_result(tmp_path, {"pass_compilation": True, "pass_correctness": True}, [], TASK, "agent",
                      create_plots=False, target_language="hip", operator="rms_norm",
                      operator_category="normalization")
    result = yaml.safe_load((tmp_path / "task_result.yaml").read_text())
    assert (result["target_language"], result["operator"], result["operator_category"]) == (
        "hip", "rms_norm", "normalization")
    write_task_result(tmp_path, {"pass_compilation": True, "pass_correctness": True}, [], "e2e/x", "agent",
                      create_plots=False, target_language="gpu")
    result = yaml.safe_load((tmp_path / "task_result.yaml").read_text())
    assert result["operator"] is None and result["operator_category"] is None


def test_evaluate_task_session_passes_contract_to_task_result():
    from src import evaluator
    source = Path(evaluator.__file__).read_text()
    assert "operator_category=(contract.get(\"operator\")" in source


def test_reports_group_by_category_and_language(tmp_path):
    reports = _run(tmp_path) / "reports"
    summary = json.loads((reports / "task_type_breakdown.json").read_text())
    assert set(summary["task_types"]) == {"normalization"}
    assert summary["task_types"]["normalization"]["count"] == 2
    assert set(summary["target_languages"]) == {"triton", "gluon"}
    assert summary["target_languages"]["gluon"]["count"] == 1
    with (reports / "overall_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {(r["Task Name"], r["Target Language"]) for r in rows} == {(TASK, "triton"), (TASK, "gluon")}
    assert {r["Task Type"] for r in rows} == {"normalization"}
    assert {r["Operator"] for r in rows} == {"rms_norm"}
    text = (reports / "overall_report.txt").read_text()
    assert f"{TASK} [gluon]" in text and f"{TASK} [triton]" in text


def test_dashboard_keeps_one_row_per_task_and_language(tmp_path):
    run = _run(tmp_path)
    reports = run / "reports"
    status = build_data.load_status_map(reports / "overall_report.txt")
    assert set(status) == {f"{TASK}|triton", f"{TASK}|gluon"}
    local = tmp_path / ".visualization" / "reports" / "r1"
    local.parent.mkdir(parents=True)
    local.mkdir()
    for name in ("overall_summary.csv", "task_type_breakdown.json", "overall_report.txt"):
        (local / name).write_text((reports / name).read_text())
    patch = dict(PROJECT_ROOT=tmp_path.resolve(), REPORTS_ROOT=(tmp_path / ".visualization" / "reports").resolve())
    with mock.patch.multiple(build_data, **patch):
        dataset = build_data.build_dataset()
    assert dataset["meta"]["targetLanguages"] == ["gluon", "triton"]
    assert dataset["meta"]["taskCount"] == 2
    tasks = dataset["reports"][0]["tasks"]
    assert {(t["taskName"], t["targetLanguage"]) for t in tasks} == {(TASK, "triton"), (TASK, "gluon")}
    assert {t["displayName"] for t in tasks} == {f"{TASK} [triton]", f"{TASK} [gluon]"}
    assert dataset["taskTypeTotals"] == {"normalization": 2}


def test_compare_runs_includes_language_breakdown(tmp_path):
    first = _run(tmp_path / "a", speedup=1.0)
    second = _run(tmp_path / "b", speedup=1.5)
    report = generate_comparison_report(first, second, tmp_path / "out.txt")
    assert "TARGET LANGUAGE BREAKDOWN COMPARISON" in report
    assert "GLUON" in report and "TRITON" in report
