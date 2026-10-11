"""Real RMSNorm consumers must gate source attribution and complete raw timing."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest
import yaml

from src import evaluator, performance
from src.benchmark_quality import validate_case_quality
from src.quality_measurements import load_quality_measurements, quality_required
from src.score import task_result_scoring
from test_rmsnorm_graph_benchmark import environment, measure, raw_report


SOURCE = "source/minimax_m3_rmsnorm.py"


@pytest.fixture
def qwen(monkeypatch, tmp_path):
    env = environment(monkeypatch, tmp_path)
    root = tmp_path / "task"
    (root / "build").mkdir(exist_ok=True)
    return root, yaml.safe_load((root / "config.yaml").read_text()), env[1], raw_report(measure(env))


def source_change(root):
    path = root / SOURCE
    tree = ast.parse(path.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_gemma_fused_add_rmsnorm_kernel")
    kernel.body.insert(0, ast.parse("tuning_constant = 1").body[0])
    path.write_text(ast.unparse(tree) + "\n")


def emit(qwen, samples, request_id="test-current-performance"):
    root, config, runner, original = qwen
    raw = copy.deepcopy(original)
    for row in raw["cases"]:
        row.update(samples_ms=list(samples), mean_ms=statistics.mean(samples),
                   median_ms=statistics.median(samples), min_ms=min(samples))
    report = {"status": "ok", "timer": "cuda_graph", "warmup_iterations": 10,
              "benchmark_iterations": 100, "test_cases": runner._benchmark_cases(raw),
              "request": {"schema_version": 1, "phase": "performance", "request_id": request_id,
                          "source_sha256": {SOURCE: hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()},
                          "manifest_sha256": hashlib.sha256((root / "ut/meta.json").read_bytes()).hexdigest()}}
    (root / "build/_bench_raw.json").write_text(json.dumps(raw))
    (root / "build/performance_report.json").write_text(json.dumps(report))
    return report


@pytest.mark.parametrize("scenario", ["changed", "unchanged", "baseline_outlier", "candidate_outlier"])
def test_actual_qwen_measurement_comparison_and_saved_score_are_gated(qwen, monkeypatch, scenario):
    root, config, _, _ = qwen
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        samples = [2.0] * 100 if len(calls) == 1 else [1.0] * 100
        if (scenario == "baseline_outlier" and len(calls) == 1
                or scenario == "candidate_outlier" and len(calls) == 2):
            samples[27] = 10000.0
        emit(qwen, samples, "fresh-invocation-" + str(len(calls)))
        return True, "Performance: measured 2 cases", ""
    monkeypatch.setattr(performance, "run_command", run)
    monkeypatch.setattr(performance, "force_jit_rebuild", lambda *a, **k: {})
    monkeypatch.setattr(evaluator, "evaluate_compilation", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, "evaluate_correctness", lambda *a, **k: (True, None))
    baseline = performance.measure_baseline(root, config)
    if scenario != "unchanged":
        source_change(root)
    result = evaluator.evaluate_kernel(root, config, baseline)
    evaluator.write_task_result(root, result, baseline, "qwen-rmsnorm", "CPU-test", create_plots=False)
    saved = yaml.safe_load((root / "task_result.yaml").read_text())
    assert len(calls) == 2 and saved["valid_optimized_cases"] == 2
    if scenario == "changed":
        assert saved["speedup_ratio"] == 2.0 and task_result_scoring(str(root)) == 320
    else:
        assert saved["speedup_ratio"] is None and not saved["accepted_gain"]
        assert task_result_scoring(str(root)) == 120
    assert quality_required(config)
    assert saved["benchmark_quality_required"]
    for case in baseline:
        proof = validate_case_quality(case)
        assert len(proof["series"]["candidate"]["raw_samples_ms"]) == 100
        assert proof["source_sha256"] == baseline[0].metadata["benchmark_quality_case"]["source_sha256"]


@pytest.mark.parametrize("scenario", ["stable", "outlier", "stale_source", "missing_raw"])
def test_actual_qwen_framework_finalizer_requires_current_complete_quality(qwen, scenario):
    from agents.task_validator.report_schema import finalize_report, validation_report_is_complete
    from test_task_validator import _valid_raw_report
    root, _, _, _ = qwen
    samples = [1.0] * 100
    if scenario == "outlier":
        samples[27] = 10000.0
    emit(qwen, samples)
    if scenario == "stale_source":
        source_change(root)
    if scenario == "missing_raw":
        (root / "build/_bench_raw.json").unlink()
    (root / "validation_report.yaml").write_text(yaml.safe_dump(_valid_raw_report()))
    final = finalize_report(root, expected_task_name="hip2hip/example")
    assert validation_report_is_complete(root)
    check = final["checks"]["benchmark_integrity"]
    assert check["benchmark_quality_required"] is True
    assert check["benchmark_quality"]["accepted_gain"] is False
    if scenario == "stable":
        assert final["overall_status"] == "PASS" and check["case_count"] == 2
        assert check["benchmark_quality"]["comparison_status"] == "task_validation_only"
    else:
        assert final["overall_status"] == "FAIL" and check["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["missing_case", "trimmed_samples", "geometry", "validation",
                                      "source", "metadata", "report_disagrees", "validator",
                                      "unsupported_adapter", "unsupported_flag", "other_target"])
def test_qwen_adapter_retains_existing_protocol_checks_and_rejects_unbound_inputs(qwen, mutation):
    root, config, _, _ = qwen
    report = emit(qwen, [1.0] * 100)
    raw_path = root / "build/_bench_raw.json"
    raw = json.loads(raw_path.read_text())
    if mutation == "missing_case":
        raw["cases"].pop()
    elif mutation == "trimmed_samples":
        raw["cases"][0]["samples_ms"].pop()
    elif mutation == "geometry":
        raw["cases"][0]["params"]["m"] = 1
    elif mutation == "validation":
        raw["cases"][0]["validation"]["measured_graph_validated"] = False
    elif mutation == "source":
        report["request"]["source_sha256"][SOURCE] = "a" * 64
    elif mutation == "metadata":
        report["request"]["manifest_sha256"] = "a" * 64
    elif mutation == "report_disagrees":
        report["test_cases"][0]["execution_time_ms"] = 0.001
    elif mutation == "validator":
        with (root / "scripts/task_runner.py").open("a") as stream:
            stream.write("\nraise AssertionError('modified validator must not execute')\n")
    elif mutation == "unsupported_adapter":
        config["benchmark_quality"] = {"schema_version": 1, "adapter": "unsupported_qwen_contract"}
    elif mutation == "unsupported_flag":
        config["benchmark_quality"] = True
    else:
        config["source_file_path"] = ["source/other_qwen_kernel.py"]
    raw_path.write_text(json.dumps(raw))
    (root / "build/performance_report.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        load_quality_measurements(root, config)


@pytest.mark.parametrize("change_during_run", [False, True])
def test_actual_runner_emits_source_bound_request_and_rejects_mid_run_change(qwen, monkeypatch, change_during_run):
    root, config, runner, raw = qwen
    monkeypatch.setattr(runner, "candidate_overlay", lambda: str(root / "ut/_cand_overlay"))
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(root / "scripts/_aka_benchmark.py")))
    def run(*args, **kwargs):
        (root / "build/_bench_raw.json").write_text(json.dumps(raw))
        if change_during_run:
            source_change(root)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", run)
    cases = runner.run_performance(config, 1)
    report = json.loads((root / "build/performance_report.json").read_text())
    if change_during_run:
        assert not cases and report["status"] == "fail"
    else:
        assert len(cases) == 2 and report["request"]["phase"] == "performance"
        assert report["request"]["source_sha256"] == {SOURCE: hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()}
        assert report["request"]["manifest_sha256"] == hashlib.sha256((root / "ut/meta.json").read_bytes()).hexdigest()
