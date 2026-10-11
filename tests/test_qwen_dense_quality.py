"""Ordinary dense routing scores require complete, current native evidence."""
from copy import deepcopy
import json
import math
import statistics

import pytest
import yaml

from src import evaluator, performance
from src.quality_measurements import load_quality_measurements
from src.score import task_result_scoring
from test_qwen_dense_native_routing import contract, load, payload_fixture, workspace


@pytest.fixture
def dense(workspace):
    config = yaml.safe_load((workspace / "config.yaml").read_text())
    runner = load("_dense_complete_quality_runner", workspace / "scripts/task_runner.py")

    def emit(samples, run_id="a" * 48):
        request, raw = payload_fixture(workspace)
        request["run_id"] = run_id
        raw["request_sha256"] = contract.fingerprint(request)
        for row in raw["cases"]:
            row.update(samples_ms=list(samples), mean_ms=math.fsum(samples) / 100,
                       median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
        build = workspace / "build"
        build.mkdir(exist_ok=True)
        (build / "performance_request.json").write_text(json.dumps(request))
        (build / "_bench_raw.json").write_text(json.dumps(raw))
        archive = build / "native-routing-runs" / run_id / "raw.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps(raw))
        report = {"status": "ok", "protocol": contract.PROTOCOL, "timer": "cuda_graph",
                  "warmup_iterations": 10, "benchmark_iterations": 100, "request": deepcopy(request),
                  "test_cases": runner._benchmark_cases(raw),
                  "complete_raw_report": str(archive.relative_to(workspace))}
        (build / "performance_report.json").write_text(json.dumps(report))
        return raw, report, archive

    return workspace, config, emit


def change_route(root, *, alias=False):
    routes = contract.source_context(root)["routes"]
    key = "decode_m64_n32_k8192" if alias else "prefill_m16384_n4608_k8192"
    assert key in routes
    routes[key] = "torch"
    (root / "source/routing.py").write_text("def select_routes():\n    return " + repr(routes) + "\n")


@pytest.mark.parametrize("scenario", ["changed", "unchanged", "comment_only", "equivalent_alias",
                                      "baseline_outlier", "candidate_outlier"])
def test_dense_ordinary_measurement_comparison_and_saved_score(dense, monkeypatch, scenario):
    root, config, emit = dense
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        samples = [2.0 if len(calls) == 1 else 1.0] * 100
        if (scenario == "baseline_outlier" and len(calls) == 1
                or scenario == "candidate_outlier" and len(calls) == 2):
            samples[27] = 10000.0
        emit(samples, str(len(calls)) * 48)
        return True, "complete five-case native graph report", ""

    monkeypatch.setattr(performance, "run_command", run)
    monkeypatch.setattr(performance, "force_jit_rebuild", lambda *a, **k: {})
    monkeypatch.setattr(evaluator, "evaluate_compilation", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, "evaluate_correctness", lambda *a, **k: (True, None))
    baseline = performance.measure_baseline(root, config)
    if scenario == "comment_only":
        path = root / "source/routing.py"
        path.write_text("# same native configuration\n" + path.read_text())
    elif scenario != "unchanged":
        change_route(root, alias=scenario == "equivalent_alias")
    result = evaluator.evaluate_kernel(root, config, baseline)
    evaluator.write_task_result(root, result, baseline, "qwen-dense", "CPU-test", create_plots=False)
    saved = yaml.safe_load((root / "task_result.yaml").read_text())
    assert len(calls) == 2 and saved["valid_optimized_cases"] == 5
    assert saved["benchmark_quality_required"] is True
    if scenario == "changed":
        assert saved["speedup_ratio"] == 2.0 and task_result_scoring(str(root)) == 320
    else:
        assert saved["speedup_ratio"] is None and saved["accepted_gain"] is False
        assert task_result_scoring(str(root)) == 120


@pytest.mark.parametrize("attack", ["case", "samples", "backend", "config", "stale_source", "request",
                                    "published_row", "raw_missing", "mirror", "validator", "helper"])
def test_dense_consumer_rejects_incomplete_or_unbound_execution(dense, attack):
    root, config, emit = dense
    raw, report, archive = emit([1.0] * 100)
    if attack == "case":
        raw["cases"].pop()
    elif attack == "samples":
        raw["cases"][0]["samples_ms"].pop()
    elif attack == "backend":
        raw["cases"][0]["validation"]["native_binding"]["backend"] = "torch"
    elif attack == "config":
        raw["cases"][0]["validation"]["native_binding"]["native_config"]["solidx"] = 0
    elif attack == "stale_source":
        change_route(root)
    elif attack == "request":
        report["request"]["run_id"] = "b" * 48
    elif attack == "published_row":
        report["test_cases"][0]["execution_time_ms"] = 0.001
    elif attack in ("validator", "helper"):
        name = "task_runner.py" if attack == "validator" else "_routing_contract.py"
        with (root / "scripts" / name).open("a") as stream:
            stream.write("\nraise AssertionError('unbound validator must not execute')\n")
    archive.write_text(json.dumps(raw))
    (root / "build/_bench_raw.json").write_text(json.dumps(raw))
    if attack == "mirror":
        (root / "build/_bench_raw.json").write_text("{}")
    if attack == "raw_missing":
        archive.unlink()
    (root / "build/performance_report.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        load_quality_measurements(root, config)


@pytest.mark.parametrize("scenario", ["stable", "equivalent_alias", "outlier", "stale_source", "raw_missing"])
def test_dense_framework_finalizer_requires_complete_current_quality(dense, scenario):
    from agents.task_validator.report_schema import finalize_report, validation_report_is_complete
    from test_task_validator import _valid_raw_report
    root, _, emit = dense
    if scenario == "equivalent_alias":
        change_route(root, alias=True)
    samples = [1.0] * 100
    if scenario == "outlier":
        samples[27] = 10000.0
    _, _, archive = emit(samples)
    if scenario == "stale_source":
        change_route(root)
    if scenario == "raw_missing":
        archive.unlink()
    (root / "validation_report.yaml").write_text(yaml.safe_dump(_valid_raw_report()))
    result = finalize_report(root, expected_task_name="hip2hip/example")
    assert validation_report_is_complete(root)
    assert result["checks"]["benchmark_integrity"]["benchmark_quality"]["accepted_gain"] is False
    assert result["overall_status"] == ("PASS" if scenario in ("stable", "equivalent_alias") else "FAIL")
