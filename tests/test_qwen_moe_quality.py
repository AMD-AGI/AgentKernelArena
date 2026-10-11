"""Ordinary MoE scoring consumes the real protected native-routing contract."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import pytest
import yaml

from src import evaluator, performance
from src.benchmark_quality import validate_case_quality
from src.harness_guard import describe_workspace_harness
from src.quality_measurements import load_quality_measurements
from src.score import task_result_scoring


TASK = Path(__file__).resolve().parents[1] / "tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4"


@pytest.fixture
def moe(tmp_path, monkeypatch):
    root = tmp_path / "task"
    shutil.copytree(TASK, root, symlinks=True,
                    ignore=shutil.ignore_patterns("build", "__pycache__", "reference_io.pt"))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    contract = load("_routing_contract", "_routing_contract.py")
    load("_moe_runtime", "_moe_runtime.py")
    load("_bench", "_bench.py")
    fixture = load("_moe_quality_cpu_fixture", "test_native_routing.py")
    config = yaml.safe_load((root / "config.yaml").read_text())
    def emit(samples, run_id="a" * 48):
        raw, _ = fixture.ContractTests().payload()
        request = contract.make_request(root, "performance", run_id)
        raw.update(request=request, request_sha256=contract.fingerprint(request))
        for row, spec in zip(raw["cases"], contract.specs(fixture.META)):
            row.update(selected_route=request["source"]["routes"][spec["sig"]],
                       samples_ms=list(samples), mean_ms=math.fsum(samples) / 100,
                       median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
        rows = contract.validate_payload(raw, fixture.META, request)
        report = {**copy.deepcopy(raw), "test_cases": [
            dict(row, test_case_id=row["sig"], execution_time_ms=row["mean_ms"]) for row in rows]}
        path = root / "build/reports" / run_id / "raw.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
        (root / "build/performance_report.json").write_text(json.dumps(report))
        return raw, report, path
    return root, config, emit


@pytest.mark.parametrize("scenario", ["changed_route", "unchanged", "comment_only", "baseline_outlier", "candidate_outlier"])
def test_moe_ordinary_scoring_requires_effective_route_change_and_complete_quality(moe, monkeypatch, scenario):
    root, config, emit = moe
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        samples = [2.0] * 100 if len(calls) == 1 else [1.0] * 100
        if (scenario == "baseline_outlier" and len(calls) == 1
                or scenario == "candidate_outlier" and len(calls) == 2):
            samples[27] = 10000.0
        emit(samples, str(len(calls)) * 48)
        return True, "Performance: complete two-case graph report", ""
    monkeypatch.setattr(performance, "run_command", run)
    monkeypatch.setattr(performance, "force_jit_rebuild", lambda *a, **k: {})
    monkeypatch.setattr(evaluator, "evaluate_compilation", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, "evaluate_correctness", lambda *a, **k: (True, None))
    baseline = performance.measure_baseline(root, config)
    source = root / "source/routing.py"
    if scenario == "comment_only":
        source.write_text("# formatting is not a different GPU route\n" + source.read_text())
    elif scenario != "unchanged":
        source.write_text(source.read_text().replace('"captured"', '"block32"', 1))
    result = evaluator.evaluate_kernel(root, config, baseline)
    evaluator.write_task_result(root, result, baseline, "qwen-moe", "CPU-test", create_plots=False)
    saved = yaml.safe_load((root / "task_result.yaml").read_text())
    assert len(calls) == 2 and saved["valid_optimized_cases"] == 2
    if scenario == "changed_route":
        assert saved["speedup_ratio"] == 2.0 and task_result_scoring(str(root)) == 320
    else:
        assert saved["speedup_ratio"] is None and not saved["accepted_gain"]
        assert task_result_scoring(str(root)) == 120
    for case in baseline:
        proof = validate_case_quality(case)
        assert len(proof["series"]["candidate"]["raw_samples_ms"]) == 100
        assert case.metadata["benchmark_source_identity_kind"] == "validated_native_routing"
        assert "source/routing.py" in case.metadata["benchmark_source_file_sha256"]


@pytest.mark.parametrize("mutation", ["missing_case", "samples", "shape", "state_validation", "selected_route",
                                      "source", "effective_source", "published_case", "validator", "missing_raw"])
def test_moe_quality_rejects_partial_or_unbound_real_producer_reports(moe, mutation):
    root, config, emit = moe
    raw, report, path = emit([1.0] * 100)
    if mutation == "missing_case":
        raw["cases"].pop()
    elif mutation == "samples":
        raw["cases"][0]["samples_ms"].pop()
    elif mutation == "shape":
        raw["cases"][0]["params"]["topk"] = 1
    elif mutation == "state_validation":
        raw["cases"][0]["validation"]["routes_weights_scales_preserved"] = False
    elif mutation == "selected_route":
        raw["cases"][0]["selected_route"] = "block32"
    elif mutation == "source":
        (root / "source/routing.py").write_text("import os\n")
    elif mutation == "effective_source":
        report["request"]["source"]["effective_source_sha256"] = "0" * 64
    elif mutation == "published_case":
        report["test_cases"][0]["execution_time_ms"] = 0.0001
    elif mutation == "validator":
        with (root / "scripts/_routing_contract.py").open("a") as stream:
            stream.write("\nraise AssertionError('changed validator must not execute')\n")
    path.write_text(json.dumps(raw))
    if mutation == "missing_raw":
        path.unlink()
    (root / "build/performance_report.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        load_quality_measurements(root, config)


@pytest.mark.parametrize("scenario", ["stable", "outlier", "stale_route", "missing_raw"])
def test_moe_framework_finalizer_independently_enforces_quality(moe, scenario):
    from agents.task_validator.report_schema import finalize_report
    from test_task_validator import _valid_raw_report
    root, _, emit = moe
    samples = [1.0] * 100
    if scenario == "outlier":
        samples[27] = 10000.0
    _, _, raw_path = emit(samples)
    if scenario == "stale_route":
        source = root / "source/routing.py"
        source.write_text(source.read_text().replace('"captured"', '"block32"', 1))
    if scenario == "missing_raw":
        raw_path.unlink()
    (root / "validation_report.yaml").write_text(yaml.safe_dump(_valid_raw_report()))
    final = finalize_report(root, expected_task_name="hip2hip/example")
    quality = final["checks"]["benchmark_integrity"]["benchmark_quality"]
    assert quality["accepted_gain"] is False
    assert final["overall_status"] == ("PASS" if scenario == "stable" else "FAIL")


def test_moe_adapter_inputs_are_in_the_authoritative_guard(moe):
    root, config, _ = moe
    protected = set(describe_workspace_harness(root)["protected_paths"])
    assert {"scripts/_routing_contract.py", "scripts/_moe_runtime.py", "ut/cases.py", "ut/meta.json"} <= protected
    assert "source/routing.py" not in protected
    assert config["benchmark_quality"]["validator_sha256"] == hashlib.sha256(
        (root / "scripts/_routing_contract.py").read_bytes()).hexdigest()
