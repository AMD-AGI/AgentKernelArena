"""Consumer checks use the actual recurrent/attention complete-report fixtures."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

import pytest
import yaml

from src import evaluator, performance
from src.benchmark_quality import validate_case_quality
from src.quality_measurements import load_quality_measurements
from src.score import task_result_scoring


REPO = Path(__file__).resolve().parents[1]
TASKS = {
    "recurrent": ("fused_recurrent_gated_delta_rule_decode", "test_qwen_recurrent_graph.py", "valid_report"),
    "attention": ("paged_attention_decode", "test_qwen_attention_graph.py", "complete_report"),
}


@pytest.fixture(params=TASKS)
def graph_task(request, tmp_path):
    name = request.param
    suffix, fixture_file, fixture_name = TASKS[name]
    original = REPO / "tasks/headkernel" / ("qwen3.8-2.4t__" + suffix)
    root = tmp_path / "task"
    shutil.copytree(original, root, symlinks=True,
                    ignore=shutil.ignore_patterns("build", "__pycache__", "reference_io.pt"))
    path = root / "scripts/_graph_contract.py"
    namespace = {"__file__": str(path), "__name__": "_quality_bound_graph_contract"}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    contract = SimpleNamespace(**namespace)
    # These existing fixture functions construct JSON/provenance only. Extract
    # them without importing the math tests, Torch, native modules or candidates.
    fixture_path = REPO / "tests" / fixture_file
    node = next(n for n in ast.parse(fixture_path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == fixture_name)
    fixture_globals = {"contract": contract, "TASK": root, "hashlib": hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(fixture_path), "exec"), fixture_globals)
    config = yaml.safe_load((root / "config.yaml").read_text())
    def emit(samples):
        raw = fixture_globals[fixture_name]()
        for row in raw["cases"]:
            row.update(samples_ms=list(samples), mean_ms=statistics.mean(samples),
                       median_ms=statistics.median(samples), min_ms=min(samples))
        report = {"status": "ok", "timer": "cuda_graph", "warmup_iterations": 10,
                  "benchmark_iterations": 100, "test_cases": contract.benchmark_cases(raw, root),
                  **{key: copy.deepcopy(raw[key]) for key in
                    ("source_sha256", "case_contract_sha256", "source_binding", "request")}}
        (root / "build").mkdir(exist_ok=True)
        raw_path = root / "build/_bench_raw.json"
        raw_path.write_text(json.dumps(raw))
        (root / "build/performance_report.json").write_text(json.dumps(report))
        return raw, report, raw_path
    return name, root, config, emit


def change_device_source(name, root, config):
    path = root / config["source_file_path"][0]
    if name == "recurrent":
        tree = ast.parse(path.read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "fused_recurrent_gated_delta_rule_packed_decode_kernel")
        kernel.body.insert(0, ast.parse("quality_test_value = 1").body[0])
        path.write_text(ast.unparse(tree) + "\n")
    else:
        source = path.read_text()
        target = re.search(r"\b_paged_attention_kernel\s*\(", source)
        start = source.index("{", target.start()) + 1
        path.write_text(source[:start] + "\n  int quality_test_value = 1;\n" + source[start:])


@pytest.mark.parametrize("scenario", ["changed", "unchanged", "baseline_outlier", "candidate_outlier"])
def test_bound_graph_ordinary_scoring_gates_all_cases(graph_task, monkeypatch, scenario):
    name, root, config, emit = graph_task
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        samples = [2.0] * 100 if len(calls) == 1 else [1.0] * 100
        if (scenario == "baseline_outlier" and len(calls) == 1
                or scenario == "candidate_outlier" and len(calls) == 2):
            samples[27] = 10000.0
        emit(samples)
        return True, "complete source-bound graph report", ""
    monkeypatch.setattr(performance, "run_command", run)
    monkeypatch.setattr(performance, "force_jit_rebuild", lambda *a, **k: {})
    monkeypatch.setattr(evaluator, "evaluate_compilation", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluator, "evaluate_correctness", lambda *a, **k: (True, None))
    baseline = performance.measure_baseline(root, config)
    if scenario != "unchanged":
        change_device_source(name, root, config)
    result = evaluator.evaluate_kernel(root, config, baseline)
    evaluator.write_task_result(root, result, baseline, "qwen-" + name, "CPU-test", create_plots=False)
    saved = yaml.safe_load((root / "task_result.yaml").read_text())
    assert len(calls) == 2 and saved["valid_optimized_cases"] == (2 if name == "recurrent" else 1)
    if scenario == "changed":
        assert saved["speedup_ratio"] == 2.0 and task_result_scoring(str(root)) == 320
    else:
        assert saved["speedup_ratio"] is None and not saved["accepted_gain"]
        assert task_result_scoring(str(root)) == 120
    for case in baseline:
        assert len(validate_case_quality(case)["series"]["candidate"]["raw_samples_ms"]) == 100
        assert case.metadata["benchmark_measurement_identity_kind"] == "complete_bound_graph_report_sha256"


@pytest.mark.parametrize("attack", ["case", "sample", "ABI", "state_or_scratch", "source_binding",
                                    "published_row", "stale_source", "case_contract", "validator", "raw_missing",
                                    "published_request", "missing_request", "mixed_error"])
def test_bound_graph_adapter_preserves_exact_state_ABI_and_source_checks(graph_task, attack):
    name, root, config, emit = graph_task
    raw, report, path = emit([1.0] * 100)
    if attack == "case":
        raw["cases"].pop()
    elif attack == "sample":
        raw["cases"][0]["samples_ms"].pop()
    elif attack == "ABI":
        raw["cases"][0]["params"]["B" if name == "recurrent" else "block_size"] = -1
    elif attack == "state_or_scratch":
        raw["cases"][0]["validation"]["restored_buffers" if name == "recurrent" else "poisoned_buffers"] = []
    elif attack == "source_binding":
        raw["source_binding"]["device_source_sha256"] = "0" * 64
        report["source_binding"] = copy.deepcopy(raw["source_binding"])
    elif attack == "published_row":
        report["test_cases"][0]["execution_time_ms"] = 0.0001
    elif attack == "stale_source":
        change_device_source(name, root, config)
    elif attack == "case_contract":
        contract_path = root / "ut/benchmark_contract.json"
        policy = json.loads(contract_path.read_text())
        policy["tolerance"] = 0.2
        contract_path.write_text(json.dumps(policy))
    elif attack == "validator":
        with (root / "scripts/_graph_contract.py").open("a") as stream:
            stream.write("\nraise AssertionError('unbound validator must not execute')\n")
    elif attack == "published_request":
        report["request"]["request_id"] = "d" * 48
    elif attack == "missing_request":
        raw.pop("request")
    elif attack == "mixed_error":
        raw["cases"][0]["validation"]["max_mixed_error_ratio"] = 1.01
    path.write_text(json.dumps(raw))
    if attack == "raw_missing":
        path.unlink()
    (root / "build/performance_report.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        load_quality_measurements(root, config)


@pytest.mark.parametrize("scenario", ["stable", "outlier", "stale_source", "state_or_scratch", "raw_missing"])
def test_bound_graph_finalizer_requires_current_complete_validated_state(graph_task, scenario):
    from agents.task_validator.report_schema import finalize_report
    from test_task_validator import _valid_raw_report
    name, root, config, emit = graph_task
    samples = [1.0] * 100
    if scenario == "outlier":
        samples[27] = 10000.0
    raw, _, path = emit(samples)
    if scenario == "stale_source":
        change_device_source(name, root, config)
    elif scenario == "state_or_scratch":
        raw["cases"][0]["validation"]["restored_buffers" if name == "recurrent" else "poisoned_buffers"] = []
        path.write_text(json.dumps(raw))
    elif scenario == "raw_missing":
        path.unlink()
    (root / "validation_report.yaml").write_text(yaml.safe_dump(_valid_raw_report()))
    final = finalize_report(root, expected_task_name="hip2hip/example")
    quality = final["checks"]["benchmark_integrity"]["benchmark_quality"]
    assert quality["accepted_gain"] is False
    assert final["overall_status"] == ("PASS" if scenario == "stable" else "FAIL")
