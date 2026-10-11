"""A route alias cannot earn a gain through the authoritative report consumer."""
import json

import yaml

from src.benchmark_quality import comparison_quality
from src.quality_measurements import load_quality_measurements
from test_qwen_dense_native_routing import contract, load, payload_fixture, workspace


def test_hk05_route_alias_is_unchanged_source_in_real_consumer(workspace):
    config = yaml.safe_load((workspace / "config.yaml").read_text())
    runner = load("_dense_alias_quality_runner", workspace / "scripts/task_runner.py")

    def publish(run_id, sample):
        request, raw = payload_fixture(workspace)
        request["run_id"] = run_id
        raw["request_sha256"] = contract.fingerprint(request)
        for row in raw["cases"]:
            row.update(samples_ms=[sample] * 100, mean_ms=sample, median_ms=sample,
                       min_ms=sample, max_ms=sample)
        build = workspace / "build"
        build.mkdir(exist_ok=True)
        (build / "performance_request.json").write_text(json.dumps(request))
        (build / "_bench_raw.json").write_text(json.dumps(raw))
        raw_path = build / "native-routing-runs" / run_id / "raw.json"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text(json.dumps(raw))
        rows = runner._benchmark_cases(raw)
        report = {"status": "ok", "protocol": contract.PROTOCOL, "timer": "cuda_graph",
                  "warmup_iterations": 10, "benchmark_iterations": 100, "request": request,
                  "test_cases": rows, "complete_raw_report": str(raw_path.relative_to(workspace)),
                  "methodology": "CPU consumer regression using complete protected producer reports",
                  "optimization_surface": "constrained image-native routing", "standalone_gain_claim": False}
        (build / "performance_report.json").write_text(json.dumps(report))
        return request, rows, load_quality_measurements(workspace, config)

    before_request, before_rows, baseline = publish("a" * 48, 2.0)
    routes = dict(before_request["source"]["routes"], decode_m64_n32_k8192="torch")
    (workspace / "source/routing.py").write_text("def select_routes():\n    return " + repr(routes) + "\n")
    after_request, after_rows, candidate = publish("b" * 48, 1.0)
    assert len(baseline) == len(candidate) == 5
    assert [row["validation"]["native_binding"]["native_config"] for row in before_rows] == [
        row["validation"]["native_binding"]["native_config"] for row in after_rows]
    assert before_request["source"]["source_file_sha256"] != after_request["source"]["source_file_sha256"]
    assert before_request["source"]["effective_source_sha256"] == after_request["source"]["effective_source_sha256"]
    quality = comparison_quality(baseline, candidate)
    assert quality["status"] == "pass" and quality["complete_case_count"] == 5
    assert quality["raw_arithmetic_mean_speedup"] == 2.0
    assert quality["comparison_status"] == "unchanged_source_control"
    assert quality["source_changed"] is False and quality["gain_eligible"] is False
    assert quality["accepted_gain"] is False and quality["accepted_arithmetic_mean_speedup"] is None
