"""Cold graph measurements remain scoreable without waiving replay review."""
import json

import pytest

from src.task_protocol import RESULT_PREFIX
from tests.test_task_validator_v2 import context, draft, normalized, replace_stdout


def graph_context(tmp_path, method):
    ctx = context(tmp_path, state="implemented", baseline_kind="provided")
    for record in ctx["actions"]:
        if record["result"]["action"] != "performance":
            continue

        def change(report):
            for row in report["cases"]:
                row["benchmark_method"] = method
                if method == "cuda_graph_cold_cache":
                    row["metadata"] = {
                        "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
                        "benchmark_cache_flush_bytes": 1 << 30,
                        "benchmark_cache_flush_in_timing": False,
                        "benchmark_effective_repeats": 1,
                        "benchmark_metric": "gpu_event_graph_span",
                    }

        replace_stdout(record, change)
    return ctx


@pytest.mark.parametrize("method", ["cuda_graph_cold_weights", "cuda_graph_cold_cache"])
def test_trusted_cold_graph_measurements_can_complete_full_validation(tmp_path, method):
    ctx = graph_context(tmp_path, method)
    result = normalized(ctx)
    assert result["overall_status"] == "PASS", result["validation_errors"]
    check = result["checks"]["benchmark_integrity"]
    assert check["benchmark_methods"] == [method]
    applicability = check["replay_validation_applicability"]
    assert applicability["status"] == "required"
    assert applicability["roles"] == ["baseline", "candidate"]
    assert applicability["case_count"] == 4


@pytest.mark.parametrize("method", ["cuda_graph_cold_weights", "cuda_graph_cold_cache"])
def test_model_cannot_call_cold_graph_replay_review_not_applicable(tmp_path, method):
    ctx = graph_context(tmp_path, method)
    raw = draft(ctx)
    raw["checks"]["benchmark_integrity"].update(
        replay_validation_valid=None,
        replay_validation_applicability={"status": "not_applicable"},
    )
    result = normalized(ctx, raw)
    assert result["overall_status"] == "WARN"
    assert result["checks"]["benchmark_integrity"]["replay_validation_applicability"]["status"] == "required"
    assert "benchmark_integrity: replay_validation_valid is undetermined" in result["policy_findings"]


def test_cold_graph_measurement_does_not_override_asymmetric_work_review(tmp_path):
    ctx = graph_context(tmp_path, "cuda_graph_cold_cache")
    raw = draft(ctx)
    raw["checks"]["benchmark_integrity"]["workload_symmetric"] = False
    result = normalized(ctx, raw)
    assert result["checks"]["benchmark_integrity"]["status"] == "FAIL"
    assert result["overall_status"] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("benchmark_cache_policy", "warm"),
    ("benchmark_cache_flush_bytes", 0),
    ("benchmark_cache_flush_in_timing", True),
    ("benchmark_effective_repeats", 2),
    ("benchmark_metric", "host_wall_clock"),
])
def test_invalid_actual_cold_cache_identity_cannot_produce_a_pass(tmp_path, field, value):
    ctx = graph_context(tmp_path, "cuda_graph_cold_cache")
    record = next(r for r in ctx["actions"]
                  if r["result"]["role"] == "baseline" and r["result"]["action"] == "performance")
    command = record["commands"][0]
    report = json.loads(command["stdout"][len(RESULT_PREFIX):])
    report["cases"][0]["metadata"][field] = value
    command["stdout"] = RESULT_PREFIX + json.dumps(report)
    # The stored result still claims a valid identity. Actual command stdout
    # remains authoritative, even with a freshly bound model draft.
    result = normalized(ctx)
    assert result["overall_status"] == "FAIL"
