"""Task-local Event evidence must satisfy the validator's actual reader."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.task_validator.report_v2 import _replay_validation_applicability


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "batched_gemm_a8w8_kernel",
    "gemm_a16w8_blockscale_kernel",
    "gemm_a16wfp4_kernel",
    "gemm_a4w4_kernel",
    "gemm_a8w8_bpreshuffle_kernel",
    "gemm_a8w8_per_token_scale_kernel",
    "gemm_afp4wfp4_kernel",
    "gemm_afp8wfp8_kernel",
    "hgemm_kernel",
)


def _load_runtime(name):
    path = ROOT / "tasks/torch2flydsl" / name / "task_runtime.py"
    spec = importlib.util.spec_from_file_location(f"warn_runtime_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _applicability(row):
    spec = SimpleNamespace(
        candidate=SimpleNamespace(initial_state="unimplemented"),
        baseline=SimpleNamespace(kind="provided"),
    )
    result = SimpleNamespace(passed=True, cases=[row])
    return _replay_validation_applicability({
        "evidence_valid": True, "accepted": True, "spec": spec,
        "results": {("baseline", "performance"): result},
    })


@pytest.mark.parametrize("name", TASKS)
def test_event_case_keeps_original_fields_and_binds_real_device_timing(name):
    runtime = _load_runtime(name)
    original = {
        "test_case_id": "legacy_0", "execution_time_ms": 1.25,
        "benchmark_method": "cuda_event_fallback",
        "benchmark_target_ms": 1.0, "benchmark_samples": 100,
        "benchmark_effective_repeats": 1,
        "benchmark_fallback_reason": "capture_unsafe_aiter_hipblaslt",
        "benchmark_method_consistent": True,
        "timed_output_checked": True,
        "timed_output_correctness": "PASS", "replay_correctness": "PASS",
    }
    row = runtime.require_result_rows(
        [original], [{"test_case_id": "case_0000", "checks": ["performance"]}],
        {"legacy_0": "case_0000"},
    )[0]
    metadata = row["metadata"]
    assert metadata["benchmark_fallback_reason"] == original["benchmark_fallback_reason"]
    assert metadata["benchmark_samples"] == original["benchmark_samples"]
    assert metadata["timed_output_checked"] is True
    assert metadata["device_timing"]["benchmark_method"] == row["benchmark_method"]
    assert metadata["device_timing"]["benchmark_fallback_reason"] == metadata["benchmark_fallback_reason"]
    assert _applicability(row)["status"] == "not_applicable"

    for damaged in (
        {**row, "metadata": {**metadata, "timed_output_checked": False}},
        {**row, "metadata": {**metadata, "device_timing": {}}},
    ):
        assert _applicability(damaged)["status"] == "undetermined"
