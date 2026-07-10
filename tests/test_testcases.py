# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for src/testcases.py pure helpers (CPU-only, no GPU deps)."""
# Import the dataclass under an alias so pytest does not try to collect
# `TestCaseResult` as a test class (its name starts with "Test").
from src.testcases import TestCaseResult as Case
from src.testcases import (
    _extract_time_from_dict,
    _safe_float,
    calculate_average_speedup,
    collect_benchmark_methods,
    load_performance_results,
    parse_test_cases_from_stdout,
    save_performance_results,
)


class TestSafeFloat:
    def test_valid_number(self):
        assert _safe_float("3.5") == 3.5
        assert _safe_float(2) == 2.0

    def test_none_and_invalid(self):
        assert _safe_float(None) is None
        assert _safe_float("abc") is None


class TestExtractTime:
    def test_prefers_ms_key(self):
        time_ms, key = _extract_time_from_dict({"execution_time_ms": 12.5})
        assert time_ms == 12.5
        assert key == "execution_time_ms"

    def test_seconds_converted_to_ms(self):
        # A bare 'time' >= 1000 is treated as seconds and converted to ms.
        time_ms, key = _extract_time_from_dict({"time": 2000.0})
        assert time_ms == 2000.0 * 1000.0
        assert key == "time"

    def test_torch2hip_baseline_uses_ori_time(self):
        time_ms, key = _extract_time_from_dict(
            {"ori_time": 9.0, "opt_time": 3.0}, is_baseline=True, task_type="torch2hip"
        )
        assert time_ms == 9.0
        assert key == "ori_time"

    def test_missing_returns_zero_none(self):
        assert _extract_time_from_dict({"foo": 1}) == (0.0, None)


def test_parse_test_cases_from_stdout():
    out = "Test case 0: 12.5 ms\nTest case 1: 8.0 ms"
    cases = parse_test_cases_from_stdout(out)
    assert [c.execution_time_ms for c in cases] == [12.5, 8.0]
    assert cases[0].test_case_id == "test_case_0"


def test_calculate_average_speedup_single_pair():
    base = [Case(test_case_id="t0", execution_time_ms=10.0)]
    opt = [Case(test_case_id="t0", execution_time_ms=5.0)]
    assert calculate_average_speedup(base, opt) == 2.0


def test_collect_benchmark_methods():
    cases = [
        Case(test_case_id="a", metadata={"benchmark_method": "cuda_event"}),
        Case(test_case_id="b", metadata={"benchmark_method": "cuda_graph"}),
        Case(test_case_id="c", metadata=None),
    ]
    assert collect_benchmark_methods(cases) == ["cuda_event", "cuda_graph"]


def test_save_load_roundtrip(tmp_path):
    cases = [
        Case(test_case_id="t0", shape=[256, 256], execution_time_ms=4.2, metadata={"params": {"n": 1}}),
    ]
    save_performance_results(cases, tmp_path, "perf.yaml")
    loaded = load_performance_results(tmp_path, "perf.yaml")
    assert len(loaded) == 1
    assert loaded[0].test_case_id == "t0"
    assert loaded[0].execution_time_ms == 4.2
    assert loaded[0].shape == [256, 256]
