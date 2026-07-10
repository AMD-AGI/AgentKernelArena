# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for src/score.py scoring logic (CPU-only, no GPU deps)."""
import pytest
import yaml

from src.score import resolve_speedup_ratio, score, task_result_scoring


class TestResolveSpeedupRatio:
    def test_explicit_positive_ratio_wins(self):
        assert resolve_speedup_ratio(
            speedup_ratio=1.5, base_execution_time=10.0, best_optimized_execution_time=1.0
        ) == 1.5

    def test_falls_back_to_time_ratio(self):
        assert resolve_speedup_ratio(
            speedup_ratio=0.0, base_execution_time=10.0, best_optimized_execution_time=5.0
        ) == 2.0

    def test_returns_zero_without_valid_inputs(self):
        assert resolve_speedup_ratio(speedup_ratio=None) == 0.0
        assert resolve_speedup_ratio(
            speedup_ratio=-1.0, base_execution_time=0.0, best_optimized_execution_time=0.0
        ) == 0.0


class TestScore:
    def test_compilation_failure_scores_zero(self):
        assert score(False, True, 10.0, 5.0, speedup_ratio=2.0) == 0.0

    def test_compilation_pass_correctness_fail_scores_twenty(self):
        assert score(True, False, 10.0, 5.0, speedup_ratio=2.0) == 20.0

    def test_both_pass_adds_base_plus_speedup(self):
        # 20 (compile) + 100 (correct) + 2.0 * 100 (speedup) = 320
        assert score(True, True, 10.0, 5.0, speedup_ratio=2.0) == 320.0

    def test_both_pass_without_speedup_scores_120(self):
        assert score(True, True, 0.0, 0.0, speedup_ratio=0.0) == 120.0

    def test_speedup_derived_from_times_when_ratio_missing(self):
        # speedup = 10/5 = 2 -> 120 + 200 = 320
        assert score(True, True, 10.0, 5.0, speedup_ratio=0.0) == 320.0


def test_task_result_scoring_roundtrip(tmp_path):
    result = {
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": 10.0,
        "best_optimized_execution_time": 5.0,
        "speedup_ratio": 2.0,
    }
    result_file = tmp_path / "task_result.yaml"
    result_file.write_text(yaml.safe_dump(result), encoding="utf-8")

    computed = task_result_scoring(str(tmp_path))
    assert computed == 320.0

    # The score is written back into the file.
    written = yaml.safe_load(result_file.read_text(encoding="utf-8"))
    assert written["score"] == 320.0


def test_task_result_scoring_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        task_result_scoring(str(tmp_path))
