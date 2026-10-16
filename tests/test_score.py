import math

import pytest
import yaml

from src.score import resolve_speedup_ratio, score, task_result_scoring


def test_explicit_zero_speedup_is_not_reconstructed_from_times():
    assert resolve_speedup_ratio(
        speedup_ratio=0.0,
        base_execution_time=8.0,
        best_optimized_execution_time=2.0,
        benchmark_method_consistent=True,
    ) == 0.0


def test_method_mismatch_disables_even_stale_positive_speedup():
    assert resolve_speedup_ratio(
        speedup_ratio=4.0,
        base_execution_time=8.0,
        best_optimized_execution_time=2.0,
        benchmark_method_consistent=False,
    ) == 0.0
    assert score(
        True,
        True,
        8.0,
        2.0,
        speedup_ratio=4.0,
        benchmark_method_consistent=False,
    ) == 0.0


def test_task_result_mismatch_cannot_regain_performance_points(tmp_path):
    result_file = tmp_path / "task_result.yaml"
    result_file.write_text(yaml.safe_dump({
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": 8.0,
        "best_optimized_execution_time": 2.0,
        "speedup_ratio": 0.0,
        "benchmark_method_consistent": False,
    }))

    assert task_result_scoring(str(tmp_path)) == 0.0
    assert yaml.safe_load(result_file.read_text())["score"] == 0.0


def test_legacy_result_without_method_metadata_gets_no_performance_points(tmp_path):
    result_file = tmp_path / "task_result.yaml"
    result_file.write_text(yaml.safe_dump({
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": 8.0,
        "best_optimized_execution_time": 2.0,
    }))

    assert task_result_scoring(str(tmp_path)) == 0.0


def test_explicit_speedup_without_method_metadata_gets_no_performance_points():
    assert resolve_speedup_ratio(
        speedup_ratio=4.0,
        base_execution_time=8.0,
        best_optimized_execution_time=2.0,
    ) == 0.0


@pytest.mark.parametrize("speedup,expected", [
    (0.5, 0.0), (1.0, 0.0), (1.049, 0.0), (1.05, 0.0),
    (1.1, 0.21320071635561044), (1.2, 0.3535533905932738),
    (1.25, 0.4), (2.0, 0.689202437604511),
])
def test_default_square_root_score(speedup, expected):
    assert score(True, True, 100.0, 100.0 / speedup, speedup, True) == pytest.approx(expected)


@pytest.mark.parametrize("compiled,correct", [(False, False), (False, True), (True, False)])
def test_failed_checks_never_receive_partial_credit(compiled, correct):
    assert score(compiled, correct, 100.0, 1.0, 100.0, True) == 0.0


@pytest.mark.parametrize("speedup", [None, 0.0, -1.0, math.nan, math.inf, -math.inf, "2.0"])
def test_invalid_speedup_never_uses_aggregate_times(speedup):
    assert score(True, True, 100.0, 1.0, speedup, True) == 0.0


@pytest.mark.parametrize("consistent", [None, False, "true", 1])
def test_score_requires_explicit_method_consistency(consistent):
    assert score(True, True, 100.0, 50.0, 2.0, consistent) == 0.0


@pytest.mark.parametrize("power,expected", [(0.5, 0.4), (1.0, 0.16), (2.0, 0.0256)])
def test_power_constant_selects_curve(monkeypatch, power, expected):
    monkeypatch.setattr("src.score.SCORE_POWER", power)
    assert score(True, True, 100.0, 80.0, 1.25, True) == pytest.approx(expected)


def test_speedup_threshold_constant_changes_cutoff(monkeypatch):
    monkeypatch.setattr("src.score.SCORE_SPEEDUP_THRESHOLD", 1.2)
    assert score(True, True, 100.0, 90.0, 1.1, True) == 0.0
    assert score(True, True, 100.0, 100.0 / 1.2, 1.2, True) == 0.0
    assert score(True, True, 100.0, 80.0, 1.25, True) == pytest.approx(0.2)


@pytest.mark.parametrize("name", ["SCORE_POWER", "SCORE_SPEEDUP_THRESHOLD"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_scoring_constants_are_rejected(monkeypatch, name, value):
    monkeypatch.setattr(f"src.score.{name}", value)
    with pytest.raises(ValueError, match=name):
        score(True, True, 100.0, 50.0, 2.0, True)


def test_score_is_bounded_and_monotonic_above_threshold():
    speedups = [1.05, math.nextafter(1.05, math.inf), 1.1, 1.25, 2.0, 10.0, 1e300]
    scores = [score(True, True, 100.0, 100.0 / s, s, True) for s in speedups]
    assert all(0.0 <= value <= 1.0 for value in scores)
    assert scores == sorted(scores)
    assert scores[1] > 0.0
    assert scores[-1] == pytest.approx(1.0)


def test_task_result_scores_explicit_mean_speedup_and_preserves_measurements(tmp_path):
    # Cases 1 -> 1 and 100 -> 50 have mean speedup 1.5, not 101/51.
    result = {
        "pass_compilation": True, "pass_correctness": True,
        "base_execution_time": 50.5, "best_optimized_execution_time": 25.5,
        "speedup_ratio": 1.5, "benchmark_method_consistent": True,
        "score": 270.0,
    }
    result_file = tmp_path / "task_result.yaml"
    result_file.write_text(yaml.safe_dump(result))

    calculated = task_result_scoring(str(tmp_path))

    assert calculated == pytest.approx(0.5477225575051661)
    assert yaml.safe_load(result_file.read_text()) == {**result, "score": calculated}
