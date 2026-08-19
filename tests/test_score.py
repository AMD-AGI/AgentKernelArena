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
    ) == 120.0


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

    assert task_result_scoring(str(tmp_path)) == 120.0
    assert yaml.safe_load(result_file.read_text())["score"] == 120.0


def test_legacy_result_without_method_metadata_gets_no_performance_points(tmp_path):
    result_file = tmp_path / "task_result.yaml"
    result_file.write_text(yaml.safe_dump({
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": 8.0,
        "best_optimized_execution_time": 2.0,
    }))

    assert task_result_scoring(str(tmp_path)) == 120.0


def test_explicit_speedup_without_method_metadata_gets_no_performance_points():
    assert resolve_speedup_ratio(
        speedup_ratio=4.0,
        base_execution_time=8.0,
        best_optimized_execution_time=2.0,
    ) == 0.0
