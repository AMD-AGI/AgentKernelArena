import json
import logging
from unittest import mock

from src.evaluator import evaluate_kernel, write_task_result
from src.performance import measure_baseline
from src.testcases import (
    TestCaseResult as CaseResult,
    analyze_benchmark_method_consistency,
    calculate_average_speedup,
    load_performance_results,
    match_test_cases,
    parse_test_cases_from_json,
    parse_test_cases_from_stdout,
    save_performance_results,
)


def _case(case_id, time_ms, method, shape):
    return CaseResult(
        test_case_id=case_id,
        shape=shape,
        execution_time_ms=time_ms,
        metadata={
            "benchmark_method": method,
            "benchmark_samples": 100,
            "benchmark_effective_repeats": 8,
        },
    )


def test_mismatched_method_refuses_speedup_but_keeps_timings():
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 1.0, "cuda_event_fallback", [1])]

    consistent, mismatches = analyze_benchmark_method_consistency(baseline, optimized)

    assert not consistent
    assert mismatches == [{
        "test_case_id": "a",
        "optimized_test_case_id": "a",
        "baseline_benchmark_method": "cuda_graph",
        "optimized_benchmark_method": "cuda_event_fallback",
    }]
    assert calculate_average_speedup(baseline, optimized) == 0.0
    assert baseline[0].execution_time_ms == 2.0
    assert optimized[0].execution_time_ms == 1.0


def test_different_methods_across_shapes_are_allowed_when_each_pair_matches():
    baseline = [
        _case("graph", 2.0, "cuda_graph", [1]),
        _case("fallback", 6.0, "cuda_event_fallback", [2]),
    ]
    optimized = [
        _case("fallback", 3.0, "cuda_event_fallback", [2]),
        _case("graph", 1.0, "cuda_graph", [1]),
    ]

    consistent, mismatches = analyze_benchmark_method_consistency(baseline, optimized)

    assert consistent
    assert mismatches == []
    assert calculate_average_speedup(baseline, optimized) == 2.0


def test_explicit_ids_win_over_reordered_duplicate_semantic_keys():
    baseline = [
        _case("a", 100.0, "cuda_graph", [1]),
        _case("b", 1.0, "cuda_graph", [1]),
    ]
    optimized = [
        _case("b", 1.0, "cuda_graph", [1]),
        _case("a", 100.0, "cuda_graph", [1]),
    ]
    for case in baseline + optimized:
        case.metadata["params"] = {"shape": [1], "variant": "same"}

    matched = match_test_cases(
        baseline,
        optimized,
        allow_index_fallback=False,
    )

    assert [
        (base.test_case_id, opt.test_case_id) for base, opt in matched
    ] == [("a", "a"), ("b", "b")]
    assert calculate_average_speedup(baseline, optimized) == 1.0


def test_duplicate_synthetic_semantic_keys_do_not_greedily_match():
    baseline = [
        _case("test_case_0", 100.0, "cuda_graph", [1]),
        _case("test_case_1", 1.0, "cuda_graph", [1]),
    ]
    optimized = [
        _case("test_case_0", 1.0, "cuda_graph", [1]),
        _case("test_case_1", 100.0, "cuda_graph", [1]),
    ]
    for case in baseline + optimized:
        case.metadata["_synthetic_test_case_id"] = True
        case.metadata["params"] = {"shape": [1], "variant": "same"}

    assert match_test_cases(
        baseline,
        optimized,
        allow_index_fallback=False,
    ) == []
    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline,
        optimized,
    )
    assert not consistent
    assert mismatches == []
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_singleton_synthetic_cases_keep_index_fallback():
    baseline = [_case("test_case_0", 2.0, "cuda_graph", [1])]
    optimized = [_case("test_case_0", 1.0, "cuda_graph", [2])]
    baseline[0].metadata["_synthetic_test_case_id"] = True
    optimized[0].metadata["_synthetic_test_case_id"] = True

    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline,
        optimized,
    )

    assert consistent
    assert mismatches == []
    assert calculate_average_speedup(baseline, optimized) == 2.0


def test_candidate_event_fallback_cannot_switch_graph_baseline():
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 3.0, "cuda_event_fallback", [1])]
    # Legacy alternate metadata must not affect the immutable baseline policy.
    baseline[0].metadata["benchmark_alternate_event_time_ms"] = 6.0
    baseline[0].metadata["benchmark_alternate_event_method"] = (
        "cuda_event_fallback"
    )

    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline, optimized
    )

    assert not consistent
    assert len(mismatches) == 1
    assert baseline[0].execution_time_ms == 2.0
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_synthetic_id_provenance_survives_yaml_round_trip(tmp_path):
    cases = [_case("test_case_0", 1.0, "cuda_graph", [1])]
    cases[0].metadata["_synthetic_test_case_id"] = True

    save_performance_results(cases, tmp_path, "perf.yaml")
    loaded = load_performance_results(tmp_path, "perf.yaml")

    assert loaded[0].metadata["_synthetic_test_case_id"] is True


def test_baseline_is_measured_once_without_candidate_selected_event_variant(tmp_path):
    baseline = [
        _case(
            "aggregate",
            2.0,
            "mixed:cuda_event_fallback,cuda_graph",
            [1],
        )
    ]
    with mock.patch(
        "src.performance.measure_performance",
        return_value=baseline,
    ) as measured:
        measured_baseline = measure_baseline(tmp_path, {}, logger=mock.Mock())

    assert measured.call_count == 1
    assert measured_baseline == baseline


def test_mixed_aggregate_method_string_must_match_exactly():
    baseline = [_case("a", 4.0, "mixed:cuda_graph,cuda_event_fallback", [1])]
    optimized = [_case("a", 2.0, "mixed:cuda_event_fallback,cuda_graph", [1])]

    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_identical_mixed_aggregate_is_still_ambiguous():
    baseline = [_case("a", 4.0, "mixed:cuda_event_fallback,cuda_graph", [1])]
    optimized = [_case("a", 2.0, "mixed:cuda_event_fallback,cuda_graph", [1])]

    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline, optimized
    )

    assert not consistent
    assert mismatches[0]["reason"] == "ambiguous_mixed_aggregate"
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_missing_method_metadata_is_never_comparable():
    baseline = [_case("a", 4.0, None, [1])]
    optimized = [_case("a", 2.0, None, [1])]

    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline, optimized
    )

    assert not consistent
    assert mismatches[0]["reason"] == "missing_or_unknown_benchmark_method"
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_nested_benchmark_metadata_is_promoted_for_scoring(tmp_path):
    report = tmp_path / "performance_report.json"
    row = {
        "test_case_id": "a",
        "shape": [1],
        "execution_time_ms": 2.0,
        "metadata": {
            "model": "representative-image-kernel",
            "benchmark_method": "cuda_graph",
            "benchmark_samples": 100,
        },
    }
    report.write_text(json.dumps([row]))
    baseline = parse_test_cases_from_json(report)

    row["execution_time_ms"] = 1.0
    report.write_text(json.dumps([row]))
    optimized = parse_test_cases_from_json(report)

    assert baseline[0].metadata["metadata"]["model"] == (
        "representative-image-kernel"
    )
    assert baseline[0].metadata["benchmark_method"] == "cuda_graph"
    assert baseline[0].metadata["benchmark_samples"] == 100
    assert calculate_average_speedup(baseline, optimized) == 2.0


def test_flat_benchmark_metadata_wins_over_nested_compatibility_value(tmp_path):
    report = tmp_path / "performance_report.json"
    report.write_text(json.dumps([{
        "test_case_id": "a",
        "execution_time_ms": 1.0,
        "benchmark_method": "cuda_event_fallback",
        "metadata": {"benchmark_method": "cuda_graph"},
    }]))

    cases = parse_test_cases_from_json(report)

    assert cases[0].metadata["benchmark_method"] == "cuda_event_fallback"


def test_stdout_geak_metadata_is_retained():
    cases = parse_test_cases_from_stdout(
        "GEAK_RESULT_LATENCY_MS=0.125\n"
        "GEAK_BENCHMARK_METHOD=mixed:cuda_graph,cuda_event_fallback\n"
        "GEAK_BENCHMARK_METHOD_CONSISTENT=0\n"
        "GEAK_BENCHMARK_FALLBACK_REASON=shape 2 reads a host scalar\n"
    )

    assert len(cases) == 1
    assert cases[0].execution_time_ms == 0.125
    assert cases[0].metadata["benchmark_method"] == (
        "mixed:cuda_graph,cuda_event_fallback"
    )
    assert cases[0].metadata["benchmark_fallback_reason"] == (
        "shape 2 reads a host scalar"
    )
    assert cases[0].metadata["benchmark_method_consistent"] is False


def test_task_reported_method_mismatch_is_unscoreable_even_if_outer_methods_match():
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 1.0, "cuda_graph", [1])]
    optimized[0].metadata["benchmark_method_consistent"] = False

    consistent, mismatches = analyze_benchmark_method_consistency(
        baseline, optimized
    )

    assert not consistent
    assert mismatches[0]["reason"] == "task_reported_method_mismatch"
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_all_benchmark_metadata_survives_yaml_round_trip(tmp_path):
    cases = [_case("a", 1.25, "cuda_graph", [4])]
    cases[0].metadata["benchmark_target_ms"] = 1.0
    cases[0].metadata["benchmark_fallback_reason"] = "not applicable"

    save_performance_results(cases, tmp_path, "perf.yaml", logging.getLogger(__name__))
    loaded = load_performance_results(tmp_path, "perf.yaml", logging.getLogger(__name__))

    assert len(loaded) == 1
    assert loaded[0].metadata["benchmark_method"] == "cuda_graph"
    assert loaded[0].metadata["benchmark_samples"] == 100
    assert loaded[0].metadata["benchmark_effective_repeats"] == 8
    assert loaded[0].metadata["benchmark_target_ms"] == 1.0
    assert loaded[0].metadata["benchmark_fallback_reason"] == "not applicable"


@mock.patch("src.evaluator.evaluate_compilation", return_value=(True, None))
@mock.patch("src.evaluator.evaluate_correctness", return_value=(True, None))
def test_evaluator_retains_times_but_disables_mismatched_speedup(
    _correctness, _compilation, tmp_path
):
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 1.0, "cuda_event_fallback", [1])]

    with mock.patch("src.evaluator.measure_performance", return_value=optimized):
        result = evaluate_kernel(
            tmp_path,
            {
                "task_type": "triton2triton",
                "compile_command": ["compile"],
                "correctness_command": ["correctness"],
                "performance_command": ["performance"],
            },
            baseline,
            logger=mock.Mock(),
        )

    assert result["best_optimized_execution_time"] == 1.0
    assert result["average_speedup"] == 0.0
    assert not result["benchmark_method_consistent"]
    assert result["benchmark_method_mismatches"]
    assert "different benchmark methods" in result["speedup_calculation_error_message"]


def test_task_result_never_reconstructs_speedup_without_method_consistency(tmp_path):
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    evaluation = {
        "pass_compilation": True,
        "pass_correctness": True,
        "best_optimized_execution_time": 1.0,
        "average_speedup": 0.0,
        "benchmark_method_consistent": False,
        "optimized_benchmark_methods": ["cuda_event_fallback"],
        "benchmark_method_mismatches": [{
            "test_case_id": "a",
            "optimized_test_case_id": "a",
            "baseline_benchmark_method": "cuda_graph",
            "optimized_benchmark_method": "cuda_event_fallback",
        }],
    }

    write_task_result(
        tmp_path,
        evaluation,
        baseline,
        task_name="task",
        agent_name="agent",
        create_plots=False,
    )

    import yaml

    result = yaml.safe_load((tmp_path / "task_result.yaml").read_text())
    assert result["speedup_ratio"] == 0.0


def test_plotting_refuses_method_mismatch_and_invalid_timings(tmp_path):
    from src.plotting import plot_performance_comparison

    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 1.0, "cuda_event_fallback", [1])]
    save_performance_results(baseline, tmp_path, "baseline_perf.yaml")
    save_performance_results(optimized, tmp_path, "optimized_perf.yaml")

    assert plot_performance_comparison(tmp_path) is None
    assert not (tmp_path / "performance_execution_time.png").exists()

    optimized[0] = _case("a", 0.0, "cuda_graph", [1])
    save_performance_results(optimized, tmp_path, "optimized_perf.yaml")
    assert plot_performance_comparison(tmp_path) is None
    assert not (tmp_path / "performance_speedup.png").exists()
