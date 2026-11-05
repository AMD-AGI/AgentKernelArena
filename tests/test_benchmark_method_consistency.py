import logging
from types import SimpleNamespace

import pytest

from src.evaluator import evaluate_kernel, write_task_result
from src.task_protocol import ActionResult, performance_cases
from src.testcases import (
    TestCaseResult as CaseResult,
    analyze_benchmark_method_consistency,
    analyze_workload_consistency,
    calculate_average_speedup,
    load_performance_results,
    match_test_cases,
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


def _performance(cases):
    """The validated action evidence a session returns for these cases."""
    rows = tuple({"test_case_id": c.test_case_id, "shape": c.shape, "status": "PASS",
                  "execution_time_ms": c.execution_time_ms, "metadata": dict(c.metadata)}
                 for c in cases)
    return ActionResult("candidate", "performance", "PASS", rows)


class _Session:
    """A v2 kernel session whose candidate passes and reports these timings."""

    spec = SimpleNamespace(contract=None)

    def __init__(self, cases):
        self.cases = cases

    def candidate_action(self, action):
        if action == "performance":
            return SimpleNamespace(result=_performance(self.cases))
        return SimpleNamespace(result=ActionResult("candidate", action, "PASS", ()))


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


@pytest.mark.parametrize("baseline_ms,optimized_ms,expected", [
    ((4.0, 1.0), (1.0, 2.0), 2.0 ** 0.5),   # 4x and 0.5x: geometric, not arithmetic (2.25)
    ((2.0, 1.0), (1.0, 2.0), 1.0),          # a 2x win cancels a 2x loss
    ((1000.0, 0.01), (500.0, 0.005), 2.0),  # a large shape does not outweigh a small one
])
def test_average_speedup_is_the_geometric_mean_of_per_case_ratios(baseline_ms, optimized_ms, expected):
    baseline = [_case(f"c{i}", t, "cuda_graph", [i]) for i, t in enumerate(baseline_ms)]
    optimized = [_case(f"c{i}", t, "cuda_graph", [i]) for i, t in enumerate(optimized_ms)]
    assert calculate_average_speedup(baseline, optimized) == pytest.approx(expected, rel=1e-12)


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

    matched = match_test_cases(baseline, optimized)

    assert [
        (base.test_case_id, opt.test_case_id) for base, opt in matched
    ] == [("a", "a"), ("b", "b")]
    assert calculate_average_speedup(baseline, optimized) == 1.0


def test_duplicate_ids_are_never_paired():
    baseline = [
        _case("a", 100.0, "cuda_graph", [1]),
        _case("a", 1.0, "cuda_graph", [1]),
    ]
    optimized = [
        _case("a", 1.0, "cuda_graph", [1]),
        _case("a", 100.0, "cuda_graph", [1]),
    ]

    assert match_test_cases(baseline, optimized) == []
    consistent, mismatches = analyze_benchmark_method_consistency(baseline, optimized)
    assert not consistent
    assert mismatches == []
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_single_cases_with_different_ids_are_never_paired():
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("b", 1.0, "cuda_graph", [1])]

    assert match_test_cases(baseline, optimized) == []
    assert analyze_benchmark_method_consistency(baseline, optimized) == (False, [])
    assert not analyze_workload_consistency(baseline, optimized)[0]
    assert calculate_average_speedup(baseline, optimized) == 0.0


@pytest.mark.parametrize('change', ['shape', 'params', 'dtype', 'missing_params'])
def test_same_case_id_cannot_hide_workload_changes(change):
    baseline = [_case('case-a', 20.0, 'cuda_graph', [64, 1024])]
    optimized = [_case('case-a', 1.0, 'cuda_graph', [64, 1024])]
    for case in baseline + optimized:
        case.metadata.update(params={'ctx_len': 1024}, dtype='bf16')
    if change == 'shape':
        optimized[0].shape = [64, 16]
    elif change == 'missing_params':
        del optimized[0].metadata['params']
    elif change == 'params':
        optimized[0].metadata['params'] = {'ctx_len': 16}
    else:
        optimized[0].metadata['dtype'] = 'fp8'

    consistent, mismatches = analyze_workload_consistency(baseline, optimized)

    assert not consistent
    assert mismatches[0]['fields'] == ['params' if change == 'missing_params' else change]
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_identical_workloads_allow_different_timing_diagnostics():
    baseline = [_case('a', 2.0, 'cuda_graph', [64, 1024])]
    optimized = [_case('a', 1.0, 'cuda_graph', [64, 1024])]
    optimized[0].metadata['benchmark_effective_repeats'] = 32
    optimized[0].metadata['tflops'] = 10

    assert analyze_workload_consistency(baseline, optimized) == (True, [])
    assert calculate_average_speedup(baseline, optimized) == 2.0


def test_action_workload_metadata_is_preserved_and_checked_after_reload(tmp_path):
    for name, ctx_len in [('baseline', 1024), ('optimized', 16)]:
        result = ActionResult(name, 'performance', 'PASS', ({
            'test_case_id': 'a', 'status': 'PASS', 'execution_time_ms': 1.0,
            'params': {'ctx_len': ctx_len}, 'dtype': 'bf16', 'benchmark_method': 'cuda_graph',
        },))
        save_performance_results(performance_cases(result), tmp_path, f'{name}.yaml')
    baseline = load_performance_results(tmp_path, 'baseline.yaml')
    optimized = load_performance_results(tmp_path, 'optimized.yaml')

    assert baseline[0].metadata['dtype'] == 'bf16'
    assert baseline[0].metadata['params'] == {'ctx_len': 1024}
    assert baseline[0].metadata['benchmark_method'] == 'cuda_graph'
    assert not analyze_workload_consistency(baseline, optimized)[0]
    assert calculate_average_speedup(baseline, optimized) == 0.0


def test_evaluator_reports_workload_mismatch_without_scoring_or_plotting(tmp_path):
    from src.plotting import plot_performance_comparison
    import yaml

    baseline = [_case('a', 20.0, 'cuda_graph', [1024])]
    optimized = [_case('a', 1.0, 'cuda_graph', [16])]
    result = evaluate_kernel(tmp_path, {}, baseline, task_session=_Session(optimized))

    assert result['pass_correctness']
    assert result['benchmark_method_consistent']
    assert not result['workload_consistent']
    assert result['best_optimized_execution_time'] == 1.0
    assert result['average_speedup'] == 0.0
    assert 'workloads differ' in result['speedup_calculation_error_message']
    write_task_result(tmp_path, result, baseline, 'task', 'any-agent', create_plots=False)
    written = yaml.safe_load((tmp_path / 'task_result.yaml').read_text())
    assert written['speedup_ratio'] == 0.0
    assert written['workload_mismatches'][0]['baseline_workload']['shape'] == [1024]
    # An explicitly failed workload gate also wins if a caller supplies a
    # positive ratio or omits the diagnostic message.
    result.update(average_speedup=20.0, speedup_calculation_error_message=None)
    write_task_result(tmp_path, result, baseline, 'task', 'any-agent', create_plots=False)
    assert yaml.safe_load((tmp_path / 'task_result.yaml').read_text())['speedup_ratio'] == 0.0
    save_performance_results(baseline, tmp_path, 'baseline_perf.yaml')
    assert plot_performance_comparison(tmp_path) is None


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


def test_evaluator_retains_times_but_disables_mismatched_speedup(tmp_path):
    baseline = [_case("a", 2.0, "cuda_graph", [1])]
    optimized = [_case("a", 1.0, "cuda_event_fallback", [1])]

    result = evaluate_kernel(tmp_path, {}, baseline, logging.getLogger(__name__),
                             task_session=_Session(optimized))

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
