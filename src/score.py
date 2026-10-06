# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import math
import yaml
from pathlib import Path


# Power-family scoring; see docs/reference/api-reference.md#scoring.
SCORE_POWER = 0.5
SCORE_SPEEDUP_THRESHOLD = 1.05


def resolve_speedup_ratio(
    speedup_ratio: float | int | None = None,
    base_execution_time: float = 0.0,
    best_optimized_execution_time: float = 0.0,
    benchmark_method_consistent: bool | None = None,
) -> float:
    """
    Resolve the speedup ratio to use for scoring/reporting.

    Prefer an explicit speedup_ratio written by the evaluator. This preserves the
    intended aggregation logic for multi-testcase tasks where each testcase should
    contribute equally. An explicit zero is authoritative: current evaluators use
    it when a comparison is invalid, so reconstructing from average times would
    bypass their fairness checks.

    Performance points require an explicit
    ``benchmark_method_consistent=True``. Missing or false method-consistency
    metadata fails closed, including for legacy result files: aggregate times do
    not prove that baseline and optimized kernels used comparable timing methods.
    """
    if benchmark_method_consistent is not True:
        return 0.0

    if speedup_ratio is not None:
        if (
            isinstance(speedup_ratio, (int, float))
            and math.isfinite(float(speedup_ratio))
            and speedup_ratio > 0
        ):
            return float(speedup_ratio)
        return 0.0

    return 0.0


def score(
    pass_compilation: bool,
    pass_correctness: bool,
    base_execution_time: float,
    best_optimized_execution_time: float,
    speedup_ratio: float | int | None = None,
    benchmark_method_consistent: bool | None = None,
) -> float:
    """
    Calculate a bounded power-family score for a correct, comparable candidate.

    The score is ``max(0, 1 - SCORE_SPEEDUP_THRESHOLD / speedup) ** SCORE_POWER``.
    Defaults use the concave square-root curve and require more than 1.05x
    speedup for a positive score. Compilation/correctness failures and invalid
    performance comparisons receive zero, with no partial-credit bonuses.

    Args:
        pass_compilation: Whether compilation succeeded
        pass_correctness: Whether correctness tests passed
        base_execution_time: Retained for caller compatibility; not used to
            reconstruct missing or invalid speedup ratios.
        best_optimized_execution_time: Retained for caller compatibility.
        speedup_ratio: Explicit speedup ratio from evaluator output. Preferred for
            multi-testcase tasks where each testcase should have equal weight.
        benchmark_method_consistent: Whether matched baseline/optimized cases used
            comparable timing methods. Only explicit ``True`` enables performance
            points.

    Returns:
        float: Score in [0, 1]. Correctness and acceptance remain separate fields;
            a correct candidate at or below the speedup threshold also scores 0.

    Raises:
        ValueError: A scoring constant is not finite and positive.
    """
    for name, value in (
        ("SCORE_POWER", SCORE_POWER),
        ("SCORE_SPEEDUP_THRESHOLD", SCORE_SPEEDUP_THRESHOLD),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    if not pass_compilation or not pass_correctness:
        return 0.0

    effective_speedup = resolve_speedup_ratio(
        speedup_ratio=speedup_ratio,
        base_execution_time=base_execution_time,
        best_optimized_execution_time=best_optimized_execution_time,
        benchmark_method_consistent=benchmark_method_consistent,
    )
    if effective_speedup <= SCORE_SPEEDUP_THRESHOLD:
        return 0.0

    return (1.0 - SCORE_SPEEDUP_THRESHOLD / effective_speedup) ** SCORE_POWER


def task_result_scoring(workspace_path: str) -> float:
    """
    Read task_result.yaml from workspace, calculate score, and append it to the file.

    Args:
        workspace_path: Path to the workspace directory containing task_result.yaml

    Returns:
        float: Calculated score

    Raises:
        FileNotFoundError: If task_result.yaml doesn't exist
        KeyError: If required fields are missing from the YAML
    """
    workspace = Path(workspace_path)
    result_file = workspace / "task_result.yaml"

    # Check if file exists
    if not result_file.exists():
        raise FileNotFoundError(f"task_result.yaml not found in {workspace_path}")

    # Read the YAML file
    with open(result_file, 'r') as f:
        result_data = yaml.safe_load(f)

    # Extract required fields
    pass_compilation = result_data.get('pass_compilation', False)
    pass_correctness = result_data.get('pass_correctness', False)
    base_execution_time = result_data.get('base_execution_time', 0.0)
    best_optimized_execution_time = result_data.get('best_optimized_execution_time', 0.0)
    # Missing speedup or method consistency fails closed. Aggregate times alone
    # cannot establish that baseline and optimized measurements are comparable.
    speedup_ratio = result_data.get('speedup_ratio')
    benchmark_method_consistent = result_data.get(
        'benchmark_method_consistent'
    )

    # Calculate score
    calculated_score = score(
        pass_compilation=pass_compilation,
        pass_correctness=pass_correctness,
        base_execution_time=base_execution_time,
        best_optimized_execution_time=best_optimized_execution_time,
        speedup_ratio=speedup_ratio,
        benchmark_method_consistent=benchmark_method_consistent,
    )

    # Add score to the data
    result_data['score'] = calculated_score

    # Write back to the YAML file
    with open(result_file, 'w') as f:
        yaml.dump(result_data, f, default_flow_style=False, sort_keys=False)

    return calculated_score
