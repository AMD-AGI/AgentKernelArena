"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [(1, 128, 4, 4, 128, True), (1, 256, 8, 8, 128, True), (2, 128, 8, 2, 128, True), (1, 256, 8, 8, 128, False), (1, 384, 4, 4, 128, True)]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004'}
CORRECTNESS_COUNT = 5

def select_role(h, role, provided):
    import os
    h.ARENA_FINAL_CANDIDATE = role == "candidate" and os.environ.get("ARENA_EVAL_PHASE") != "task_validation"
    if provided:
        raise ValueError("This task uses a frozen initial candidate baseline")
    if h.TEST_SHAPES != EXPECTED_SHAPES:
        raise ValueError("Protected case manifest disagrees with harness")

def check(h):
    from scripts.candidate_checks import audit_candidate_calls
    with audit_candidate_calls(h) as observed:
        result = h.run_correctness()
        if (not isinstance(result, tuple) or len(result) != 3
                or result[0] is not True or not isinstance(result[2], list)
                or len(result[2]) != CORRECTNESS_COUNT):
            raise RuntimeError(f"Incomplete/failed correctness evidence: {result}")
        for i, record in enumerate(result[2], 1):
            if record.get("shape_id") != i or record.get("passed") is not True or "error" in record:
                raise RuntimeError(f"Invalid or failed correctness case: {record}")
    return sorted(observed)


def performance(h):
    import json
    from pathlib import Path
    records = h.run_performance()
    # Preserve raw successful and failed cases before strict protocol parsing.
    build = Path(h.__file__).resolve().parent / "build"
    build.mkdir(exist_ok=True)
    (build / "raw_performance_cases.json").write_text(json.dumps(records, indent=2) + "\n")
    failures = [f"{row.get('test_case_id')}: {row.get('benchmark_fallback_reason', 'missing/invalid timing')}"
                for row in records if row.get("execution_time_ms", -1) <= 0]
    if failures:
        raise RuntimeError("Performance case failures: " + "; ".join(failures))
    return records
