"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [(1, 8, 16, 128, 128, 32), (2, 8, 16, 128, 128, 32), (4, 8, 16, 128, 128, 32), (8, 8, 16, 128, 128, 64), (16, 8, 16, 128, 128, 64), (32, 8, 16, 128, 128, 128), (64, 8, 16, 128, 128, 128), (128, 8, 16, 128, 128, 256), (1, 16, 32, 128, 128, 32), (32, 16, 32, 128, 128, 128)]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006', 'perf8': 'case_0007', 'perf9': 'case_0008', 'perf10': 'case_0009'}
CORRECTNESS_COUNT = 10

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
    return h.run_performance()
