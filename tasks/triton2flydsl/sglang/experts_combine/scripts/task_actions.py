"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'tokens': 128, 'combine_k': 1, 'hidden': 4096, 'dtype': 'bf16'}, {'tokens': 128, 'combine_k': 2, 'hidden': 4096, 'dtype': 'bf16'}, {'tokens': 64, 'combine_k': 4, 'hidden': 2048, 'dtype': 'bf16'}, {'tokens': 256, 'combine_k': 8, 'hidden': 1024, 'dtype': 'bf16'}, {'tokens': 1, 'combine_k': 2, 'hidden': 7168, 'dtype': 'bf16'}, {'tokens': 32, 'combine_k': 2, 'hidden': 3072, 'dtype': 'bf16'}, {'tokens': 16, 'combine_k': 2, 'hidden': 4096, 'dtype': 'fp16'}, {'tokens': 8, 'combine_k': 2, 'hidden': 2048, 'dtype': 'fp32'}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006', 'perf8': 'case_0007'}
CORRECTNESS_COUNT = 8

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
