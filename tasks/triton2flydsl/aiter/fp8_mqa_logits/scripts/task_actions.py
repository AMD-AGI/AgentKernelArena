"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [(64, 512, 32, 128, 'full'), (128, 1000, 64, 128, 'full'), (256, 2048, 32, 64, 'causal'), (96, 777, 16, 128, 'band'), (2048, 1024, 64, 128, 'full')]
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
        if not isinstance(result, dict) or result.get("correct") is not True:
            raise RuntimeError(f"Correctness/output-contract failure: {result}")
        details = result.get("details")
        if not isinstance(details, list) or len(details) != CORRECTNESS_COUNT:
            raise RuntimeError("Incomplete MQA correctness evidence")
        for i, (record, shape) in enumerate(zip(details, EXPECTED_SHAPES), 1):
            if record.get("shape_id") != i or record.get("shape") != list(shape) or record.get("passed") is not True or "error" in record:
                raise RuntimeError(f"Invalid or failed MQA correctness case: {record}")
    return sorted(observed)


def performance(h):
    result = h.run_benchmark()
    if not isinstance(result, dict) or not isinstance(result.get("cases"), list):
        raise RuntimeError("MQA benchmark did not return per-case timing evidence")
    return result["cases"]
