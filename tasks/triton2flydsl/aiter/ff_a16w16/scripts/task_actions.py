"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'b1_h1_i1', 'batch': 1, 'hidden': 1, 'intermediate': 1}, {'name': 'b3_h5_i2', 'batch': 3, 'hidden': 5, 'intermediate': 2}, {'name': 'b32_h1024_i1024', 'batch': 32, 'hidden': 1024, 'intermediate': 1024}, {'name': 'b128_h2048_i8192', 'batch': 128, 'hidden': 2048, 'intermediate': 8192}, {'name': 'b64_h5120_i2880', 'batch': 64, 'hidden': 5120, 'intermediate': 2880}]
PERFORMANCE_IDS = {'perf1': 'case_0001', 'perf2': 'case_0005', 'perf3': 'case_0009', 'perf4': 'case_0013', 'perf5': 'case_0017'}
CORRECTNESS_COUNT = 20

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
        if isinstance(result, tuple):
            if result[0] is not True or len(result[2]) != CORRECTNESS_COUNT:
                raise RuntimeError(f"Incomplete/failed correctness evidence: {result}")
        elif result is not True:
            raise RuntimeError(f"Correctness/output-contract failure: {result}")
    return sorted(observed)


def performance(h):
    return h.run_benchmark()
