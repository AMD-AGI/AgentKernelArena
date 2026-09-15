"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm256_n512_k1024', 'M': 256, 'N': 512, 'K': 1024}, {'name': 'm1024_n1024_k1024', 'M': 1024, 'N': 1024, 'K': 1024}, {'name': 'm2048_n2048_k2048', 'M': 2048, 'N': 2048, 'K': 2048}, {'name': 'm64_n256_k7168', 'M': 64, 'N': 256, 'K': 7168}, {'name': 'm128_n2048_k4096', 'M': 128, 'N': 2048, 'K': 4096}]
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
        if h.run_correctness() is not True:
            raise RuntimeError("Correctness/output-contract failure")
    return sorted(observed)


def performance(h):
    return h.run_benchmark()
