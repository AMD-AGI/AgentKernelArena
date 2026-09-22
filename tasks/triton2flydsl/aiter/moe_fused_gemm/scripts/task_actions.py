"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm64_k1024_n512_e8_top2', 'M': 64, 'K': 1024, 'N': 512, 'E': 8, 'top_k': 2}, {'name': 'm128_k2048_n1024_e8_top2', 'M': 128, 'K': 2048, 'N': 1024, 'E': 8, 'top_k': 2}, {'name': 'm256_k4096_n512_e16_top1', 'M': 256, 'K': 4096, 'N': 512, 'E': 16, 'top_k': 1}, {'name': 'm512_k1024_n768_e16_top2', 'M': 512, 'K': 1024, 'N': 768, 'E': 16, 'top_k': 2}, {'name': 'm1024_k2048_n256_e32_top4', 'M': 1024, 'K': 2048, 'N': 256, 'E': 32, 'top_k': 4}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0002', 'perf3': 'case_0004', 'perf4': 'case_0006', 'perf5': 'case_0008'}
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
        if h.run_correctness() is not True:
            raise RuntimeError("Correctness/output-contract failure")
    return sorted(observed)


def performance(h):
    return h.run_benchmark()
