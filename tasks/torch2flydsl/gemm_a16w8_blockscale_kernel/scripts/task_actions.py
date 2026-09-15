"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'sq_m1024_n1024_k1024', 'm': 1024, 'n': 1024, 'k': 1024}, {'name': 'sq_m2048_n2048_k2048', 'm': 2048, 'n': 2048, 'k': 2048}, {'name': 'dsr1_router_m32_n256_k7168', 'm': 32, 'n': 256, 'k': 7168}, {'name': 'dsr1_router_m128_n256_k7168', 'm': 128, 'n': 256, 'k': 7168}, {'name': 'gptoss_oproj_m256_n2048_k2048', 'm': 256, 'n': 2048, 'k': 2048}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    from scripts.candidate_checks import audit_candidate_calls
    with audit_candidate_calls(h) as observed:
        if h.run_correctness(verbose=True) is not True:
            raise RuntimeError("Correctness/output-contract check failed")
    return sorted(observed)

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
