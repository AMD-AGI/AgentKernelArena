"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003'}
EXPECTED_CASES = [{'name': 'm1_n4096', 'm': 1, 'n': 4096}, {'name': 'm128_n8192', 'm': 128, 'n': 8192}, {'name': 'm1024_n6400', 'm': 1024, 'n': 6400}, {'name': 'm4096_n4096', 'm': 4096, 'n': 4096}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    from scripts.candidate_checks import audit_candidate_calls
    from scripts.limit_controls import check_positive_limits
    with audit_candidate_calls(h) as observed:
        if h.run_correctness(verbose=True) is not True:
            raise RuntimeError("Correctness/output-contract check failed")
        check_positive_limits(h)
    return sorted(observed)

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
