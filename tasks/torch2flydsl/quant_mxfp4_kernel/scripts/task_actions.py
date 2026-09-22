"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005', 'test_case_6': 'case_0006'}
EXPECTED_CASES = [{'name': 'm1_n32', 'm': 1, 'n': 32}, {'name': 'm3_n128', 'm': 3, 'n': 128}, {'name': 'm125_n64', 'm': 125, 'n': 64}, {'name': 'm4096_n128', 'm': 4096, 'n': 128}, {'name': 'm4096_n256', 'm': 4096, 'n': 256}, {'name': 'm4096_n1024', 'm': 4096, 'n': 1024}, {'name': 'm4097_n256', 'm': 4097, 'n': 256}]

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
