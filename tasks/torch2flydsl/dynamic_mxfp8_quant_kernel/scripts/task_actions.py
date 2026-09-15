"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005', 'test_case_6': 'case_0006', 'test_case_7': 'case_0007'}
EXPECTED_CASES = [{'name': 'm1_k32', 'm': 1, 'n': 32}, {'name': 'm8_k64', 'm': 8, 'n': 64}, {'name': 'm16_k128', 'm': 16, 'n': 128}, {'name': 'm32_k256', 'm': 32, 'n': 256}, {'name': 'm64_k512', 'm': 64, 'n': 512}, {'name': 'm128_k1024', 'm': 128, 'n': 1024}, {'name': 'm137_k64', 'm': 137, 'n': 64}, {'name': 'm256_k32', 'm': 256, 'n': 32}]

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
