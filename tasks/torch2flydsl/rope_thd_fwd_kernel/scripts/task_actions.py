"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003'}
EXPECTED_CASES = [{'name': 'seqs5_h8_d128', 'cu': [0, 100, 228, 484, 712, 1024], 'h': 8, 'd': 128}, {'name': 'seqs4_h16_d128', 'cu': [0, 233, 456, 711, 1024], 'h': 16, 'd': 128}, {'name': 'seqs8_h8_d64', 'cu': [0, 100, 102, 128, 233, 456, 460, 711, 1024], 'h': 8, 'd': 64}, {'name': 'seqs3_h32_d128', 'cu': [0, 512, 1024, 2048], 'h': 32, 'd': 128}]

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
