"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'minimal_m1_n1_k1', 'm': 1, 'n': 1, 'k': 1}, {'name': 'irregular_m3_n5_k2', 'm': 3, 'n': 5, 'k': 2}, {'name': 'sq_m1024_n1024_k1024', 'm': 1024, 'n': 1024, 'k': 1024}, {'name': 'gptoss_qkv_m128_n5120_k2880', 'm': 128, 'n': 5120, 'k': 2880}, {'name': 'gptoss_oproj_m128_n2880_k4096', 'm': 128, 'n': 2880, 'k': 4096}]

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
