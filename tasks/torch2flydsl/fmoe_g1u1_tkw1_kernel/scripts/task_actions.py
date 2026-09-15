"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002'}
EXPECTED_CASES = [{'name': 'e16_t128_d5120_i1024_k1', 'tokens': 128, 'model_dim': 5120, 'inter_dim': 1024, 'experts': 16, 'topk': 1}, {'name': 'e16_t128_d5120_i1024_k2', 'tokens': 128, 'model_dim': 5120, 'inter_dim': 1024, 'experts': 16, 'topk': 2}, {'name': 'e128_t128_d5120_i1024_k2', 'tokens': 128, 'model_dim': 5120, 'inter_dim': 1024, 'experts': 128, 'topk': 2}]

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
