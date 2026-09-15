"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002'}
EXPECTED_CASES = [{'name': 'dsv3_t16_e257_k9', 'tokens': 16, 'model_dim': 7168, 'inter_dim': 256, 'experts': 257, 'topk': 9}, {'name': 'dsv3_t32_e257_k9', 'tokens': 32, 'model_dim': 7168, 'inter_dim': 256, 'experts': 257, 'topk': 9}, {'name': 'glm5_t32_e257_k9', 'tokens': 32, 'model_dim': 6144, 'inter_dim': 256, 'experts': 257, 'topk': 9}]

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
