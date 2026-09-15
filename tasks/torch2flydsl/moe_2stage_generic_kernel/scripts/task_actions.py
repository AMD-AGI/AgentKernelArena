"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001'}
EXPECTED_CASES = [{'name': 'e32_t32_d4096_i1024_k5', 'tokens': 32, 'model_dim': 4096, 'inter_dim': 1024, 'experts': 32, 'topk': 5}, {'name': 'dsv3_t32_e257_k9', 'tokens': 32, 'model_dim': 7168, 'inter_dim': 256, 'experts': 257, 'topk': 9}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
