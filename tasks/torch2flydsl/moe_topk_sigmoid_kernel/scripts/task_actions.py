"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003'}
EXPECTED_CASES = [{'name': 'llama4_t64_e128_k1', 'tokens': 64, 'experts': 128, 'topk': 1}, {'name': 'llama4_t1024_e128_k1', 'tokens': 1024, 'experts': 128, 'topk': 1}, {'name': 't256_e256_k8', 'tokens': 256, 'experts': 256, 'topk': 8}, {'name': 't64_e64_k2', 'tokens': 64, 'experts': 64, 'topk': 2}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
