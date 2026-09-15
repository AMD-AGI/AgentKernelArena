"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'm128_n1536_k4096', 'm': 128, 'n': 1536, 'k': 4096}, {'name': 'm64_n4096_k1024', 'm': 64, 'n': 4096, 'k': 1024}, {'name': 'm128_n2048_k7168', 'm': 128, 'n': 2048, 'k': 7168}, {'name': 'm32_n8192_k1536', 'm': 32, 'n': 8192, 'k': 1536}, {'name': 'm256_n768_k7168', 'm': 256, 'n': 768, 'k': 7168}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
