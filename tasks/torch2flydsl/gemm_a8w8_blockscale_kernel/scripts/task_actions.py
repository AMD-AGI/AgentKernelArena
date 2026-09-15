"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'dsv3_m16_n256_k7168', 'm': 16, 'n': 256, 'k': 7168}, {'name': 'dsv3_m128_n512_k7168', 'm': 128, 'n': 512, 'k': 7168}, {'name': 'dsv3_m256_n3072_k1536', 'm': 256, 'n': 3072, 'k': 1536}, {'name': 'dsv3_m512_n4096_k512', 'm': 512, 'n': 4096, 'k': 512}, {'name': 'dsv3_m128_n7168_k2048', 'm': 128, 'n': 7168, 'k': 2048}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
