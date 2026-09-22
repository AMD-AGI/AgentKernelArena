"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'untuned_m64_n256_k5120', 'm': 64, 'n': 256, 'k': 5120}, {'name': 'untuned_m256_n256_k5120', 'm': 256, 'n': 256, 'k': 5120}, {'name': 'untuned_m512_n256_k5120', 'm': 512, 'n': 256, 'k': 5120}, {'name': 'dsv3_m128_n3072_k1536', 'm': 128, 'n': 3072, 'k': 1536}, {'name': 'dsv3_m64_n2112_k7168_tn64', 'm': 64, 'n': 2112, 'k': 7168, 'tile_n': 64}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
