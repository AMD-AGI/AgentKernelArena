"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005'}
EXPECTED_CASES = [{'name': 'm1_n4096', 'm': 1, 'n': 4096}, {'name': 'm8_n4096', 'm': 8, 'n': 4096}, {'name': 'm32_n8192', 'm': 32, 'n': 8192}, {'name': 'm128_n4096', 'm': 128, 'n': 4096}, {'name': 'm256_n8192', 'm': 256, 'n': 8192}, {'name': 'm64_n16384', 'm': 64, 'n': 16384}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
