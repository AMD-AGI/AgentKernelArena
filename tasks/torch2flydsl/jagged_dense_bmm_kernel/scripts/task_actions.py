"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'b4_varied', 'm_per_group': [100, 128, 64, 200]}, {'name': 'b3_aligned', 'm_per_group': [128, 256, 128]}, {'name': 'b5_with_empty', 'm_per_group': [0, 130, 0, 300, 50]}, {'name': 'b8_small', 'm_per_group': [10, 33, 128, 200, 1, 64, 129, 255]}, {'name': 'b2_single_tile', 'm_per_group': [64, 96]}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
