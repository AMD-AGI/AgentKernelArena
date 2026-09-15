"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005'}
EXPECTED_CASES = [{'name': 'T1_H16', 'T': 1, 'H': 16, 'D': 512, 'RD': 64, 'group_size': 64}, {'name': 'T16_H16', 'T': 16, 'H': 16, 'D': 512, 'RD': 64, 'group_size': 64}, {'name': 'T64_H16', 'T': 64, 'H': 16, 'D': 512, 'RD': 64, 'group_size': 32}, {'name': 'T256_H16', 'T': 256, 'H': 16, 'D': 512, 'RD': 64, 'group_size': 128}, {'name': 'T64_H128', 'T': 64, 'H': 128, 'D': 512, 'RD': 64, 'group_size': 64}, {'name': 'T512_H128', 'T': 512, 'H': 128, 'D': 512, 'RD': 64, 'group_size': 64}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
