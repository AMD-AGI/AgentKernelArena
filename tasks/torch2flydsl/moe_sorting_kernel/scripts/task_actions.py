"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005'}
EXPECTED_CASES = [{'name': 'deepseek_M16_E256_k8', 'M': 16, 'E': 256, 'topk': 8}, {'name': 'deepseek_M8_E256_k8', 'M': 8, 'E': 256, 'topk': 8}, {'name': 'M16_E32_k2', 'M': 16, 'E': 32, 'topk': 2}, {'name': 'M64_E32_k2', 'M': 64, 'E': 32, 'topk': 2}, {'name': 'M16_E8_k2', 'M': 16, 'E': 8, 'topk': 2}, {'name': 'M64_E8_k2', 'M': 64, 'E': 8, 'topk': 2}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
