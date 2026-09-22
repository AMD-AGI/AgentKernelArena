"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [(3, 1, (8, 1), 'per_token'), (3, 1, (16, 1), 'per_token'), (3, 2, (8, 1), 'per_token'), (3, 4, (16, 1), 'per_token'), (81, 1, (8, 1), 'per_token'), (3, 1, (8, 1), 'per_tensor'), (3, 1, (16, 1), 'per_tensor'), (81, 1, (16, 1), 'per_token')]
PERFORMANCE_CASES = [(3, 1, (8, 1), 'per_token'), (3, 1, (16, 1), 'per_token'), (3, 2, (8, 1), 'per_token'), (3, 4, (16, 1), 'per_token'), (81, 1, (8, 1), 'per_token'), (3, 1, (8, 1), 'per_tensor'), (3, 1, (16, 1), 'per_tensor'), (81, 1, (16, 1), 'per_token')]
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005', 'test_case_6': 'case_0006', 'test_case_7': 'case_0007'}

def check(h):
    result = h.run_correctness(shapes=CORRECTNESS_CASES, verbose=True)
    if not isinstance(result, dict) or result.get("correct") is not True or result.get("num_correct") != len(CORRECTNESS_CASES):
        raise RuntimeError(f"Incomplete or failed correctness check: {result}")

def performance(h):
    return h.arena_benchmark(shapes=PERFORMANCE_CASES, warmup=10, iters=100)
