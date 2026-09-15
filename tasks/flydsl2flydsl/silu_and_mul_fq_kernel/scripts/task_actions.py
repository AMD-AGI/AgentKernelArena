"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [(64, 1024, 2, 'fp4'), (128, 1024, 2, 'fp4'), (256, 1024, 2, 'fp4'), (512, 1024, 2, 'fp4'), (1024, 1024, 2, 'fp4')]
PERFORMANCE_CASES = [(64, 1024, 2, 'fp4'), (128, 1024, 2, 'fp4'), (256, 1024, 2, 'fp4'), (512, 1024, 2, 'fp4'), (1024, 1024, 2, 'fp4')]
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}

def check(h):
    result = h.run_correctness(shapes=CORRECTNESS_CASES, verbose=True)
    if not isinstance(result, dict) or result.get("correct") is not True or result.get("num_correct") != len(CORRECTNESS_CASES):
        raise RuntimeError(f"Incomplete or failed correctness check: {result}")

def performance(h):
    return h.arena_benchmark(shapes=PERFORMANCE_CASES, warmup=10, iters=100)
