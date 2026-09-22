"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [(1, 128, 8, 128, 'f16', True), (1, 256, 8, 128, 'f16', True), (1, 512, 8, 128, 'f16', True), (1, 1024, 8, 128, 'f16', True), (1, 2048, 8, 128, 'f16', True), (4, 512, 16, 128, 'f16', True), (8, 256, 32, 128, 'f16', True), (8, 512, 64, 128, 'f16', True), (1, 512, 8, 128, 'bf16', True), (1, 1024, 8, 128, 'bf16', True)]
PERFORMANCE_CASES = [(1, 128, 8, 128, 'f16', True), (1, 256, 8, 128, 'f16', True), (1, 512, 8, 128, 'f16', True), (1, 1024, 8, 128, 'f16', True), (1, 2048, 8, 128, 'f16', True), (4, 512, 16, 128, 'f16', True), (8, 256, 32, 128, 'f16', True), (8, 512, 64, 128, 'f16', True), (1, 512, 8, 128, 'bf16', True), (1, 1024, 8, 128, 'bf16', True)]
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005', 'test_case_6': 'case_0006', 'test_case_7': 'case_0007', 'test_case_8': 'case_0008', 'test_case_9': 'case_0009'}

def check(h):
    result = h.run_correctness(shapes=CORRECTNESS_CASES, verbose=True)
    if not isinstance(result, dict) or result.get("correct") is not True or result.get("num_correct") != len(CORRECTNESS_CASES):
        raise RuntimeError(f"Incomplete or failed correctness check: {result}")

def performance(h):
    return h.arena_benchmark(shapes=PERFORMANCE_CASES, warmup=10, iters=100)
