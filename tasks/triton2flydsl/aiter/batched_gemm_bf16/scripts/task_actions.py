"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'b16_m1_n1_k1', 'B': 16, 'M': 1, 'N': 1, 'K': 1}, {'name': 'b16_m3_n5_k2', 'B': 16, 'M': 3, 'N': 5, 'K': 2}, {'name': 'b16_m16_n16_k16', 'B': 16, 'M': 16, 'N': 16, 'K': 16}, {'name': 'b16_m128_n256_k512', 'B': 16, 'M': 128, 'N': 256, 'K': 512}, {'name': 'b16_m256_n512_k1024', 'B': 16, 'M': 256, 'N': 512, 'K': 1024}, {'name': 'b16_m1_n1280_k1024', 'B': 16, 'M': 1, 'N': 1280, 'K': 1024}, {'name': 'b16_m512_n1024_k1024', 'B': 16, 'M': 512, 'N': 1024, 'K': 1024}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0002', 'perf3': 'case_0004', 'perf4': 'case_0006', 'perf5': 'case_0008', 'perf6': 'case_0010', 'perf7': 'case_0012'}
CORRECTNESS_COUNT = 14

def select_role(h, role, provided):
    if provided:
        raise ValueError("This task uses a frozen initial candidate baseline")
    if h.TEST_SHAPES != EXPECTED_SHAPES:
        raise ValueError("Protected case manifest disagrees with harness")

def check(h):
    result = h.run_correctness()
    if isinstance(result, tuple):
        if result[0] is not True or len(result[2]) != CORRECTNESS_COUNT:
            raise RuntimeError(f"Incomplete/failed correctness evidence: {result}")
    elif result is not True:
        raise RuntimeError(f"Correctness/output-contract failure: {result}")

def performance(h):
    return h.run_benchmark()
