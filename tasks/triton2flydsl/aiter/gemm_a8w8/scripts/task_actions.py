"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm1_n1_k1', 'M': 1, 'N': 1, 'K': 1}, {'name': 'm3_n5_k2', 'M': 3, 'N': 5, 'K': 2}, {'name': 'm16_n1024_k1024', 'M': 16, 'N': 1024, 'K': 1024}, {'name': 'm128_n8192_k512', 'M': 128, 'N': 8192, 'K': 512}, {'name': 'm256_n512_k8192', 'M': 256, 'N': 512, 'K': 8192}, {'name': 'm1024_n1024_k1024', 'M': 1024, 'N': 1024, 'K': 1024}, {'name': 'm2048_n2048_k2048', 'M': 2048, 'N': 2048, 'K': 2048}, {'name': 'm4096_n4096_k4096', 'M': 4096, 'N': 4096, 'K': 4096}, {'name': 'll3_70b_qkv_m256_n10240_k8192', 'M': 256, 'N': 10240, 'K': 8192}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0002', 'perf3': 'case_0004', 'perf4': 'case_0006', 'perf5': 'case_0008', 'perf6': 'case_0010', 'perf7': 'case_0012', 'perf8': 'case_0014', 'perf9': 'case_0016'}
CORRECTNESS_COUNT = 18

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
