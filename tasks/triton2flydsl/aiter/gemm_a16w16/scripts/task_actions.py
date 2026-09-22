"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm1_n1_k1', 'M': 1, 'N': 1, 'K': 1}, {'name': 'm3_n5_k2', 'M': 3, 'N': 5, 'K': 2}, {'name': 'm1024_n1024_k1024', 'M': 1024, 'N': 1024, 'K': 1024}, {'name': 'm2048_n2048_k2048', 'M': 2048, 'N': 2048, 'K': 2048}, {'name': 'dsr1_router_m32_n256_k7168', 'M': 32, 'N': 256, 'K': 7168}, {'name': 'gptoss_qkv_m128_n5120_k2880', 'M': 128, 'N': 5120, 'K': 2880}, {'name': 'gptoss_out_m256_n2880_k4096', 'M': 256, 'N': 2880, 'K': 4096}, {'name': 'gptoss_router_m128_n128_k2880', 'M': 128, 'N': 128, 'K': 2880}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006', 'perf8': 'case_0007'}
CORRECTNESS_COUNT = 8

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
