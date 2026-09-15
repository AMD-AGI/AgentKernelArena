"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'N': 128, 'H': 32, 'D': 128, 'dtype': 'bf16'}, {'N': 1, 'H': 32, 'D': 128, 'dtype': 'bf16'}, {'N': 256, 'H': 8, 'D': 64, 'dtype': 'bf16'}, {'N': 64, 'H': 16, 'D': 256, 'dtype': 'bf16'}, {'N': 100, 'H': 40, 'D': 192, 'dtype': 'bf16'}, {'N': 512, 'H': 64, 'D': 128, 'dtype': 'fp16'}, {'N': 33, 'H': 12, 'D': 128, 'dtype': 'fp32'}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006'}
CORRECTNESS_COUNT = 7

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
    return h.run_performance()
