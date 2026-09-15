"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'N': 1, 'dtype': 'fp32'}, {'N': 7, 'dtype': 'fp32'}, {'N': 128, 'dtype': 'fp32'}, {'N': 1000, 'dtype': 'fp32'}, {'N': 64, 'dtype': 'bf16'}, {'N': 256, 'dtype': 'fp16'}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005'}
CORRECTNESS_COUNT = 6

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
