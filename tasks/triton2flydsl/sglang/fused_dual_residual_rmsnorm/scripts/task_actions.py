"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'bs': 16, 'hidden': 4096, 'dtype': 'bf16'}, {'bs': 128, 'hidden': 4096, 'dtype': 'bf16'}, {'bs': 1, 'hidden': 8192, 'dtype': 'bf16'}, {'bs': 64, 'hidden': 5120, 'dtype': 'bf16'}, {'bs': 256, 'hidden': 2048, 'dtype': 'bf16'}, {'bs': 32, 'hidden': 3072, 'dtype': 'bf16'}, {'bs': 8, 'hidden': 4096, 'dtype': 'fp16'}, {'bs': 4, 'hidden': 1024, 'dtype': 'fp32'}]
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
    return h.run_performance()
