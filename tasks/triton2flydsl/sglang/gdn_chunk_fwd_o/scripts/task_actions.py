"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [(1, 1024, 8, 16, 128, 128), (2, 512, 8, 16, 128, 128), (1, 2048, 8, 16, 128, 128), (1, 1024, 16, 32, 128, 128), (4, 256, 8, 16, 128, 128), (1, 1024, 16, 16, 128, 128), (1, 4096, 8, 16, 128, 128), (1, 1024, 8, 16, 128, 64)]
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
