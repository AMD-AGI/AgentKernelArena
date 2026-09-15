"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [(2, 4, 64, 8, 8, 64, 16, 0, 0.0), (1, 1, 128, 16, 4, 64, 16, 0, 0.0), (4, 1, 256, 32, 8, 128, 16, 0, 0.0), (2, 8, 128, 16, 16, 64, 32, 32, 0.0), (1, 16, 512, 8, 8, 128, 32, 64, 1.0), (1, 8, 1024, 8, 8, 128, 32, 0, 0.0), (1, 16, 768, 8, 8, 64, 32, 0, 1.0)]
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
