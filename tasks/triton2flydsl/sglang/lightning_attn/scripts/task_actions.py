"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'B': 4, 'H': 32, 'D': 128, 'block': 32, 'dtype': 'bf16'}, {'B': 1, 'H': 32, 'D': 128, 'block': 32, 'dtype': 'bf16'}, {'B': 8, 'H': 16, 'D': 64, 'block': 32, 'dtype': 'bf16'}, {'B': 16, 'H': 8, 'D': 128, 'block': 64, 'dtype': 'bf16'}, {'B': 2, 'H': 40, 'D': 128, 'block': 32, 'dtype': 'fp16'}, {'B': 4, 'H': 16, 'D': 128, 'block': 64, 'dtype': 'fp32'}]
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
