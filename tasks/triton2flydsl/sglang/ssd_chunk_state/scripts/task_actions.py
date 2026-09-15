"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'b': 1, 'H': 8, 'P': 64, 'G': 1, 'N': 128, 'cs': 128, 'C': 2, 'dtype': 'bf16'}, {'b': 2, 'H': 16, 'P': 64, 'G': 1, 'N': 128, 'cs': 256, 'C': 2, 'dtype': 'bf16'}, {'b': 1, 'H': 24, 'P': 128, 'G': 8, 'N': 64, 'cs': 128, 'C': 3, 'dtype': 'bf16'}, {'b': 1, 'H': 4, 'P': 64, 'G': 1, 'N': 64, 'cs': 64, 'C': 2, 'dtype': 'bf16'}, {'b': 2, 'H': 8, 'P': 64, 'G': 2, 'N': 128, 'cs': 128, 'C': 2, 'dtype': 'fp16'}, {'b': 1, 'H': 8, 'P': 96, 'G': 1, 'N': 80, 'cs': 128, 'C': 2, 'dtype': 'fp32'}]
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
