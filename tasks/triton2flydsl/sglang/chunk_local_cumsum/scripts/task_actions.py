"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'ndim': 3, 'B': 1, 'T': 1024, 'H': 16, 'reverse': False, 'scale': None}, {'ndim': 3, 'B': 1, 'T': 4096, 'H': 32, 'reverse': False, 'scale': None}, {'ndim': 3, 'B': 2, 'T': 512, 'H': 16, 'reverse': False, 'scale': None}, {'ndim': 3, 'B': 1, 'T': 1000, 'H': 32, 'reverse': False, 'scale': None}, {'ndim': 3, 'B': 4, 'T': 256, 'H': 16, 'reverse': False, 'scale': None}, {'ndim': 3, 'B': 1, 'T': 1024, 'H': 16, 'reverse': True, 'scale': None}, {'ndim': 3, 'B': 1, 'T': 1024, 'H': 16, 'reverse': False, 'scale': 0.5}, {'ndim': 4, 'B': 1, 'T': 512, 'H': 8, 'S': 64, 'reverse': False, 'scale': None}, {'ndim': 4, 'B': 1, 'T': 1024, 'H': 4, 'S': 128, 'reverse': False, 'scale': None}, {'ndim': 4, 'B': 2, 'T': 256, 'H': 8, 'S': 64, 'reverse': True, 'scale': None}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006', 'perf8': 'case_0007', 'perf9': 'case_0008', 'perf10': 'case_0009'}
CORRECTNESS_COUNT = 10

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
