"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'bs': 128, 'E': 8, 'hidden': 2048, 'topk': 2, 'cap': 30.0, 'bias': False}, {'bs': 64, 'E': 8, 'hidden': 2048, 'topk': 4, 'cap': 30.0, 'bias': False}, {'bs': 512, 'E': 8, 'hidden': 2048, 'topk': 2, 'cap': 30.0, 'bias': False}, {'bs': 256, 'E': 16, 'hidden': 4096, 'topk': 2, 'cap': 30.0, 'bias': False}, {'bs': 128, 'E': 32, 'hidden': 2048, 'topk': 2, 'cap': 30.0, 'bias': True}, {'bs': 64, 'E': 8, 'hidden': 2048, 'topk': 2, 'cap': 0.0, 'bias': False}, {'bs': 200, 'E': 64, 'hidden': 1024, 'topk': 1, 'cap': 30.0, 'bias': False}, {'bs': 128, 'E': 8, 'hidden': 2048, 'topk': 1, 'cap': 30.0, 'bias': True}]
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
