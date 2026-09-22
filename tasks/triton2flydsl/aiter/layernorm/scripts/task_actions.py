"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm2_n128', 'M': 2, 'N': 128}, {'name': 'm1_n4', 'M': 1, 'N': 4}, {'name': 'm128_n2', 'M': 128, 'N': 2}, {'name': 'm1_n128', 'M': 1, 'N': 128}, {'name': 'm359_n1', 'M': 359, 'N': 1}, {'name': 'm1_n359', 'M': 1, 'N': 359}, {'name': 'm1_n131072', 'M': 1, 'N': 131072}, {'name': 'm1_n89999', 'M': 1, 'N': 89999}, {'name': 'm4096_n4096', 'M': 4096, 'N': 4096}, {'name': 'm2048_n8192', 'M': 2048, 'N': 8192}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0002', 'perf3': 'case_0004', 'perf4': 'case_0006', 'perf5': 'case_0008', 'perf6': 'case_0010', 'perf7': 'case_0012', 'perf8': 'case_0014', 'perf9': 'case_0016', 'perf10': 'case_0018'}
CORRECTNESS_COUNT = 20

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
