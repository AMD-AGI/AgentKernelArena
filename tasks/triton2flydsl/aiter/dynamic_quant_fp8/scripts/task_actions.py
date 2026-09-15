"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm1_n32', 'M': 1, 'N': 32}, {'name': 'm32_n32', 'M': 32, 'N': 32}, {'name': 'm2_n16', 'M': 2, 'N': 16}, {'name': 'm10_n128', 'M': 10, 'N': 128}, {'name': 'm193_n75', 'M': 193, 'N': 75}, {'name': 'm1024_n128', 'M': 1024, 'N': 128}, {'name': 'm32_n8192', 'M': 32, 'N': 8192}, {'name': 'm400_n400', 'M': 400, 'N': 400}]
PERFORMANCE_IDS = {'perf1': 'case_0005', 'perf2': 'case_0011', 'perf3': 'case_0017', 'perf4': 'case_0023', 'perf5': 'case_0029', 'perf6': 'case_0035', 'perf7': 'case_0041', 'perf8': 'case_0047'}
CORRECTNESS_COUNT = 48

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
