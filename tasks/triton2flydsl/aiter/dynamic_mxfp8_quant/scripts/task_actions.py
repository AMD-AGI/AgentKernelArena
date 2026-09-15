"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm1_k32', 'shape': (1, 32)}, {'name': 'm1_k128', 'shape': (1, 128)}, {'name': 'm8_k64', 'shape': (8, 64)}, {'name': 'm16_k128', 'shape': (16, 128)}, {'name': 'm32_k256', 'shape': (32, 256)}, {'name': 'm64_k512', 'shape': (64, 512)}, {'name': 'm128_k1024', 'shape': (128, 1024)}, {'name': 'm137_k64', 'shape': (137, 64)}, {'name': 'm256_k32', 'shape': (256, 32)}, {'name': 'b4_m8_k128', 'shape': (4, 8, 128)}]
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
    return h.run_benchmark()
