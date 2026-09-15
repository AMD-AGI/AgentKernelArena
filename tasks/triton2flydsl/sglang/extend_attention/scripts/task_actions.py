"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'prefix': [64, 128], 'extend': [32, 32], 'head': 32, 'kv_head': 8, 'Lq': 128, 'Lv': 128, 'causal': True}, {'prefix': [200], 'extend': [50], 'head': 16, 'kv_head': 16, 'Lq': 128, 'Lv': 128, 'causal': True}, {'prefix': [0, 64, 100], 'extend': [16, 16, 24], 'head': 28, 'kv_head': 4, 'Lq': 128, 'Lv': 128, 'causal': True}, {'prefix': [128, 128], 'extend': [40, 40], 'head': 16, 'kv_head': 2, 'Lq': 64, 'Lv': 64, 'causal': True}, {'prefix': [128], 'extend': [32], 'head': 16, 'kv_head': 16, 'Lq': 192, 'Lv': 128, 'causal': True}, {'prefix': [96], 'extend': [48], 'head': 32, 'kv_head': 8, 'Lq': 128, 'Lv': 128, 'causal': False}, {'prefix': [150], 'extend': [70], 'head': 8, 'kv_head': 1, 'Lq': 128, 'Lv': 128, 'causal': True}]
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
