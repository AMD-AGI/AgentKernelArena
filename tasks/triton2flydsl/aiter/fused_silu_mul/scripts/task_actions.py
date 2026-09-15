"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'r4_l64', 'rows': 4, 'last': 64}, {'name': 'r128_l256', 'rows': 128, 'last': 256}, {'name': 'r31_l500', 'rows': 31, 'last': 500}, {'name': 'glm47_tp4_decode4', 'rows': 32, 'last': 768}, {'name': 'kimi_k25_tp4_decode4', 'rows': 32, 'last': 1024}, {'name': 'glm47_tp4_rows256x8', 'rows': 2048, 'last': 768}, {'name': 'kimi_k25_tp4_rows256x8', 'rows': 2048, 'last': 1024}, {'name': 'glm47_tp4_pref8190_dec3', 'rows': 65544, 'last': 768}, {'name': 'kimi_k25_tp4_pref7235_dec3', 'rows': 57904, 'last': 1024}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0002', 'perf3': 'case_0004', 'perf4': 'case_0006', 'perf5': 'case_0008', 'perf6': 'case_0010', 'perf7': 'case_0012', 'perf8': 'case_0014', 'perf9': 'case_0016'}
CORRECTNESS_COUNT = 18

def select_role(h, role, provided):
    import os
    h.ARENA_FINAL_CANDIDATE = role == "candidate" and os.environ.get("ARENA_EVAL_PHASE") != "task_validation"
    if provided:
        raise ValueError("This task uses a frozen initial candidate baseline")
    if h.TEST_SHAPES != EXPECTED_SHAPES:
        raise ValueError("Protected case manifest disagrees with harness")

def check(h):
    from scripts.candidate_checks import audit_candidate_calls
    with audit_candidate_calls(h) as observed:
        result = h.run_correctness()
        if isinstance(result, tuple):
            if result[0] is not True or len(result[2]) != CORRECTNESS_COUNT:
                raise RuntimeError(f"Incomplete/failed correctness evidence: {result}")
        elif result is not True:
            raise RuntimeError(f"Correctness/output-contract failure: {result}")
    return sorted(observed)


def performance(h):
    return h.run_benchmark()
