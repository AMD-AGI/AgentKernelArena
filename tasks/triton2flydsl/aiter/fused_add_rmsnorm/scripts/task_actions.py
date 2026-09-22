"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm1_n4', 'M': 1, 'N': 4}, {'name': 'm1_n65536', 'M': 1, 'N': 65536}, {'name': 'm256_n4096', 'M': 256, 'N': 4096}, {'name': 'm4096_n8192', 'M': 4096, 'N': 8192}, {'name': 'm873_n1245', 'M': 873, 'N': 1245}, {'name': 'm8192_n8192', 'M': 8192, 'N': 8192}, {'name': 'm2048_n4096', 'M': 2048, 'N': 4096}, {'name': 'm768_n2048', 'M': 768, 'N': 2048}, {'name': 'm64_n512', 'M': 64, 'N': 512}]
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
        if h.run_correctness() is not True:
            raise RuntimeError("Correctness/output-contract failure")
    return sorted(observed)


def performance(h):
    return h.run_benchmark()
