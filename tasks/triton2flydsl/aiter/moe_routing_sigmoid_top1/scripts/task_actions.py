"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'name': 'm256_k2048_n16', 'M': 256, 'K': 2048, 'N': 16, 'shared': False}, {'name': 'm1024_k4096_n32', 'M': 1024, 'K': 4096, 'N': 32, 'shared': False}, {'name': 'm2048_k5120_n128', 'M': 2048, 'K': 5120, 'N': 128, 'shared': False}, {'name': 'm512_k4096_n64_shared', 'M': 512, 'K': 4096, 'N': 64, 'shared': True}, {'name': 'm4096_k1024_n128', 'M': 4096, 'K': 1024, 'N': 128, 'shared': False}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004'}
CORRECTNESS_COUNT = 5

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
