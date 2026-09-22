"""Task-owned initial Triton and final FlyDSL action bindings."""
EXPECTED_SHAPES = [{'nt': 64, 'n_qh': 28, 'n_kh': 4, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': False, 'neox': True}, {'nt': 128, 'n_qh': 28, 'n_kh': 4, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': False, 'neox': True}, {'nt': 32, 'n_qh': 16, 'n_kh': 2, 'hd': 128, 'rd': 128, 'section': [24, 20, 20], 'interleaved': False, 'neox': True}, {'nt': 64, 'n_qh': 28, 'n_kh': 4, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': False, 'neox': False}, {'nt': 48, 'n_qh': 12, 'n_kh': 2, 'hd': 128, 'rd': 64, 'section': [8, 12, 12], 'interleaved': False, 'neox': True}, {'nt': 96, 'n_qh': 28, 'n_kh': 4, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': True, 'neox': True}, {'nt': 16, 'n_qh': 8, 'n_kh': 1, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': True, 'neox': False}, {'nt': 1, 'n_qh': 28, 'n_kh': 4, 'hd': 128, 'rd': 128, 'section': [16, 24, 24], 'interleaved': False, 'neox': True}]
PERFORMANCE_IDS = {'perf1': 'case_0000', 'perf2': 'case_0001', 'perf3': 'case_0002', 'perf4': 'case_0003', 'perf5': 'case_0004', 'perf6': 'case_0005', 'perf7': 'case_0006', 'perf8': 'case_0007'}
CORRECTNESS_COUNT = 8

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
        if (not isinstance(result, tuple) or len(result) != 3
                or result[0] is not True or not isinstance(result[2], list)
                or len(result[2]) != CORRECTNESS_COUNT):
            raise RuntimeError(f"Incomplete/failed correctness evidence: {result}")
        for i, record in enumerate(result[2], 1):
            if record.get("shape_id") != i or record.get("passed") is not True or "error" in record:
                raise RuntimeError(f"Invalid or failed correctness case: {record}")
    return sorted(observed)


def performance(h):
    return h.run_performance()
