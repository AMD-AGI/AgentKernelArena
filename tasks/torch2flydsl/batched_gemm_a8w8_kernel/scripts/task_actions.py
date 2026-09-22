"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'b16_m32_n1280_k8192', 'b': 16, 'm': 32, 'n': 1280, 'k': 8192}, {'name': 'b16_m128_n1280_k8192', 'b': 16, 'm': 128, 'n': 1280, 'k': 8192}, {'name': 'b16_m64_n8192_k1024', 'b': 16, 'm': 64, 'n': 8192, 'k': 1024}, {'name': 'b16_m256_n8192_k1024', 'b': 16, 'm': 256, 'n': 8192, 'k': 1024}, {'name': 'b16_m512_n1280_k8192', 'b': 16, 'm': 512, 'n': 1280, 'k': 8192}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    from scripts.candidate_checks import audit_candidate_calls
    with audit_candidate_calls(h) as observed:
        if h.run_correctness(verbose=True) is not True:
            raise RuntimeError("Correctness/output-contract check failed")
    return sorted(observed)

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
