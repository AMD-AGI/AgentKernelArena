"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005'}
EXPECTED_CASES = [{'name': 'neox_full_reuse', 's': 2048, 'b': 2, 'h': 8, 'd': 128, 'rotary_pct': 1.0, 'rotate_style': 0, 'reuse': True, 'nope_first': False}, {'name': 'gptj_full_reuse', 's': 2048, 'b': 2, 'h': 8, 'd': 128, 'rotary_pct': 1.0, 'rotate_style': 1, 'reuse': True, 'nope_first': False}, {'name': 'neox_half_noreuse', 's': 1024, 'b': 4, 'h': 8, 'd': 128, 'rotary_pct': 0.5, 'rotate_style': 0, 'reuse': False, 'nope_first': False}, {'name': 'gptj_half_reuse', 's': 1024, 'b': 4, 'h': 8, 'd': 128, 'rotary_pct': 0.5, 'rotate_style': 1, 'reuse': True, 'nope_first': False}, {'name': 'neox_half_nopefirst', 's': 1024, 'b': 2, 'h': 16, 'd': 128, 'rotary_pct': 0.5, 'rotate_style': 0, 'reuse': True, 'nope_first': True}, {'name': 'neox_full_noreuse', 's': 512, 'b': 2, 'h': 16, 'd': 64, 'rotary_pct': 1.0, 'rotate_style': 0, 'reuse': False, 'nope_first': False}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
