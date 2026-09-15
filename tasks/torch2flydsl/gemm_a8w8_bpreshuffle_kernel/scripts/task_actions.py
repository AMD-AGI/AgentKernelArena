"""Explicit model/production baseline and candidate action bindings."""
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004'}
EXPECTED_CASES = [{'name': 'skinny_m16_n5120_k1280', 'm': 16, 'n': 5120, 'k': 1280, 'tile_m': 16, 'tile_n': 64, 'tile_k': 256}, {'name': 'm64_n5120_k1280', 'm': 64, 'n': 5120, 'k': 1280, 'tile_m': 32, 'tile_n': 64, 'tile_k': 256}, {'name': 'm512_n5120_k1280', 'm': 512, 'n': 5120, 'k': 1280, 'tile_m': 128, 'tile_n': 128, 'tile_k': 256}, {'name': 'm1024_n8192_k1024', 'm': 1024, 'n': 8192, 'k': 1024, 'tile_m': 128, 'tile_n': 128, 'tile_k': 128}, {'name': 'm2048_n5120_k1280', 'm': 2048, 'n': 5120, 'k': 1280, 'tile_m': 128, 'tile_n': 128, 'tile_k': 256}]

def select_role(h, role, provided):
    h.ARENA_PROVIDED_BASELINE = provided
    if h.SHAPES != EXPECTED_CASES:
        raise ValueError("Harness case definitions disagree with protected manifest")

def check(h):
    if h.run_correctness(verbose=True) is not True:
        raise RuntimeError("Correctness/output-contract check failed")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100, verbose=True)
