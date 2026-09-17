"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [(8, 32, 4), (32, 32, 4), (128, 64, 8), (32, 64, 8)]
PERFORMANCE_CASES = [(32, 64, 8), (128, 64, 8)]
PERFORMANCE_IDS = {'moe_sort_0': 'case_0003', 'moe_sort_1': 'case_0002'}

def check(h):
    for case in CORRECTNESS_CASES:
        ok, _ = h.run_test(*case)
        if not ok:
            raise RuntimeError(f"Correctness/output-contract failure for {case}")

def performance(h):
    return h.arena_benchmark(shapes=PERFORMANCE_CASES, warmup=10, iters=100)
