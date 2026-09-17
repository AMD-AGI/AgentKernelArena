"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [(64, 128, 8, 'bf16'), (128, 128, 6, 'bf16'), (32, 64, 4, 'f32'), (256, 128, 8, 'bf16'), (512, 128, 6, 'bf16')]
PERFORMANCE_CASES = [(256, 128, 8, 'bf16'), (512, 128, 6, 'bf16')]
PERFORMANCE_IDS = {'topk_0': 'case_0003', 'topk_1': 'case_0004'}

def check(h):
    for case in CORRECTNESS_CASES:
        ok, _ = h.run_test(*case)
        if not ok:
            raise RuntimeError(f"Correctness/output-contract failure for {case}")

def performance(h):
    return h.arena_benchmark(warmup=10, iters=100)
