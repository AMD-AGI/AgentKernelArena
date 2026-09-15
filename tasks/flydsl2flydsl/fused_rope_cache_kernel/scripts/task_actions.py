"""Protected action bindings; no agent/family runtime dispatch."""
CORRECTNESS_CASES = [{'num_tokens': 16, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 64, 'block_size': 16}, {'num_tokens': 32, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 64, 'block_size': 16}, {'num_tokens': 32, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 64, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 64, 'num_q_heads': 64, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 128, 'num_q_heads': 64, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}]
PERFORMANCE_CASES = [{'num_tokens': 16, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 64, 'block_size': 16}, {'num_tokens': 32, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 64, 'block_size': 16}, {'num_tokens': 32, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 64, 'num_q_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 64, 'num_q_heads': 64, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}, {'num_tokens': 128, 'num_q_heads': 64, 'num_kv_heads': 8, 'head_dim': 128, 'block_size': 16}]
PERFORMANCE_IDS = {'test_case_0': 'case_0000', 'test_case_1': 'case_0001', 'test_case_2': 'case_0002', 'test_case_3': 'case_0003', 'test_case_4': 'case_0004', 'test_case_5': 'case_0005'}

def check(h):
    result = h.run_correctness(configs=CORRECTNESS_CASES, verbose=True)
    if not isinstance(result, dict) or result.get("correct") is not True or result.get("num_correct") != len(CORRECTNESS_CASES):
        raise RuntimeError(f"Incomplete or failed correctness check: {result}")

def performance(h):
    return h.arena_benchmark(configs=PERFORMANCE_CASES, warmup=10, iters=100)
