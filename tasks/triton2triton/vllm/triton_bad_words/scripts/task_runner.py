#!/usr/bin/env python3
"""Task runner for triton2triton/triton_bad_words"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_bad_words"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_bad_words.py")

def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f: source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "apply_bad_words"), "Missing apply_bad_words"
        assert hasattr(mod, "_bad_words_kernel"), "Missing _bad_words_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256, 2),   # (batch, vocab, num_bad_words)
    (8, 1024, 4),
    (16, 4096, 8),
    (32, 8192, 16),
    (64, 16384, 8),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab, nbw) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            ref = logits.clone()
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            # Simple: single-token bad words (no prefix matching needed)
            max_tokens = nbw
            bad_word_ids = torch.randint(0, vocab, (batch, max_tokens), dtype=torch.int32, device=device)
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.zeros(batch, 128, dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), 10, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), 20, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
            torch.cuda.synchronize()
            # For single-token bad words, the last token should be masked
            for b in range(batch):
                for j in range(nbw):
                    tid = bad_word_ids[b, j].item()
                    ref[b, tid] = float("-inf")
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2):
                diff = (logits - ref).abs().max().item()
                return False, f"Shape {i+1}: max diff = {diff}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    # Multi-token bad words: exercise the prefix-matching loop (prefix_len >= 1),
    # which the single-token cases above never trigger. A bad word [t0..t_{L-1}] must
    # mask its final token t_{L-1} iff the last (L-1) output tokens equal [t0..t_{L-2}].
    multi_cases = [
        (4, 256, 2, 2),    # (batch, vocab, num_bad_words, word_len)
        (8, 512, 3, 2),
        (6, 1024, 3, 3),
    ]
    OUTPUT_LEN = 10
    PROMPT_LEN = 10
    for i, (batch, vocab, nbw, word_len) in enumerate(multi_cases):
        try:
            torch.manual_seed(1234 + i)
            prefix_len = word_len - 1
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            max_tokens = nbw * word_len
            bad_word_ids = torch.randint(0, vocab, (batch, max_tokens), dtype=torch.int32, device=device)
            # Each bad word occupies a contiguous span of length `word_len`.
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j * word_len
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.randint(0, vocab, (batch, 128), dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), PROMPT_LEN, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), PROMPT_LEN + OUTPUT_LEN, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            # Force a prefix match for one bad word per request so the match branch
            # is hit; the other bad words almost surely do not match (no-match branch).
            for b in range(batch):
                jj = b % nbw
                start = jj * word_len
                for t in range(prefix_len):
                    seq_pos = PROMPT_LEN + (OUTPUT_LEN - prefix_len + t)
                    all_token_ids[b, seq_pos] = bad_word_ids[b, start + t]

            # Reference from the intended semantics (pos=0, no spec input).
            ref = logits.clone()
            for b in range(batch):
                for j in range(nbw):
                    start = j * word_len
                    matched = True
                    for t in range(prefix_len):
                        expected = int(bad_word_ids[b, start + t].item())
                        actual = int(all_token_ids[b, PROMPT_LEN + (OUTPUT_LEN - prefix_len + t)].item())
                        if expected != actual:
                            matched = False
                            break
                    if matched:
                        last_token = int(bad_word_ids[b, start + word_len - 1].item())
                        ref[b, last_token] = float("-inf")

            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw,
                                all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
            torch.cuda.synchronize()
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2):
                diff = (logits - ref).abs().max().item()
                return False, f"Multi-token case {i+1} (word_len={word_len}): max diff = {diff}"
        except Exception as e:
            return False, f"Multi-token case {i+1}: exception: {e}"

    return True, None

def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab, nbw) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            bad_word_ids = torch.randint(0, vocab, (batch, nbw), dtype=torch.int32, device=device)
            offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
            for b in range(batch):
                for j in range(nbw + 1):
                    offsets[b, j] = j
            num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
            all_token_ids = torch.zeros(batch, 128, dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), 10, dtype=torch.int32, device=device)
            total_len = torch.full((batch,), 20, dtype=torch.int32, device=device)
            input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
            local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
            logits_work = logits.clone()
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                lambda: mod.apply_bad_words(
                    logits_work, idx_mapping, bad_word_ids, offsets, num_bw,
                    all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw,
                ),
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                target_ms=20.0,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_bad_words": nbw
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_bad_words": nbw
                }
            })

    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()
