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

    def reference_bad_words(
        logits, idx_mapping, bad_word_ids, offsets, num_bw,
        all_token_ids, prompt_len, total_len, input_ids, local_pos,
    ):
        """Small scalar reference for targeted irregular-shape cases."""
        ref = logits.clone()
        for logit_idx in range(logits.shape[0]):
            req_idx = int(idx_mapping[logit_idx].item())
            pos = int(local_pos[logit_idx].item())
            first_pos = logit_idx - pos
            prompt = int(prompt_len[req_idx].item())
            output_len = int(total_len[req_idx].item()) - prompt
            effective_len = output_len + pos

            for bw_idx in range(int(num_bw[req_idx].item())):
                start = int(offsets[req_idx, bw_idx].item())
                end = int(offsets[req_idx, bw_idx + 1].item())
                prefix_len = end - start - 1
                if prefix_len > effective_len:
                    continue

                matched = True
                for prefix_idx in range(prefix_len):
                    expected = int(bad_word_ids[req_idx, start + prefix_idx].item())
                    actual_pos = effective_len - prefix_len + prefix_idx
                    if actual_pos >= output_len:
                        actual = int(input_ids[first_pos + actual_pos - output_len].item())
                    else:
                        actual = int(all_token_ids[req_idx, prompt + actual_pos].item())
                    if expected != actual:
                        matched = False
                        break

                if matched:
                    last_token = int(bad_word_ids[req_idx, end - 1].item())
                    ref[logit_idx, last_token] = float("-inf")
        return ref

    def exact_error(name, actual, expected):
        if torch.equal(actual, expected):
            return None
        mismatch = (actual != expected).nonzero()
        first = mismatch[0].tolist()
        return (f"{name}: {mismatch.shape[0]} mismatched value(s); "
                f"first at {first}: got {actual[tuple(first)].item()}, "
                f"expected {expected[tuple(first)].item()}")

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

    # Irregular request state and speculative decoding. Rows are grouped by
    # request, but request 2 comes first and requests 1 and 2 each have multiple
    # speculative positions. Bad-word lists are ragged and contain lengths 1-4.
    # This also includes prefixes longer than the row's effective history.
    try:
        vocab = 96
        logits = (torch.arange(6 * vocab, device=device, dtype=torch.float32)
                  .reshape(6, vocab).to(torch.float16) / 100)
        idx_mapping = torch.tensor([2, 2, 0, 1, 1, 1], dtype=torch.int32, device=device)
        local_pos = torch.tensor([0, 1, 0, 0, 1, 2], dtype=torch.int32, device=device)

        # Request 0: [11], [21, 22], [20, 21, 23, 34]
        # Request 1: [41, 42, 43], [42, 44, 45]
        # Request 2: [51], [61, 62, 63], [61, 62, 64, 66]
        bad_word_ids = torch.zeros((4, 8), dtype=torch.int32, device=device)
        bad_word_ids[0, :7] = torch.tensor(
            [11, 21, 22, 20, 21, 23, 34], dtype=torch.int32, device=device)
        bad_word_ids[1, :6] = torch.tensor(
            [41, 42, 43, 42, 44, 45], dtype=torch.int32, device=device)
        bad_word_ids[2, :8] = torch.tensor(
            [51, 61, 62, 63, 61, 62, 64, 66], dtype=torch.int32, device=device)
        offsets = torch.tensor(
            [[0, 1, 3, 7, 7],
             [0, 3, 6, 6, 6],
             [0, 1, 4, 8, 8],
             [0, 0, 0, 0, 0]],
            dtype=torch.int32, device=device,
        )
        num_bw = torch.tensor([3, 2, 3, 0], dtype=torch.int32, device=device)
        prompt_len = torch.tensor([2, 1, 3, 0], dtype=torch.int32, device=device)
        total_len = torch.tensor([4, 2, 5, 0], dtype=torch.int32, device=device)
        all_token_ids = torch.zeros((4, 8), dtype=torch.int32, device=device)
        all_token_ids[0, 2:4] = torch.tensor([20, 21], dtype=torch.int32, device=device)
        all_token_ids[1, 1] = 41
        all_token_ids[2, 3:5] = torch.tensor([61, 62], dtype=torch.int32, device=device)
        input_ids = torch.tensor([64, 90, 91, 42, 44, 92], dtype=torch.int32, device=device)

        ref = reference_bad_words(
            logits, idx_mapping, bad_word_ids, offsets, num_bw,
            all_token_ids, prompt_len, total_len, input_ids, local_pos,
        )
        mod.apply_bad_words(
            logits, idx_mapping, bad_word_ids, offsets, num_bw,
            all_token_ids, prompt_len, total_len, input_ids, local_pos, 4,
        )
        torch.cuda.synchronize()
        err = exact_error("Irregular speculative case", logits, ref)
        if err:
            return False, err
    except Exception as e:
        return False, f"Irregular speculative case: exception: {e}"

    # Empty grids are valid no-ops for both an empty logits batch and a request
    # with no bad words. Keep real state tensors in the former case so only the
    # expanded/token dimension is empty.
    try:
        logits = torch.empty((0, 32), dtype=torch.float32, device=device)
        idx_mapping = torch.empty((0,), dtype=torch.int32, device=device)
        bad_word_ids = torch.tensor([[7]], dtype=torch.int32, device=device)
        offsets = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        num_bw = torch.tensor([1], dtype=torch.int32, device=device)
        all_token_ids = torch.zeros((1, 4), dtype=torch.int32, device=device)
        prompt_len = torch.tensor([1], dtype=torch.int32, device=device)
        total_len = torch.tensor([1], dtype=torch.int32, device=device)
        input_ids = torch.empty((0,), dtype=torch.int32, device=device)
        local_pos = torch.empty((0,), dtype=torch.int32, device=device)
        mod.apply_bad_words(
            logits, idx_mapping, bad_word_ids, offsets, num_bw,
            all_token_ids, prompt_len, total_len, input_ids, local_pos, 1,
        )

        logits = torch.arange(64, dtype=torch.float32, device=device).reshape(2, 32)
        ref = logits.clone()
        idx_mapping = torch.zeros((2,), dtype=torch.int32, device=device)
        bad_word_ids = torch.empty((1, 0), dtype=torch.int32, device=device)
        offsets = torch.zeros((1, 1), dtype=torch.int32, device=device)
        num_bw = torch.zeros((1,), dtype=torch.int32, device=device)
        all_token_ids = torch.zeros((1, 4), dtype=torch.int32, device=device)
        prompt_len = torch.tensor([1], dtype=torch.int32, device=device)
        total_len = torch.tensor([1], dtype=torch.int32, device=device)
        input_ids = torch.zeros((2,), dtype=torch.int32, device=device)
        local_pos = torch.tensor([0, 1], dtype=torch.int32, device=device)
        mod.apply_bad_words(
            logits, idx_mapping, bad_word_ids, offsets, num_bw,
            all_token_ids, prompt_len, total_len, input_ids, local_pos, 0,
        )
        torch.cuda.synchronize()
        err = exact_error("Zero-bad-words case", logits, ref)
        if err:
            return False, err
    except Exception as e:
        return False, f"Zero-sized case: exception: {e}"

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
                prepare_fn=lambda: logits_work.copy_(logits),
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
