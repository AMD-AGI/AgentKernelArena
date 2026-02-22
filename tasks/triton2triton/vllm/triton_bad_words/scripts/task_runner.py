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
PERF_SHAPE_IDX = 3

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
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch, vocab, nbw = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
    idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
    bad_word_ids = torch.randint(0, vocab, (batch, nbw), dtype=torch.int32, device=device)
    offsets = torch.zeros(batch, nbw + 1, dtype=torch.int32, device=device)
    for b in range(batch):
        for j in range(nbw + 1): offsets[b, j] = j
    num_bw = torch.full((batch,), nbw, dtype=torch.int32, device=device)
    all_token_ids = torch.zeros(batch, 128, dtype=torch.int32, device=device)
    prompt_len = torch.full((batch,), 10, dtype=torch.int32, device=device)
    total_len = torch.full((batch,), 20, dtype=torch.int32, device=device)
    input_ids = torch.zeros(batch, dtype=torch.int32, device=device)
    local_pos = torch.zeros(batch, dtype=torch.int32, device=device)
    for _ in range(5): mod.apply_bad_words(logits.clone(), idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        l = logits.clone()
        start_events[j].record()
        mod.apply_bad_words(l, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, nbw)
        end_events[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


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
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)

if __name__ == "__main__": main()
