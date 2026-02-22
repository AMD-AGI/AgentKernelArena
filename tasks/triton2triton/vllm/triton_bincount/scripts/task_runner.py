#!/usr/bin/env python3
"""Task runner for triton2triton/triton_bincount"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_bincount"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_bincount.py")

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
        assert hasattr(mod, "bincount"), "Missing bincount"
        assert hasattr(mod, "_bincount_kernel"), "Missing _bincount_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 128, 256),    # (batch, seq_len, vocab)
    (8, 256, 512),
    (16, 512, 1024),
    (32, 1024, 2048),
    (64, 2048, 4096),
]
PERF_SHAPE_IDX = 3

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, seq_len, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            prompt_len_val = seq_len // 2
            all_token_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.int32, device=device)
            prompt_len = torch.full((batch,), prompt_len_val, dtype=torch.int32, device=device)
            prefill_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
            prompt_mask = torch.zeros(batch, (vocab + 31) // 32, dtype=torch.int32, device=device)
            output_counts = torch.zeros(batch, vocab, dtype=torch.int32, device=device)
            mod.bincount(idx_mapping, all_token_ids, prompt_len, prefill_len, prompt_mask, output_counts, seq_len)
            torch.cuda.synchronize()
            # Reference for ALL outputs: prompt bitmask and output token counts.
            ref_prompt_mask = torch.zeros_like(prompt_mask)
            ref_output_counts = torch.zeros_like(output_counts)
            for b in range(batch):
                plen = prompt_len[b].item()
                flen = prefill_len[b].item()
                for j in range(plen):
                    tid = all_token_ids[b, j].item()
                    idx = tid // 32
                    bit = 1 << (tid % 32)
                    ref_prompt_mask[b, idx] |= bit
                for j in range(plen, flen):
                    tid = all_token_ids[b, j].item()
                    ref_output_counts[b, tid] += 1

            if not torch.equal(output_counts, ref_output_counts):
                return False, f"Shape {i+1}: output_bin_counts mismatch"
            if not torch.equal(prompt_mask, ref_prompt_mask):
                return False, f"Shape {i+1}: prompt_bin_mask mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch, seq_len, vocab = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
    all_token_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.int32, device=device)
    prompt_len = torch.full((batch,), seq_len // 2, dtype=torch.int32, device=device)
    prefill_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    prompt_mask = torch.zeros(batch, (vocab + 31) // 32, dtype=torch.int32, device=device)
    output_counts = torch.zeros(batch, vocab, dtype=torch.int32, device=device)
    for _ in range(5): mod.bincount(idx_mapping, all_token_ids, prompt_len, prefill_len, prompt_mask, output_counts, seq_len)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        prompt_mask.zero_(); output_counts.zero_()
        start_events[j].record()
        mod.bincount(idx_mapping, all_token_ids, prompt_len, prefill_len, prompt_mask, output_counts, seq_len)
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
