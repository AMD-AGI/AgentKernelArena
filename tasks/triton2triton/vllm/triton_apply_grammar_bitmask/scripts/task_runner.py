#!/usr/bin/env python3
"""Task runner for triton2triton/triton_apply_grammar_bitmask"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_apply_grammar_bitmask"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_apply_grammar_bitmask.py")

# (num_masks, vocab_size)
TEST_SHAPES = [
    (4, 1024),
    (8, 4096),
    (16, 8192),
    (32, 32000),
    (64, 65536),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_apply_grammar_bitmask(logits, logits_indices, bitmask, vocab_size):
    """CPU reference: unpack bitmask and apply to logits."""
    import torch
    logits = logits.clone()
    num_masks = bitmask.shape[0]
    for m in range(num_masks):
        logits_idx = logits_indices[m].item()
        for v in range(vocab_size):
            word_idx = v // 32
            bit_idx = v % 32
            bit = (bitmask[m, word_idx].item() >> bit_idx) & 1
            if bit == 0:
                logits[logits_idx, v] = float('-inf')
    return logits


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "apply_grammar_bitmask")
        assert hasattr(mod, "_apply_grammar_bitmask_kernel")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, (num_masks, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_total_logits = num_masks + 4
            logits = torch.randn(num_total_logits, vocab_size, device=device, dtype=torch.float32)
            logits_indices = torch.arange(num_masks, dtype=torch.int32, device=device)
            bitmask_words = (vocab_size + 31) // 32
            # Random bitmask with ~50% bits set
            bitmask = torch.randint(0, 2**31, (num_masks, bitmask_words), dtype=torch.int32, device=device)

            logits_gpu = logits.clone()
            mod.apply_grammar_bitmask(logits_gpu, logits_indices, bitmask, vocab_size)
            torch.cuda.synchronize()

            ref = reference_apply_grammar_bitmask(
                logits.cpu(), logits_indices.cpu(), bitmask.cpu(), vocab_size
            )

            # Check: where ref is -inf, gpu should be -inf; where ref is not -inf, gpu should match
            ref_neginf = ref.isinf() & (ref < 0)
            gpu_neginf = logits_gpu.cpu().isinf() & (logits_gpu.cpu() < 0)
            if not torch.equal(ref_neginf[:num_masks], gpu_neginf[:num_masks]):
                return False, f"Shape {i+1}: bitmask application mismatch"

            # Non-inf values should be unchanged
            non_inf_mask = ~ref_neginf[:num_masks]
            if not torch.allclose(logits_gpu.cpu()[:num_masks][non_inf_mask],
                                   ref[:num_masks][non_inf_mask]):
                return False, f"Shape {i+1}: non-masked values changed"

        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    num_masks, vocab_size = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    logits = torch.randn(num_masks, vocab_size, device=device, dtype=torch.float32)
    logits_indices = torch.arange(num_masks, dtype=torch.int32, device=device)
    bitmask_words = (vocab_size + 31) // 32
    bitmask = torch.randint(0, 2**31, (num_masks, bitmask_words), dtype=torch.int32, device=device)

    for _ in range(5):
        mod.apply_grammar_bitmask(logits.clone(), logits_indices, bitmask, vocab_size)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.apply_grammar_bitmask(logits.clone(), logits_indices, bitmask, vocab_size)
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
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
