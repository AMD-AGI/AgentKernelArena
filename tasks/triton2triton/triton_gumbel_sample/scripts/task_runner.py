#!/usr/bin/env python3
"""Task runner for triton2triton/triton_gumbel_sample"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_gumbel_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_gumbel_sample.py")

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
        assert hasattr(mod, "gumbel_sample"), "Missing gumbel_sample"
        assert hasattr(mod, "_gumbel_sample_kernel"), "Missing _gumbel_sample_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256),
    (8, 1024),
    (16, 4096),
    (32, 8192),
    (64, 32768),
]
PERF_SHAPE_IDX = 3

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (num_reqs, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(num_reqs, vocab_size, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=device)
            pos = torch.arange(num_reqs, dtype=torch.int64, device=device)
            # 1) Temperature=0 path must match plain argmax exactly.
            temp_zero = torch.zeros(num_reqs, device=device)
            result_argmax = mod.gumbel_sample(logits, idx_mapping, temp_zero, seed, pos, False)
            ref_argmax = logits.argmax(dim=-1)
            if not torch.equal(result_argmax, ref_argmax):
                return False, f"Shape {i+1}: temp=0 argmax mismatch"

            # 2) Stochastic path should be deterministic for fixed inputs/seeds/pos.
            temp = torch.ones(num_reqs, device=device)
            result = mod.gumbel_sample(logits, idx_mapping, temp, seed, pos, False)
            result_repeat = mod.gumbel_sample(logits, idx_mapping, temp, seed, pos, False)
            if not torch.equal(result, result_repeat):
                return False, f"Shape {i+1}: non-deterministic under fixed seed/pos"

            torch.cuda.synchronize()
            assert result.shape == (num_reqs,), f"Wrong shape: {result.shape}"
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    num_reqs, vocab_size = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    logits = torch.randn(num_reqs, vocab_size, device=device, dtype=torch.float32)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    temp = torch.ones(num_reqs, device=device)
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=device)
    pos = torch.arange(num_reqs, dtype=torch.int64, device=device)
    for _ in range(5): mod.gumbel_sample(logits, idx_mapping, temp, seed, pos, False)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.gumbel_sample(logits, idx_mapping, temp, seed, pos, False)
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
