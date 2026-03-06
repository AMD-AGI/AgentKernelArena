#!/usr/bin/env python3
"""Task runner for triton2triton/triton_min_p"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_min_p"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_min_p.py")

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
        assert hasattr(mod, "apply_min_p"), "Missing apply_min_p"
        assert hasattr(mod, "_min_p_kernel"), "Missing _min_p_kernel"
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
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def run_correctness():
    import torch, math
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            min_p_val = 0.1
            min_p = torch.full((batch,), min_p_val, device=device, dtype=torch.float32)
            ref = logits.clone()
            for b in range(batch):
                max_val = ref[b].max().item()
                threshold = max_val + math.log(min_p_val)
                ref[b] = torch.where(ref[b] < threshold, torch.tensor(float("-inf"), device=device), ref[b])
            mod.apply_min_p(logits, idx_mapping, min_p)
            torch.cuda.synchronize()
            if not torch.allclose(logits, ref, atol=1e-2, rtol=1e-2, equal_nan=True):
                return False, f"Shape {i+1}: max diff = {(logits - ref).abs().max().item()}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            min_p = torch.full((batch,), 0.1, device=device, dtype=torch.float32)
            for _ in range(WARMUP_ITERATIONS): mod.apply_min_p(logits.clone(), idx_mapping, min_p)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                l = logits.clone()
                start_events[j].record()
                mod.apply_min_p(l, idx_mapping, min_p)
                end_events[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "batch": batch,
                    "vocab": vocab
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab
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
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()
