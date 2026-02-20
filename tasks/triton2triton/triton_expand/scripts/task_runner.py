#!/usr/bin/env python3
"""Task runner for triton2triton/triton_expand"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_expand"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_expand.py")

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
        assert hasattr(mod, "expand_batch_to_tokens"), "Missing expand_batch_to_tokens"
        assert hasattr(mod, "expand_kernel"), "Missing expand_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 3),   # (batch, tokens_per_req)
    (8, 5),
    (16, 8),
    (32, 10),
    (64, 16),
]
PERF_SHAPE_IDX = 3

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, tpr) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randint(0, 100, (batch_size,), dtype=torch.int32, device=device)
            counts = torch.full((batch_size,), tpr, dtype=torch.int32, device=device)
            cu = torch.cumsum(counts, dim=0)
            num_tokens = int(cu[-1].item())
            result = mod.expand_batch_to_tokens(x, cu, num_tokens)
            torch.cuda.synchronize()
            # CPU ref
            ref = []
            for b in range(batch_size):
                ref.extend([x[b].item()] * tpr)
            ref_t = torch.tensor(ref, dtype=x.dtype, device=device)
            if not torch.equal(result, ref_t):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch_size, tpr = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    x = torch.randint(0, 100, (batch_size,), dtype=torch.int32, device=device)
    counts = torch.full((batch_size,), tpr, dtype=torch.int32, device=device)
    cu = torch.cumsum(counts, dim=0)
    num_tokens = int(cu[-1].item())
    for _ in range(5): mod.expand_batch_to_tokens(x, cu, num_tokens)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.expand_batch_to_tokens(x, cu, num_tokens)
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
