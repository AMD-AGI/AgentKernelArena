#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prompt_logprobs_token_ids"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_prompt_logprobs_token_ids"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prompt_logprobs_token_ids.py")

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
        assert hasattr(mod, "get_prompt_logprobs_token_ids"), "Missing get_prompt_logprobs_token_ids"
        assert hasattr(mod, "_prompt_logprobs_token_ids_kernel"), "Missing _prompt_logprobs_token_ids_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 32, 128),    # (batch, query_len, max_model_len)
    (8, 64, 256),
    (16, 128, 512),
    (32, 256, 1024),
    (64, 512, 2048),
]
PERF_SHAPE_IDX = 3

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch, qlen, max_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_tokens = batch * qlen
            query_start_loc = torch.arange(0, batch * qlen + 1, qlen, dtype=torch.int32, device=device)
            idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
            num_computed = torch.zeros(batch, dtype=torch.int32, device=device)
            all_token_ids = torch.randint(0, 1000, (batch, max_len), dtype=torch.int64, device=device)
            result = mod.get_prompt_logprobs_token_ids(num_tokens, query_start_loc, idx_mapping, num_computed, all_token_ids)
            torch.cuda.synchronize()
            # CPU ref
            ref = torch.empty(num_tokens, dtype=torch.int64, device=device)
            for b in range(batch):
                start = b * qlen
                nc = num_computed[b].item()
                for j in range(qlen):
                    ref[start + j] = all_token_ids[b, nc + 1 + j]
            if not torch.equal(result, ref):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch, qlen, max_len = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    num_tokens = batch * qlen
    query_start_loc = torch.arange(0, batch * qlen + 1, qlen, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)
    num_computed = torch.zeros(batch, dtype=torch.int32, device=device)
    all_token_ids = torch.randint(0, 1000, (batch, max_len), dtype=torch.int64, device=device)
    for _ in range(5): mod.get_prompt_logprobs_token_ids(num_tokens, query_start_loc, idx_mapping, num_computed, all_token_ids)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.get_prompt_logprobs_token_ids(num_tokens, query_start_loc, idx_mapping, num_computed, all_token_ids)
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
