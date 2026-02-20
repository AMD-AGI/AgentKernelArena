#!/usr/bin/env python3
"""Task runner for triton2triton/triton_rejection_random_sample"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_rejection_random_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rejection_random_sample.py")

TEST_SHAPES = [(4, 3, 8, 64), (8, 5, 16, 128), (16, 4, 16, 256), (32, 6, 32, 512), (64, 8, 64, 256)]
PERF_SHAPE_IDX = 3

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
        assert hasattr(mod, "rejection_random_sample"), "Missing wrapper"
        assert hasattr(mod, "rejection_random_sample_kernel"), "Missing kernel"
        return True, None
    except Exception as e:
        return False, str(e)

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, max_spec_len, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_draft_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
            total_tokens = sum(num_draft_per_req)
            draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total_tokens, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total_tokens, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
            recovered = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
            uniform = torch.rand(total_tokens, dtype=torch.float64, device=device)
            is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
            output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
            mod.rejection_random_sample(output, cu, draft_ids, draft_probs, target_probs, bonus, recovered, uniform, is_greedy, max_spec_len, vocab_size)
            torch.cuda.synchronize()
            # CPU ref
            ref = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32)
            for b in range(batch_size):
                start = sum(num_draft_per_req[:b])
                rejected = False
                for pos in range(max_draft):
                    if not rejected:
                        did = draft_ids[start + pos].item()
                        dp = draft_probs[start + pos, did].item()
                        tp = target_probs[start + pos, did].item()
                        up = uniform[start + pos].item()
                        if dp > 0 and tp / dp >= up:
                            ref[b, pos] = did
                        else:
                            rejected = True
                            ref[b, pos] = recovered[start + pos].item()
                if not rejected:
                    ref[b, max_draft] = bonus[b].item()
            ref = ref.to(device)
            if not torch.equal(output, ref):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return -1.0
    device = "cuda"
    batch_size, max_draft, max_spec_len, vocab_size = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    num_draft_per_req = [max_draft] * batch_size
    cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
    total_tokens = sum(num_draft_per_req)
    draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
    draft_probs = torch.rand(total_tokens, vocab_size, device=device)
    draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
    target_probs = torch.rand(total_tokens, vocab_size, device=device)
    target_probs = target_probs / target_probs.sum(-1, keepdim=True)
    bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
    recovered = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
    uniform = torch.rand(total_tokens, dtype=torch.float64, device=device)
    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(5):
        output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
        mod.rejection_random_sample(output, cu, draft_ids, draft_probs, target_probs, bonus, recovered, uniform, is_greedy, max_spec_len, vocab_size)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
        start_events[j].record()
        mod.rejection_random_sample(output, cu, draft_ids, draft_probs, target_probs, bonus, recovered, uniform, is_greedy, max_spec_len, vocab_size)
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
        print(f"Compilation: {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms"); sys.exit(0)

if __name__ == "__main__": main()
