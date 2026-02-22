#!/usr/bin/env python3
"""Task runner for triton2triton/triton_rejection_greedy_sample"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_rejection_greedy_sample"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_rejection_greedy_sample.py")

TEST_SHAPES = [
    (4, 3, 8),   # (batch_size, max_draft_tokens, max_spec_len)
    (8, 5, 16),
    (16, 4, 16),
    (32, 6, 32),
    (64, 8, 64),
]
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
        assert hasattr(mod, "rejection_greedy_sample"), "Missing rejection_greedy_sample"
        assert hasattr(mod, "rejection_greedy_sample_kernel"), "Missing rejection_greedy_sample_kernel"
        return True, None
    except Exception as e:
        return False, str(e)

def cpu_reference(draft_token_ids_list, target_argmax_list, bonus_token_ids, max_spec_len):
    """CPU reference for greedy rejection sampling."""
    import torch
    batch_size = len(draft_token_ids_list)
    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32)
    for b in range(batch_size):
        draft = draft_token_ids_list[b]
        target = target_argmax_list[b]
        rejected = False
        for pos in range(len(draft)):
            if not rejected:
                output[b, pos] = target[pos]
                if draft[pos] != target[pos]:
                    rejected = True
        if not rejected:
            output[b, len(draft)] = bonus_token_ids[b]
    return output

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, max_spec_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            vocab_size = 100
            num_draft_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
            total_tokens = sum(num_draft_per_req)
            draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
            target_argmax = draft_ids.clone()
            # Make some mismatches
            for b in range(batch_size):
                start = sum(num_draft_per_req[:b])
                if max_draft > 1:
                    pos = torch.randint(0, max_draft, (1,)).item()
                    target_argmax[start + pos] = (draft_ids[start + pos] + 1) % vocab_size
            bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
            output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
            mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
            torch.cuda.synchronize()
            # CPU ref
            draft_list, target_list = [], []
            for b in range(batch_size):
                start = sum(num_draft_per_req[:b])
                end = start + num_draft_per_req[b]
                draft_list.append(draft_ids[start:end].cpu().tolist())
                target_list.append(target_argmax[start:end].cpu().tolist())
            ref = cpu_reference(draft_list, target_list, bonus.cpu().tolist(), max_spec_len).to(device)
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
    batch_size, max_draft, max_spec_len = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    vocab_size = 100
    num_draft_per_req = [max_draft] * batch_size
    cu = torch.cumsum(torch.tensor(num_draft_per_req, dtype=torch.int32), dim=0).to(device)
    total_tokens = sum(num_draft_per_req)
    draft_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.int32, device=device)
    target_argmax = draft_ids.clone()
    bonus = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
    for _ in range(5):
        mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
    torch.cuda.synchronize()
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        output.fill_(-1)
        start_events[j].record()
        mod.rejection_greedy_sample(output, cu, draft_ids, target_argmax, bonus, None, max_spec_len)
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

if __name__ == "__main__":
    main()
