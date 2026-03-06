#!/usr/bin/env python3
"""Task runner for triton2triton/triton_sample_recovered_tokens"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_sample_recovered_tokens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_sample_recovered_tokens.py")

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
        assert hasattr(mod, "sample_recovered_tokens"), "Missing sample_recovered_tokens"
        assert hasattr(mod, "sample_recovered_tokens_kernel"), "Missing sample_recovered_tokens_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 3, 64),   # (batch, max_draft, vocab)
    (8, 5, 128),
    (16, 4, 256),
    (32, 6, 512),
    (64, 8, 1024),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def reference_sample_recovered_tokens(
    cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, q, vocab_size
):
    import torch
    batch_size = cu_num_draft_tokens.shape[0]
    out = torch.empty_like(draft_token_ids)
    start = 0
    for req in range(batch_size):
        end = cu_num_draft_tokens[req].item()
        for idx in range(start, end):
            if draft_probs is None:
                prob = target_probs[idx].clone()
                prob[draft_token_ids[idx]] = 0
            else:
                prob = torch.maximum(target_probs[idx] - draft_probs[idx], torch.zeros_like(target_probs[idx]))
            out[idx] = torch.argmax(prob / q[req]).to(out.dtype)
        start = end
    return out

def run_correctness():
    import torch
    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    device = "cuda"
    for i, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_per_req = [(max_draft - (j % 2)) for j in range(batch_size)]
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            result = mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            ref = reference_sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, vocab_size)
            if not torch.equal(result, ref):
                return False, f"Shape {i+1}: mismatch with draft_probs path"

            # Also test NO_DRAFT_PROBS path.
            result_no_draft = mod.sample_recovered_tokens(cu, draft_ids, None, target_probs, q, max_draft, vocab_size)
            ref_no_draft = reference_sample_recovered_tokens(cu, draft_ids, None, target_probs, q, vocab_size)
            if not torch.equal(result_no_draft, ref_no_draft):
                return False, f"Shape {i+1}: mismatch with NO_DRAFT_PROBS path"
            torch.cuda.synchronize()
            assert result.shape == (total,), f"Wrong shape: {result.shape}"
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch_size, max_draft, vocab_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            num_per_req = [max_draft] * batch_size
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32), dim=0).to(device)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            for _ in range(WARMUP_ITERATIONS): mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                start_events[j].record()
                mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
                end_events[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch_size,
                    "max_draft": max_draft,
                    "vocab": vocab_size
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
