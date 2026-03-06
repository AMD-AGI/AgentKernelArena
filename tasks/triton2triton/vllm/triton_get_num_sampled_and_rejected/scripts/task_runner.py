#!/usr/bin/env python3
"""Task runner for triton2triton/triton_get_num_sampled_and_rejected"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_get_num_sampled_and_rejected"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_get_num_sampled_and_rejected.py")

# (num_reqs, num_spec_steps)
TEST_SHAPES = [
    (4, 2),
    (8, 3),
    (16, 4),
    (32, 5),
    (64, 4),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_get_num_sampled_and_rejected(num_sampled, seq_lens, cu_num_logits,
                                            idx_mapping, prefill_len):
    import torch
    num_reqs = idx_mapping.shape[0]
    num_sampled_out = num_sampled.clone()
    num_rejected_out = torch.empty_like(num_sampled)

    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        sl = seq_lens[b].item()
        pl = prefill_len[req_state_idx].item()
        is_chunked = sl < pl

        ns = num_sampled[b].item()
        if is_chunked:
            ns = 0
        num_sampled_out[b] = ns

        logits_start = cu_num_logits[b].item()
        logits_end = cu_num_logits[b + 1].item()
        num_logits = logits_end - logits_start

        nr = num_logits - ns
        if is_chunked:
            nr = 0
        num_rejected_out[b] = nr

    return num_sampled_out, num_rejected_out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "get_num_sampled_and_rejected")
        assert hasattr(mod, "_get_num_sampled_and_rejected_kernel")
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
    for i, (num_reqs, num_spec) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 8
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

            # Mix of decode and chunked prefill requests
            seq_lens = torch.randint(50, 200, (num_reqs,), dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), 100, dtype=torch.int32, device=device)

            logits_per_req = num_spec + 1
            cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                cu_num_logits[r + 1] = cu_num_logits[r] + logits_per_req

            num_sampled = torch.randint(1, logits_per_req + 1, (num_reqs,), dtype=torch.int32, device=device)
            num_sampled_copy = num_sampled.clone()

            ns_out, nr_out = mod.get_num_sampled_and_rejected(
                num_sampled_copy, seq_lens, cu_num_logits, idx_mapping, prefill_len,
            )
            torch.cuda.synchronize()

            ref_ns, ref_nr = reference_get_num_sampled_and_rejected(
                num_sampled.cpu(), seq_lens.cpu(), cu_num_logits.cpu(),
                idx_mapping.cpu(), prefill_len.cpu(),
            )

            if not torch.equal(ns_out.cpu(), ref_ns):
                return False, f"Shape {i+1}: num_sampled mismatch"
            if not torch.equal(nr_out.cpu(), ref_nr):
                return False, f"Shape {i+1}: num_rejected mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_reqs, num_spec) in enumerate(TEST_SHAPES):
        try:
            max_num_reqs = num_reqs + 8
            torch.manual_seed(42 + test_idx)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            seq_lens = torch.randint(100, 200, (num_reqs,), dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), 50, dtype=torch.int32, device=device)
            logits_per_req = num_spec + 1
            cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                cu_num_logits[r + 1] = cu_num_logits[r] + logits_per_req
            num_sampled = torch.randint(1, logits_per_req + 1, (num_reqs,), dtype=torch.int32, device=device)

            for _ in range(WARMUP_ITERATIONS):
                mod.get_num_sampled_and_rejected(num_sampled.clone(), seq_lens, cu_num_logits, idx_mapping, prefill_len)
            torch.cuda.synchronize()

            n_iter = BENCHMARK_ITERATIONS
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                start_events[j].record()
                mod.get_num_sampled_and_rejected(num_sampled.clone(), seq_lens, cu_num_logits, idx_mapping, prefill_len)
                end_events[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "num_reqs": num_reqs,
                    "num_spec_steps": num_spec
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "num_spec_steps": num_spec
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
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
