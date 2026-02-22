#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prepare_prefill_inputs"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_prepare_prefill_inputs"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_prefill_inputs.py")

# Test configurations: (num_reqs, max_seq_len, query_len)
TEST_SHAPES = [
    (4, 128, 32),
    (8, 256, 64),
    (16, 512, 128),
    (32, 1024, 256),
    (64, 2048, 512),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_prepare_prefill_inputs(
    idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens
):
    """CPU reference implementation."""
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    input_ids = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")
    next_prefill_tokens = torch.zeros(idx_mapping.max().item() + 1, dtype=torch.int32, device="cpu")

    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        plen = prefill_len[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        if num_computed >= plen:
            continue
        qstart = query_start_loc[b].item()
        qend = query_start_loc[b + 1].item()
        qlen = qend - qstart
        for k in range(qlen):
            input_ids[qstart + k] = all_token_ids[req_state_idx, num_computed + k]
        next_pos = num_computed + qlen
        if next_pos < plen:
            next_prefill_tokens[req_state_idx] = all_token_ids[req_state_idx, next_pos]

    return input_ids, next_prefill_tokens


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_prefill_inputs"), "Missing prepare_prefill_inputs"
        assert hasattr(mod, "_prepare_prefill_inputs_kernel"), "Missing _prepare_prefill_inputs_kernel"
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

    for i, (num_reqs, max_seq_len, query_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 16

            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len

            total_tokens = int(query_start_loc[-1].item())
            all_token_ids = torch.randint(0, 32000, (max_num_reqs, max_seq_len), dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), max_seq_len, dtype=torch.int32, device=device)
            num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            mod.prepare_prefill_inputs(
                input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                all_token_ids, prefill_len, num_computed_tokens,
            )
            torch.cuda.synchronize()

            ref_ids, ref_next = reference_prepare_prefill_inputs(
                idx_mapping.cpu(), query_start_loc.cpu(), all_token_ids.cpu(),
                prefill_len.cpu(), num_computed_tokens.cpu(),
            )

            if not torch.equal(input_ids.cpu(), ref_ids):
                return False, f"Shape {i+1}: input_ids mismatch"
            if not torch.equal(next_prefill_tokens.cpu()[:num_reqs], ref_next[:num_reqs]):
                return False, f"Shape {i+1}: next_prefill_tokens mismatch"

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
    num_reqs, max_seq_len, query_len = TEST_SHAPES[PERF_SHAPE_IDX]
    max_num_reqs = num_reqs + 16

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + query_len
    total_tokens = int(query_start_loc[-1].item())
    all_token_ids = torch.randint(0, 32000, (max_num_reqs, max_seq_len), dtype=torch.int32, device=device)
    prefill_len = torch.full((max_num_reqs,), max_seq_len, dtype=torch.int32, device=device)
    num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
    next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    for _ in range(5):
        mod.prepare_prefill_inputs(
            input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
            all_token_ids, prefill_len, num_computed_tokens,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.prepare_prefill_inputs(
            input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
            all_token_ids, prefill_len, num_computed_tokens,
        )
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
