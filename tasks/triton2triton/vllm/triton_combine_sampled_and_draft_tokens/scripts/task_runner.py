#!/usr/bin/env python3
"""Task runner for triton2triton/triton_combine_sampled_and_draft_tokens"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_combine_sampled_and_draft_tokens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_combine_sampled_and_draft_tokens.py")

# (num_reqs, num_speculative_steps, seq_len_base)
TEST_SHAPES = [
    (4, 2, 64),
    (8, 4, 128),
    (16, 3, 256),
    (32, 5, 512),
    (64, 4, 1024),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_combine(input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
                       seq_lens, prefill_len, draft_tokens, cu_num_logits):
    import torch
    num_reqs = seq_lens.shape[0]
    total_logits = int(cu_num_logits[-1].item())
    logits_indices = torch.empty(total_logits, dtype=torch.int64)
    input_ids_out = input_ids.clone()

    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        cu_start = cu_num_logits[b].item()
        cu_end = cu_num_logits[b + 1].item()
        num_logits = cu_end - cu_start
        num_draft = num_logits - 1

        query_end = query_start_loc[b + 1].item()
        logits_start = query_end - num_logits
        for k in range(num_logits):
            logits_indices[cu_start + k] = logits_start + k

        sl = seq_lens[b].item()
        pl = prefill_len[req_state_idx].item()
        if sl <= pl:
            continue

        last_tok = last_sampled_tokens[req_state_idx].item()
        input_ids_out[query_end - num_logits] = last_tok

        if num_draft > 0:
            for k in range(num_draft):
                input_ids_out[query_end - num_draft + k] = draft_tokens[req_state_idx, k]

    return input_ids_out, logits_indices


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "combine_sampled_and_draft_tokens")
        assert hasattr(mod, "_combine_sampled_and_draft_tokens_kernel")
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
    for i, (num_reqs, num_spec, seq_base) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 8
            # Each req has 1 + num_spec tokens (1 sampled + num_spec draft)
            tokens_per_req = 1 + num_spec
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + tokens_per_req
            total_tokens = int(query_start_loc[-1].item())

            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            input_ids = torch.randint(0, 32000, (total_tokens,), dtype=torch.int32, device=device)
            last_sampled_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int32, device=device)
            # All decode (seq_len > prefill_len)
            seq_lens = torch.full((num_reqs,), seq_base + tokens_per_req, dtype=torch.int32, device=device)
            prefill_len = torch.full((max_num_reqs,), seq_base, dtype=torch.int32, device=device)
            draft_tokens = torch.randint(0, 32000, (max_num_reqs, num_spec), dtype=torch.int32, device=device)
            # Each req contributes 1 + num_spec logits
            cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                cu_num_logits[r + 1] = cu_num_logits[r] + tokens_per_req
            total_logits = int(cu_num_logits[-1].item())

            input_ids_copy = input_ids.clone()
            logits_indices = mod.combine_sampled_and_draft_tokens(
                input_ids_copy, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, draft_tokens, cu_num_logits, total_logits,
            )
            torch.cuda.synchronize()

            ref_ids, ref_logits = reference_combine(
                input_ids.cpu(), idx_mapping.cpu(), last_sampled_tokens.cpu(),
                query_start_loc.cpu(), seq_lens.cpu(), prefill_len.cpu(),
                draft_tokens.cpu(), cu_num_logits.cpu(),
            )

            if not torch.equal(input_ids_copy.cpu(), ref_ids):
                return False, f"Shape {i+1}: input_ids mismatch"
            if not torch.equal(logits_indices.cpu(), ref_logits):
                return False, f"Shape {i+1}: logits_indices mismatch"
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
    num_reqs, num_spec, seq_base = TEST_SHAPES[PERF_SHAPE_IDX]
    max_num_reqs = num_reqs + 8
    tokens_per_req = 1 + num_spec
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + tokens_per_req
    total_tokens = int(query_start_loc[-1].item())

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    input_ids = torch.randint(0, 32000, (total_tokens,), dtype=torch.int32, device=device)
    last_sampled_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int32, device=device)
    seq_lens = torch.full((num_reqs,), seq_base + tokens_per_req, dtype=torch.int32, device=device)
    prefill_len = torch.full((max_num_reqs,), seq_base, dtype=torch.int32, device=device)
    draft_tokens = torch.randint(0, 32000, (max_num_reqs, num_spec), dtype=torch.int32, device=device)
    cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        cu_num_logits[r + 1] = cu_num_logits[r] + tokens_per_req
    total_logits = int(cu_num_logits[-1].item())

    for _ in range(5):
        mod.combine_sampled_and_draft_tokens(
            input_ids.clone(), idx_mapping, last_sampled_tokens, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits, total_logits,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.combine_sampled_and_draft_tokens(
            input_ids.clone(), idx_mapping, last_sampled_tokens, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits, total_logits,
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
