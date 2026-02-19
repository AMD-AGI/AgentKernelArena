#!/usr/bin/env python3
"""Task runner for triton2triton/triton_post_update"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_post_update"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_post_update.py")

# (num_reqs, vocab_size, max_model_len, num_spec_steps)
TEST_SHAPES = [
    (4, 256, 128, 1),
    (8, 512, 256, 2),
    (16, 1024, 512, 3),
    (32, 2048, 1024, 4),
    (64, 4096, 2048, 3),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_post_update(idx_mapping, num_computed_tokens, last_sampled_tokens,
                           output_bin_counts, sampled_tokens, num_sampled,
                           num_rejected, query_start_loc, all_token_ids, total_len):
    import torch
    num_reqs = idx_mapping.shape[0]
    num_computed_tokens = num_computed_tokens.clone()
    last_sampled_tokens = last_sampled_tokens.clone()
    output_bin_counts = output_bin_counts.clone()
    all_token_ids = all_token_ids.clone()
    total_len = total_len.clone()

    for r in range(num_reqs):
        req_state_idx = idx_mapping[r].item()
        tlen = total_len[req_state_idx].item()
        ns = num_sampled[r].item()

        if ns > 0:
            tok = sampled_tokens[r, ns - 1].item()
            last_sampled_tokens[req_state_idx] = tok
            total_len[req_state_idx] = tlen + ns

        for j in range(ns):
            tok = sampled_tokens[r, j].item()
            output_bin_counts[req_state_idx, tok] += 1
            all_token_ids[req_state_idx, tlen + j] = tok

        qstart = query_start_loc[r].item()
        qend = query_start_loc[r + 1].item()
        qlen = qend - qstart
        nr = num_rejected[r].item()
        nc = num_computed_tokens[req_state_idx].item()
        num_computed_tokens[req_state_idx] = nc + qlen - nr

    return num_computed_tokens, last_sampled_tokens, output_bin_counts, all_token_ids, total_len


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "post_update")
        assert hasattr(mod, "_post_update_kernel")
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
    for i, (num_reqs, vocab_size, max_model_len, num_spec) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 8
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

            query_len = num_spec + 1
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len

            num_computed_tokens = torch.randint(10, 50, (max_num_reqs,), dtype=torch.int32, device=device)
            last_sampled_tokens = torch.randint(0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=device)
            output_bin_counts = torch.zeros((max_num_reqs, vocab_size), dtype=torch.int32, device=device)
            sampled_tokens = torch.randint(0, vocab_size, (num_reqs, num_spec + 1), dtype=torch.int32, device=device)
            num_sampled = torch.randint(1, num_spec + 2, (num_reqs,), dtype=torch.int32, device=device)
            num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                nr = query_len - num_sampled[r].item()
                num_rejected[r] = max(0, nr)
            all_token_ids = torch.zeros((max_num_reqs, max_model_len), dtype=torch.int32, device=device)
            total_len = torch.randint(10, 50, (max_num_reqs,), dtype=torch.int32, device=device)

            # Clone for GPU
            nct_g = num_computed_tokens.clone()
            lst_g = last_sampled_tokens.clone()
            obc_g = output_bin_counts.clone()
            ati_g = all_token_ids.clone()
            tl_g = total_len.clone()

            mod.post_update(
                idx_mapping, nct_g, lst_g, obc_g, sampled_tokens,
                num_sampled, num_rejected, query_start_loc, ati_g, tl_g,
            )
            torch.cuda.synchronize()

            ref = reference_post_update(
                idx_mapping.cpu(), num_computed_tokens.cpu(), last_sampled_tokens.cpu(),
                output_bin_counts.cpu(), sampled_tokens.cpu(), num_sampled.cpu(),
                num_rejected.cpu(), query_start_loc.cpu(), all_token_ids.cpu(), total_len.cpu(),
            )

            if not torch.equal(nct_g.cpu(), ref[0]):
                return False, f"Shape {i+1}: num_computed_tokens mismatch"
            if not torch.equal(lst_g.cpu(), ref[1]):
                return False, f"Shape {i+1}: last_sampled_tokens mismatch"
            if not torch.equal(obc_g.cpu(), ref[2]):
                return False, f"Shape {i+1}: output_bin_counts mismatch"
            if not torch.equal(ati_g.cpu(), ref[3]):
                return False, f"Shape {i+1}: all_token_ids mismatch"
            if not torch.equal(tl_g.cpu(), ref[4]):
                return False, f"Shape {i+1}: total_len mismatch"

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
    num_reqs, vocab_size, max_model_len, num_spec = TEST_SHAPES[PERF_SHAPE_IDX]
    max_num_reqs = num_reqs + 8
    query_len = num_spec + 1

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + query_len

    def make_inputs():
        nct = torch.randint(10, 50, (max_num_reqs,), dtype=torch.int32, device=device)
        lst = torch.randint(0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=device)
        obc = torch.zeros((max_num_reqs, vocab_size), dtype=torch.int32, device=device)
        st = torch.randint(0, vocab_size, (num_reqs, num_spec + 1), dtype=torch.int32, device=device)
        ns = torch.full((num_reqs,), num_spec + 1, dtype=torch.int32, device=device)
        nr = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        ati = torch.zeros((max_num_reqs, max_model_len), dtype=torch.int32, device=device)
        tlen = torch.randint(10, 50, (max_num_reqs,), dtype=torch.int32, device=device)
        return nct, lst, obc, st, ns, nr, ati, tlen

    for _ in range(5):
        nct, lst, obc, st, ns, nr, ati, tlen = make_inputs()
        mod.post_update(idx_mapping, nct, lst, obc, st, ns, nr, query_start_loc, ati, tlen)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        nct, lst, obc, st, ns, nr, ati, tlen = make_inputs()
        start_events[j].record()
        mod.post_update(idx_mapping, nct, lst, obc, st, ns, nr, query_start_loc, ati, tlen)
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
