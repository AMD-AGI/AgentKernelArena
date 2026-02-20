#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prepare_pos_seq_lens"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_prepare_pos_seq_lens"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_pos_seq_lens.py")

# (num_reqs, max_num_reqs, query_len, num_computed_base)
TEST_SHAPES = [
    (4, 32, 16, 0),
    (8, 64, 32, 10),
    (16, 128, 64, 50),
    (32, 256, 128, 100),
    (64, 512, 256, 200),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, max_num_reqs):
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    pos = torch.zeros(total_tokens, dtype=torch.int64, device="cpu")
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu")

    for r in range(num_reqs):
        req_state_idx = idx_mapping[r].item()
        nc = num_computed_tokens[req_state_idx].item()
        start = query_start_loc[r].item()
        end = query_start_loc[r + 1].item()
        qlen = end - start
        seq_lens[r] = nc + qlen
        for k in range(qlen):
            pos[start + k] = nc + k

    return pos, seq_lens


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_pos_seq_lens"), "Missing prepare_pos_seq_lens"
        assert hasattr(mod, "_prepare_pos_seq_lens_kernel"), "Missing _prepare_pos_seq_lens_kernel"
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
    for i, (num_reqs, max_num_reqs, query_len, nc_base) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())
            num_computed_tokens = torch.full((max_num_reqs,), nc_base, dtype=torch.int32, device=device)
            pos = torch.zeros(total_tokens, dtype=torch.int64, device=device)
            seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

            mod.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
            torch.cuda.synchronize()

            ref_pos, ref_seq = reference_prepare_pos_seq_lens(
                idx_mapping.cpu(), query_start_loc.cpu(), num_computed_tokens.cpu(), max_num_reqs
            )

            if not torch.equal(pos.cpu(), ref_pos):
                return False, f"Shape {i+1}: pos mismatch"
            if not torch.equal(seq_lens.cpu(), ref_seq):
                return False, f"Shape {i+1}: seq_lens mismatch"
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
    num_reqs, max_num_reqs, query_len, nc_base = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + query_len
    total_tokens = int(query_start_loc[-1].item())
    num_computed_tokens = torch.full((max_num_reqs,), nc_base, dtype=torch.int32, device=device)
    pos = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    for _ in range(5):
        mod.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
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
