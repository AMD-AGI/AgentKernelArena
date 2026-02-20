#!/usr/bin/env python3
"""Task runner for triton2triton/triton_compute_slot_mappings"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_compute_slot_mappings"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_compute_slot_mappings.py")

# (num_reqs, query_len, block_size, max_num_blocks)
TEST_SHAPES = [
    (4, 32, 16, 64),
    (8, 64, 16, 128),
    (16, 128, 32, 128),
    (32, 256, 16, 256),
    (64, 512, 32, 256),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_compute_slot_mappings(idx_mapping, query_start_loc, positions,
                                      block_table, block_size):
    import torch
    num_reqs = idx_mapping.shape[0]
    num_tokens = positions.shape[0]
    slot_mappings = torch.full((num_tokens,), -1, dtype=torch.int64)

    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        start = query_start_loc[b].item()
        end = query_start_loc[b + 1].item()
        for t in range(start, end):
            p = positions[t].item()
            block_idx = p // block_size
            block_off = p % block_size
            block_num = block_table[req_idx, block_idx].item()
            slot_mappings[t] = block_num * block_size + block_off

    return slot_mappings


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "compute_slot_mappings")
        assert hasattr(mod, "_compute_slot_mappings_kernel")
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
    for i, (num_reqs, query_len, block_size, max_num_blocks) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 16
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())

            # Positions: each request starts at some offset
            positions = torch.zeros(total_tokens, dtype=torch.int64, device=device)
            for r in range(num_reqs):
                start = query_start_loc[r].item()
                for k in range(query_len):
                    positions[start + k] = k  # positions 0..query_len-1

            block_table = torch.randint(0, 10000, (max_num_reqs, max_num_blocks),
                                         dtype=torch.int32, device=device)
            max_num_tokens = total_tokens + 64

            result = mod.compute_slot_mappings(
                idx_mapping, query_start_loc, positions, block_table,
                block_size, max_num_tokens,
            )
            torch.cuda.synchronize()

            ref = reference_compute_slot_mappings(
                idx_mapping.cpu(), query_start_loc.cpu(), positions.cpu(),
                block_table.cpu(), block_size,
            )

            if not torch.equal(result.cpu(), ref):
                diff_mask = result.cpu() != ref
                first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item()
                return False, f"Shape {i+1}: mismatch at index {first_diff}, got {result.cpu()[first_diff].item()} expected {ref[first_diff].item()}"
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
    num_reqs, query_len, block_size, max_num_blocks = TEST_SHAPES[PERF_SHAPE_IDX]
    max_num_reqs = num_reqs + 16

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + query_len
    total_tokens = int(query_start_loc[-1].item())
    positions = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    for r in range(num_reqs):
        start = query_start_loc[r].item()
        for k in range(query_len):
            positions[start + k] = k
    block_table = torch.randint(0, 10000, (max_num_reqs, max_num_blocks),
                                 dtype=torch.int32, device=device)
    max_num_tokens = total_tokens + 64

    for _ in range(5):
        mod.compute_slot_mappings(idx_mapping, query_start_loc, positions, block_table, block_size, max_num_tokens)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.compute_slot_mappings(idx_mapping, query_start_loc, positions, block_table, block_size, max_num_tokens)
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
