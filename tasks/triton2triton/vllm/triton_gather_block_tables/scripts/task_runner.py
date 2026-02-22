#!/usr/bin/env python3
"""Task runner for triton2triton/triton_gather_block_tables"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_gather_block_tables"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_gather_block_tables.py")

# (num_reqs, max_num_reqs, max_num_blocks)
TEST_SHAPES = [
    (4, 32, 16),
    (8, 64, 32),
    (16, 128, 64),
    (32, 256, 128),
    (64, 512, 256),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_gather(idx_mapping, src_block_table, num_blocks):
    import torch
    num_reqs = idx_mapping.shape[0]
    max_num_blocks = src_block_table.shape[1]
    dst = torch.zeros_like(src_block_table)
    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        nb = num_blocks[req_idx].item()
        dst[b, :nb] = src_block_table[req_idx, :nb]
    return dst[:num_reqs]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "gather_block_tables")
        assert hasattr(mod, "_gather_block_tables_kernel")
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
    for i, (num_reqs, max_num_reqs, max_num_blocks) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            idx_mapping = torch.randperm(max_num_reqs, device=device, dtype=torch.int32)[:num_reqs]
            src_block_table = torch.randint(0, 10000, (max_num_reqs, max_num_blocks),
                                             dtype=torch.int32, device=device)
            dst_block_table = torch.zeros_like(src_block_table)
            num_blocks = torch.randint(1, max_num_blocks + 1, (max_num_reqs,),
                                        dtype=torch.int32, device=device)

            result = mod.gather_block_tables(idx_mapping, src_block_table, dst_block_table, num_blocks)
            torch.cuda.synchronize()

            ref = reference_gather(idx_mapping.cpu(), src_block_table.cpu(), num_blocks.cpu())
            if not torch.equal(result.cpu(), ref):
                return False, f"Shape {i+1}: mismatch"
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
    num_reqs, max_num_reqs, max_num_blocks = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    src_block_table = torch.randint(0, 10000, (max_num_reqs, max_num_blocks),
                                     dtype=torch.int32, device=device)
    dst_block_table = torch.zeros_like(src_block_table)
    num_blocks = torch.full((max_num_reqs,), max_num_blocks, dtype=torch.int32, device=device)

    for _ in range(5):
        mod.gather_block_tables(idx_mapping, src_block_table, dst_block_table, num_blocks)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.gather_block_tables(idx_mapping, src_block_table, dst_block_table, num_blocks)
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
