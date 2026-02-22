#!/usr/bin/env python3
"""Task runner for triton2triton/triton_convert_req_to_global_index"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_convert_req_to_global_index"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_convert_req_to_global_index.py")

# Test configs: (num_tokens, num_requests, max_blocks_per_req, num_topk_tokens, block_size, block_n)
TEST_SHAPES = [
    (16, 4, 8, 128, 64, 128),
    (32, 8, 16, 256, 64, 128),
    (64, 16, 32, 512, 64, 128),
    (128, 32, 16, 256, 32, 128),
    (256, 64, 32, 512, 64, 128),
]
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_data(num_tokens, num_requests, max_blocks_per_req, num_topk_tokens,
                   block_size, device="cuda"):
    import torch
    # req_id: each token assigned to a random request
    req_id = torch.randint(0, num_requests, (num_tokens,), device=device, dtype=torch.int32)

    # block_table: random physical block indices
    block_table = torch.randint(0, 1000, (num_requests, max_blocks_per_req),
                                device=device, dtype=torch.int32)

    # token_indices: random token indices, some set to -1
    max_token_idx = max_blocks_per_req * block_size
    token_indices = torch.randint(0, max_token_idx, (num_tokens, num_topk_tokens),
                                  device=device, dtype=torch.int32)
    # Set ~10% to -1 (invalid)
    mask = torch.rand(num_tokens, num_topk_tokens, device=device) < 0.1
    token_indices[mask] = -1

    return req_id, block_table, token_indices


def reference_convert(req_id, block_table, token_indices, block_size):
    """CPU reference implementation."""
    import torch
    num_tokens = req_id.shape[0]
    num_topk = token_indices.shape[1]
    max_blocks = block_table.shape[1]
    out = torch.empty_like(token_indices)

    for i in range(num_tokens):
        r = req_id[i].item()
        for j in range(num_topk):
            tok = token_indices[i, j].item()
            if tok < 0:
                out[i, j] = -1
                continue
            block_id = tok // block_size
            inblock_off = tok % block_size
            if block_id < 0 or block_id >= max_blocks:
                out[i, j] = -1
            else:
                base = block_table[r, block_id].item()
                out[i, j] = base * block_size + inblock_off

    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "convert_req_to_global_index"), "Missing convert_req_to_global_index"
        assert hasattr(mod, "_convert_req_index_to_global_index_kernel"), \
            "Missing _convert_req_index_to_global_index_kernel"
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

    for i, (nt, nr, mbpr, ntk, bs, bn) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            req_id, block_table, token_indices = make_test_data(nt, nr, mbpr, ntk, bs, device)

            result = mod.convert_req_to_global_index(
                req_id, block_table, token_indices,
                BLOCK_SIZE=bs, BLOCK_N=bn,
            )
            torch.cuda.synchronize()

            ref = reference_convert(req_id, block_table, token_indices, bs)

            if not torch.equal(result, ref):
                diff_mask = result != ref
                num_diffs = diff_mask.sum().item()
                return False, f"Shape {i+1}: {num_diffs} mismatched entries"

            # Also test with valid counts
            result2, counts = mod.convert_req_to_global_index(
                req_id, block_table, token_indices,
                BLOCK_SIZE=bs, BLOCK_N=bn, return_valid_counts=True,
            )
            torch.cuda.synchronize()

            ref_counts = (ref != -1).sum(dim=1).to(torch.int32)
            if not torch.equal(counts, ref_counts):
                return False, f"Shape {i+1}: valid counts mismatch"

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
    nt, nr, mbpr, ntk, bs, bn = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    req_id, block_table, token_indices = make_test_data(nt, nr, mbpr, ntk, bs, device)

    for _ in range(5):
        mod.convert_req_to_global_index(
            req_id, block_table, token_indices,
            BLOCK_SIZE=bs, BLOCK_N=bn,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.convert_req_to_global_index(
            req_id, block_table, token_indices,
            BLOCK_SIZE=bs, BLOCK_N=bn,
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
        shape = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "num_tokens": shape[0], "num_requests": shape[1],
                "max_blocks_per_req": shape[2], "num_topk_tokens": shape[3],
                "block_size": shape[4],
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
