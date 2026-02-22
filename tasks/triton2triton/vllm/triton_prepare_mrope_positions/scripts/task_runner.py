#!/usr/bin/env python3
"""Task runner for triton2triton/triton_prepare_mrope_positions"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_prepare_mrope_positions"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_prepare_mrope_positions.py")

# (num_reqs, query_len, max_model_len, is_prefill)
TEST_SHAPES = [
    (4, 32, 128, True),
    (8, 64, 256, True),
    (16, 128, 512, False),
    (32, 256, 1024, True),
    (64, 16, 2048, False),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_prepare_mrope(mrope_positions, prefill_mrope_positions, max_model_len,
                             prefill_mrope_delta, idx_mapping, query_start_loc,
                             prefill_lens, num_computed_tokens):
    import torch
    mrope_positions = mrope_positions.clone()
    num_reqs = idx_mapping.shape[0]

    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        prefill_len = prefill_lens[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        is_prefill = num_computed < prefill_len

        qstart = query_start_loc[b].item()
        qend = query_start_loc[b + 1].item()
        qlen = qend - qstart
        delta = prefill_mrope_delta[req_state_idx].item()

        for k in range(qlen):
            orig_pos = num_computed + k
            for j in range(3):
                if is_prefill:
                    pos = prefill_mrope_positions[
                        req_state_idx * 3 + j, orig_pos
                    ].item()
                else:
                    pos = orig_pos + delta
                mrope_positions[j, qstart + k] = pos

    return mrope_positions


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "prepare_mrope_positions")
        assert hasattr(mod, "_prepare_mrope_positions_kernel")
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
    for i, (num_reqs, query_len, max_model_len, is_prefill) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            max_num_reqs = num_reqs + 8
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())

            # Set up prefill/decode scenario
            if is_prefill:
                prefill_lens = torch.full((max_num_reqs,), max_model_len, dtype=torch.int32, device=device)
                num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            else:
                prefill_lens = torch.full((max_num_reqs,), 10, dtype=torch.int32, device=device)
                num_computed_tokens = torch.full((max_num_reqs,), 50, dtype=torch.int32, device=device)

            prefill_mrope_positions = torch.randint(
                0, max_model_len, (max_num_reqs * 3, max_model_len),
                dtype=torch.int32, device=device
            )
            prefill_mrope_delta = torch.randint(
                -10, 10, (max_num_reqs,), dtype=torch.int32, device=device
            )
            mrope_positions = torch.zeros(3, total_tokens + 1, dtype=torch.int64, device=device)

            mod.prepare_mrope_positions(
                mrope_positions, prefill_mrope_positions, max_model_len,
                prefill_mrope_delta, idx_mapping, query_start_loc,
                prefill_lens, num_computed_tokens,
            )
            torch.cuda.synchronize()

            ref = reference_prepare_mrope(
                torch.zeros(3, total_tokens + 1, dtype=torch.int64),
                prefill_mrope_positions.cpu(), max_model_len,
                prefill_mrope_delta.cpu(), idx_mapping.cpu(), query_start_loc.cpu(),
                prefill_lens.cpu(), num_computed_tokens.cpu(),
            )

            if not torch.equal(mrope_positions.cpu()[:, :total_tokens], ref[:, :total_tokens]):
                diff = (mrope_positions.cpu()[:, :total_tokens] != ref[:, :total_tokens])
                first_diff = diff.nonzero()
                if len(first_diff) > 0:
                    fd = first_diff[0]
                    return False, f"Shape {i+1}: mismatch at [{fd[0]},{fd[1]}] got {mrope_positions.cpu()[fd[0],fd[1]].item()} expected {ref[fd[0],fd[1]].item()}"
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
    num_reqs, query_len, max_model_len, is_prefill = TEST_SHAPES[PERF_SHAPE_IDX]
    max_num_reqs = num_reqs + 8

    torch.manual_seed(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        query_start_loc[r + 1] = query_start_loc[r] + query_len
    total_tokens = int(query_start_loc[-1].item())

    prefill_lens = torch.full((max_num_reqs,), max_model_len, dtype=torch.int32, device=device)
    num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    prefill_mrope_positions = torch.randint(
        0, max_model_len, (max_num_reqs * 3, max_model_len),
        dtype=torch.int32, device=device
    )
    prefill_mrope_delta = torch.randint(-10, 10, (max_num_reqs,), dtype=torch.int32, device=device)
    mrope_positions = torch.zeros(3, total_tokens + 1, dtype=torch.int64, device=device)

    for _ in range(5):
        mod.prepare_mrope_positions(
            mrope_positions, prefill_mrope_positions, max_model_len,
            prefill_mrope_delta, idx_mapping, query_start_loc,
            prefill_lens, num_computed_tokens,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.prepare_mrope_positions(
            mrope_positions, prefill_mrope_positions, max_model_len,
            prefill_mrope_delta, idx_mapping, query_start_loc,
            prefill_lens, num_computed_tokens,
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
