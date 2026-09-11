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
TARGETED_CORRECTNESS_CASES = [
    {
        "name": "mixed_mapped_boundary_lengths",
        "max_model_len": 64,
        "idx_mapping": [6, 2, 7, 0, 5, 3],
        "query_lens": [0, 1, 3, 1, 0, 7],
        # Per-batch values are scattered through idx_mapping below. These cover
        # zero, the final prefill position, and the exact decode boundary.
        "prefill_lens": [5, 9, 12, 1, 4, 11],
        "num_computed_tokens": [0, 8, 12, 1, 4, 4],
    },
    {
        "name": "mixed_multi_tile_requests",
        "max_model_len": 4096,
        "idx_mapping": [3, 0],
        "query_lens": [1025, 2051],
        # Both requests cross the kernel's 1024-element tile. The prefill
        # request ends at the last valid lookup entry; decode starts exactly at
        # its prefill length and remains within max_model_len.
        "prefill_lens": [4096, 2045],
        "num_computed_tokens": [3071, 2045],
    },
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

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


def check_mrope_case(mod, case_name, mrope_positions, prefill_mrope_positions,
                     max_model_len, prefill_mrope_delta, idx_mapping,
                     query_start_loc, prefill_lens, num_computed_tokens):
    import torch

    ref = reference_prepare_mrope(
        mrope_positions.cpu(), prefill_mrope_positions.cpu(), max_model_len,
        prefill_mrope_delta.cpu(), idx_mapping.cpu(), query_start_loc.cpu(),
        prefill_lens.cpu(), num_computed_tokens.cpu(),
    )
    mod.prepare_mrope_positions(
        mrope_positions, prefill_mrope_positions, max_model_len,
        prefill_mrope_delta, idx_mapping, query_start_loc, prefill_lens,
        num_computed_tokens,
    )
    torch.cuda.synchronize()

    actual = mrope_positions.cpu()
    if torch.equal(actual, ref):
        return True, None

    first_diff = (actual != ref).nonzero()[0]
    dim = first_diff[0].item()
    token = first_diff[1].item()
    return False, (
        f"{case_name}: mismatch at [{dim},{token}] "
        f"got {actual[dim, token].item()} expected {ref[dim, token].item()}"
    )


def run_targeted_correctness_case(mod, case, seed):
    import torch

    device = "cuda"
    idx_mapping_values = case["idx_mapping"]
    query_lens = case["query_lens"]
    prefill_lens_by_batch = case["prefill_lens"]
    num_computed_by_batch = case["num_computed_tokens"]
    max_model_len = case["max_model_len"]
    num_reqs = len(idx_mapping_values)

    if not (
        len(query_lens) == len(prefill_lens_by_batch)
        == len(num_computed_by_batch) == num_reqs
    ):
        raise ValueError("targeted case arrays must have one entry per request")
    if len(set(idx_mapping_values)) != num_reqs:
        raise ValueError("targeted cases require unique request-state mappings")

    max_num_reqs = max(num_reqs + 8, max(idx_mapping_values) + 1)
    idx_mapping = torch.tensor(
        idx_mapping_values, dtype=torch.int32, device=device
    )
    query_starts = [0]
    for query_len in query_lens:
        if query_len < 0:
            raise ValueError("query lengths must be nonnegative")
        query_starts.append(query_starts[-1] + query_len)
    query_start_loc = torch.tensor(
        query_starts, dtype=torch.int32, device=device
    )

    prefill_lens = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=device
    )
    num_computed_tokens = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=device
    )
    for batch_idx, req_state_idx in enumerate(idx_mapping_values):
        prefill_len = prefill_lens_by_batch[batch_idx]
        num_computed = num_computed_by_batch[batch_idx]
        query_len = query_lens[batch_idx]
        if not (0 <= prefill_len <= max_model_len):
            raise ValueError("prefill length is outside max_model_len")
        if not (0 <= num_computed <= max_model_len):
            raise ValueError("num_computed is outside max_model_len")
        if num_computed + query_len > max_model_len:
            raise ValueError("request tokens exceed max_model_len")
        if num_computed < prefill_len and num_computed + query_len > prefill_len:
            raise ValueError("prefill query extends beyond its prefill length")
        prefill_lens[req_state_idx] = prefill_len
        num_computed_tokens[req_state_idx] = num_computed

    torch.manual_seed(seed)
    prefill_mrope_positions = torch.randint(
        0, max_model_len, (max_num_reqs * 3, max_model_len),
        dtype=torch.int32, device=device,
    )
    prefill_mrope_delta = torch.randint(
        -10, 10, (max_num_reqs,), dtype=torch.int32, device=device
    )
    # The int64-only sentinel also verifies that empty requests and the output
    # guard element remain untouched.
    mrope_positions = torch.full(
        (3, query_starts[-1] + 1), -(1 << 40),
        dtype=torch.int64, device=device,
    )
    return check_mrope_case(
        mod, case["name"], mrope_positions, prefill_mrope_positions,
        max_model_len, prefill_mrope_delta, idx_mapping, query_start_loc,
        prefill_lens, num_computed_tokens,
    )


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

            ok, err = check_mrope_case(
                mod, f"Shape {i + 1}", mrope_positions,
                prefill_mrope_positions, max_model_len, prefill_mrope_delta,
                idx_mapping, query_start_loc, prefill_lens,
                num_computed_tokens,
            )
            if not ok:
                return False, err
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(TARGETED_CORRECTNESS_CASES):
        try:
            ok, err = run_targeted_correctness_case(mod, case, 100 + i)
            if not ok:
                return False, err
        except Exception as e:
            return False, f"Case {case['name']}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_reqs, query_len, max_model_len, is_prefill) in enumerate(TEST_SHAPES):
        try:
            max_num_reqs = num_reqs + 8

            torch.manual_seed(0)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            for r in range(num_reqs):
                query_start_loc[r + 1] = query_start_loc[r] + query_len
            total_tokens = int(query_start_loc[-1].item())

            if is_prefill:
                prefill_lens = torch.full(
                    (max_num_reqs,), max_model_len,
                    dtype=torch.int32, device=device,
                )
                num_computed_tokens = torch.zeros(
                    max_num_reqs, dtype=torch.int32, device=device
                )
            else:
                prefill_lens = torch.full(
                    (max_num_reqs,), 10, dtype=torch.int32, device=device
                )
                num_computed_tokens = torch.full(
                    (max_num_reqs,), 50, dtype=torch.int32, device=device
                )
            prefill_mrope_positions = torch.randint(
                0, max_model_len, (max_num_reqs * 3, max_model_len),
                dtype=torch.int32, device=device
            )
            prefill_mrope_delta = torch.randint(-10, 10, (max_num_reqs,), dtype=torch.int32, device=device)
            mrope_positions = torch.zeros(3, total_tokens + 1, dtype=torch.int64, device=device)

            def _bench_fn():
                mod.prepare_mrope_positions(
                    mrope_positions, prefill_mrope_positions, max_model_len,
                    prefill_mrope_delta, idx_mapping, query_start_loc,
                    prefill_lens, num_computed_tokens,
                )
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "num_reqs": num_reqs,
                    "query_len": query_len,
                    "max_model_len": max_model_len,
                    "is_prefill": is_prefill
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "query_len": query_len,
                    "max_model_len": max_model_len,
                    "is_prefill": is_prefill
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(TARGETED_CORRECTNESS_CASES),
        }
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
