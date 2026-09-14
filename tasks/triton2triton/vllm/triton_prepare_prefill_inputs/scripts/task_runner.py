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

# Correctness-only cases. Keep these separate from TEST_SHAPES so correctness
# coverage can grow without changing the scored performance workload.
CORRECTNESS_CASES = [
    {
        "name": "variable_lengths_sparse_mapping",
        "max_num_reqs": 16,
        "max_seq_len": 512,
        "idx_mapping": [9, 2, 14, 5, 11, 0],
        "query_lens": [0, 1, 7, 64, 255, 257],
        "prefill_lens": [6, 14, 37, 95, 297, 310],
        "num_computed_tokens": [5, 13, 29, 31, 41, 53],
    },
    {
        "name": "completed_prefill_early_return",
        "max_num_reqs": 12,
        "max_seq_len": 64,
        "idx_mapping": [8, 1, 10, 4],
        "query_lens": [3, 0, 5, 2],
        "prefill_lens": [12, 15, 19, 12],
        "num_computed_tokens": [12, 16, 20, 9],
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


def reference_prepare_prefill_inputs(
    idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens
):
    """CPU reference implementation."""
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    input_ids = torch.zeros(total_tokens, dtype=torch.int32, device="cpu")
    next_prefill_tokens = torch.zeros(all_token_ids.shape[0], dtype=torch.int32, device="cpu")

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
            if not torch.equal(next_prefill_tokens.cpu(), ref_next):
                return False, f"Shape {i+1}: next_prefill_tokens mismatch"

        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for case in CORRECTNESS_CASES:
        name = case["name"]
        try:
            max_num_reqs = case["max_num_reqs"]
            max_seq_len = case["max_seq_len"]
            idx_mapping = torch.tensor(
                case["idx_mapping"], dtype=torch.int32, device=device
            )
            query_lens = torch.tensor(
                case["query_lens"], dtype=torch.int32, device=device
            )
            query_start_loc = torch.zeros(
                len(case["query_lens"]) + 1, dtype=torch.int32, device=device
            )
            query_start_loc[1:] = torch.cumsum(query_lens, dim=0)

            all_token_ids = (
                torch.arange(
                    max_num_reqs * max_seq_len,
                    dtype=torch.int32,
                    device=device,
                ).reshape(max_num_reqs, max_seq_len)
                % 31999
            ) + 1
            prefill_len = torch.full(
                (max_num_reqs,), max_seq_len, dtype=torch.int32, device=device
            )
            num_computed_tokens = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            prefill_len[idx_mapping.long()] = torch.tensor(
                case["prefill_lens"], dtype=torch.int32, device=device
            )
            num_computed_tokens[idx_mapping.long()] = torch.tensor(
                case["num_computed_tokens"], dtype=torch.int32, device=device
            )

            total_tokens = int(query_start_loc[-1].item())
            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            next_prefill_tokens = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )

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
                return False, f"Case {name}: input_ids mismatch"
            if not torch.equal(next_prefill_tokens.cpu(), ref_next):
                return False, f"Case {name}: next_prefill_tokens mismatch"

        except Exception as e:
            return False, f"Case {name}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_reqs, max_seq_len, query_len) in enumerate(TEST_SHAPES):
        try:
            max_num_reqs = num_reqs + 16
            torch.manual_seed(42 + test_idx)
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

            def _bench_fn():
                mod.prepare_prefill_inputs(
                    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                    all_token_ids, prefill_len, num_computed_tokens,
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
                    "max_seq_len": max_seq_len,
                    "query_len": query_len
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "max_seq_len": max_seq_len,
                    "query_len": query_len
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
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_CASES),
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
