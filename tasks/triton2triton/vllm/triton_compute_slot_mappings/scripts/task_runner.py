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

# Correctness-only coverage. Keep these separate from TEST_SHAPES so coverage
# changes do not alter the performance workload.
ADDITIONAL_CORRECTNESS_CASES = [
    {
        "name": "uneven_empty_permuted_int64",
        "query_lengths": [0, 1, 9, 17, 3],
        "idx_mapping": [6, 0, 4, 2, 7],
        "position_starts": [0, 7, 31, 4093, 1024],
        "block_size": 8,
        "max_num_reqs": 8,
        "max_num_blocks": 520,
        "block_table_row_padding": 5,
        "max_num_tokens": 47,
        "index_dtype": "int64",
        "block_table_dtype": "int64",
    },
    {
        "name": "multi_tile_data_and_padding",
        "query_lengths": [1025, 0, 6],
        "idx_mapping": [2, 5, 1],
        "position_starts": [8190, 0, 65533],
        "block_size": 64,
        "max_num_reqs": 6,
        "max_num_blocks": 1025,
        "block_table_row_padding": 7,
        "max_num_tokens": 3082,
        "index_dtype": "int32",
        "block_table_dtype": "int32",
    },
    {
        "name": "all_empty_multi_tile_padding",
        "query_lengths": [0, 0, 0],
        "idx_mapping": [2, 0, 1],
        "position_starts": [0, 0, 0],
        "block_size": 8,
        "max_num_reqs": 3,
        "max_num_blocks": 1,
        "block_table_row_padding": 3,
        "max_num_tokens": 2053,
        "index_dtype": "int32",
        "block_table_dtype": "int32",
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


def validate_slot_mappings(result, ref, case_name):
    """Validate the token prefix returned by compute_slot_mappings."""
    import torch

    if result.shape != ref.shape:
        return f"{case_name}: output shape {tuple(result.shape)} expected {tuple(ref.shape)}"
    if result.dtype != torch.int64:
        return f"{case_name}: output dtype {result.dtype} expected torch.int64"

    result_cpu = result.cpu()
    if not torch.equal(result_cpu, ref):
        diff_mask = result_cpu != ref
        first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item()
        return (
            f"{case_name}: mismatch at index {first_diff}, "
            f"got {result_cpu[first_diff].item()} expected {ref[first_diff].item()}"
        )

    return None


def validate_padding(mod, idx_mapping, query_start_loc, positions, block_table,
                     block_size, max_num_tokens, case_name):
    """Launch into a poisoned full buffer and directly validate all padding."""
    import torch

    num_tokens = positions.shape[0]
    padding_poison = 123456789
    full_result = torch.full(
        (max_num_tokens,), padding_poison, dtype=torch.int64,
        device=positions.device,
    )
    mod._compute_slot_mappings_kernel[(idx_mapping.shape[0] + 1,)](
        num_tokens,
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        full_result,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    padding = full_result[num_tokens:].cpu()
    if not torch.all(padding == -1).item():
        first_diff = (padding != -1).nonzero(as_tuple=True)[0][0].item()
        return (
            f"{case_name}: padding mismatch at index {num_tokens + first_diff}, "
            f"got {padding[first_diff].item()} expected -1"
        )

    return None


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

            error = validate_slot_mappings(result, ref, f"Shape {i+1}")
            if error:
                return False, error
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(ADDITIONAL_CORRECTNESS_CASES):
        case_name = case["name"]
        try:
            torch.manual_seed(1042 + i)
            query_lengths = case["query_lengths"]
            query_start_values = [0]
            for query_len in query_lengths:
                query_start_values.append(query_start_values[-1] + query_len)

            index_dtype = getattr(torch, case["index_dtype"])
            idx_mapping = torch.tensor(
                case["idx_mapping"], dtype=index_dtype, device=device
            )
            query_start_loc = torch.tensor(
                query_start_values, dtype=index_dtype, device=device
            )
            position_values = []
            for position_start, query_len in zip(
                case["position_starts"], query_lengths
            ):
                position_values.extend(range(position_start, position_start + query_len))
            positions = torch.tensor(
                position_values, dtype=torch.int64, device=device
            )

            # Slice a wider allocation to retain a nonstandard row stride while
            # keeping columns contiguous, as required by the kernel contract.
            table_storage = torch.randint(
                0,
                10000,
                (
                    case["max_num_reqs"],
                    case["max_num_blocks"] + case["block_table_row_padding"],
                ),
                dtype=getattr(torch, case["block_table_dtype"]),
                device=device,
            )
            block_table = table_storage[:, :case["max_num_blocks"]]

            result = mod.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                block_table,
                case["block_size"],
                case["max_num_tokens"],
            )
            torch.cuda.synchronize()

            ref = reference_compute_slot_mappings(
                idx_mapping.cpu(),
                query_start_loc.cpu(),
                positions.cpu(),
                block_table.cpu(),
                case["block_size"],
            )
            error = validate_slot_mappings(result, ref, case_name)
            if error:
                return False, error
            error = validate_padding(
                mod,
                idx_mapping,
                query_start_loc,
                positions,
                block_table,
                case["block_size"],
                case["max_num_tokens"],
                case_name,
            )
            if error:
                return False, error
        except Exception as e:
            return False, f"{case_name}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_reqs, query_len, block_size, max_num_blocks) in enumerate(TEST_SHAPES):
        try:
            max_num_reqs = num_reqs + 16
            torch.manual_seed(42 + test_idx)
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

            def _bench_fn():
                mod.compute_slot_mappings(idx_mapping, query_start_loc, positions, block_table, block_size, max_num_tokens)
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
                    "block_size": block_size,
                    "max_num_blocks": max_num_blocks
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "query_len": query_len,
                    "block_size": block_size,
                    "max_num_blocks": max_num_blocks
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
            "num_shapes": len(TEST_SHAPES) + len(ADDITIONAL_CORRECTNESS_CASES),
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
