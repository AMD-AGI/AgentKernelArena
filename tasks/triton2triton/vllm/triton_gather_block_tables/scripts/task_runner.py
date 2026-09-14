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

# Correctness-only cases for non-power-of-two row counts and widths. The
# explicit mappings repeat requests whose block counts cover empty, full, and
# partial rows. Keeping these separate from TEST_SHAPES ensures correctness
# coverage does not alter the performance workload.
CORRECTNESS_EDGE_CASES = [
    {
        "name": "short_non_power_of_two",
        "shape": (5, 11, 17),
        "idx_mapping": (0, 1, 0, 10, 2),
        "num_blocks": {0: 0, 1: 17, 10: 8, 2: 1},
    },
    {
        "name": "wide_non_power_of_two",
        "shape": (7, 13, 1009),
        "idx_mapping": (3, 1, 3, 12, 0, 1, 6),
        "num_blocks": {3: 0, 1: 1009, 12: 1008, 0: 1, 6: 503},
    },
    {
        "name": "just_above_block_size",
        "shape": (9, 17, 1025),
        "idx_mapping": (4, 2, 4, 16, 0, 2, 7, 1, 9),
        "num_blocks": {4: 0, 2: 1025, 16: 1024, 0: 1, 7: 513, 1: 17, 9: 1009},
    },
    {
        "name": "multi_iteration_non_power_of_two",
        "shape": (11, 19, 1537),
        "idx_mapping": (5, 3, 5, 18, 0, 3, 8, 1, 10, 2, 12),
        "num_blocks": {
            5: 0,
            3: 1537,
            18: 1536,
            0: 1,
            8: 769,
            1: 1024,
            10: 1025,
            2: 17,
            12: 1501,
        },
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


def reference_gather(idx_mapping, src_block_table, dst_block_table, num_blocks):
    num_reqs = idx_mapping.shape[0]
    dst = dst_block_table.clone()
    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        nb = num_blocks[req_idx].item()
        dst[b, :nb] = src_block_table[req_idx, :nb]
    return dst


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

            ref = reference_gather(
                idx_mapping.cpu(),
                src_block_table.cpu(),
                torch.zeros_like(src_block_table, device="cpu"),
                num_blocks.cpu(),
            )
            if not torch.equal(result.cpu(), ref[:num_reqs]):
                return False, f"Shape {i+1}: mismatch"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for case in CORRECTNESS_EDGE_CASES:
        name = case["name"]
        num_reqs, max_num_reqs, max_num_blocks = case["shape"]
        try:
            torch.manual_seed(1000 + max_num_blocks)
            idx_mapping = torch.tensor(
                case["idx_mapping"], dtype=torch.int32, device=device
            )
            src_block_table = torch.randint(
                0,
                10000,
                (max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            dst_block_table = -torch.arange(
                1,
                max_num_reqs * max_num_blocks + 1,
                dtype=torch.int32,
                device=device,
            ).reshape(max_num_reqs, max_num_blocks)
            initial_dst = dst_block_table.clone()
            num_blocks = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            for req_idx, count in case["num_blocks"].items():
                num_blocks[req_idx] = count

            result = mod.gather_block_tables(
                idx_mapping, src_block_table, dst_block_table, num_blocks
            )
            torch.cuda.synchronize()

            ref = reference_gather(
                idx_mapping.cpu(),
                src_block_table.cpu(),
                initial_dst.cpu(),
                num_blocks.cpu(),
            )
            if not torch.equal(result.cpu(), ref[:num_reqs]):
                return False, f"Edge case {name}: returned rows mismatch"
            if not torch.equal(dst_block_table.cpu(), ref):
                return False, f"Edge case {name}: destination buffer mismatch"
        except Exception as e:
            return False, f"Edge case {name}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_reqs, max_num_reqs, max_num_blocks) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
            src_block_table = torch.randint(0, 10000, (max_num_reqs, max_num_blocks),
                                             dtype=torch.int32, device=device)
            dst_block_table = torch.zeros_like(src_block_table)
            num_blocks = torch.full((max_num_reqs,), max_num_blocks, dtype=torch.int32, device=device)

            def _bench_fn():
                mod.gather_block_tables(idx_mapping, src_block_table, dst_block_table, num_blocks)
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
                    "max_num_reqs": max_num_reqs,
                    "max_num_blocks": max_num_blocks
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_reqs": num_reqs,
                    "max_num_reqs": max_num_reqs,
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
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_EDGE_CASES),
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
