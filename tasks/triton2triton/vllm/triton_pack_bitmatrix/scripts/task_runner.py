#!/usr/bin/env python3
"""Task runner for triton2triton/triton_pack_bitmatrix"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_pack_bitmatrix"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_pack_bitmatrix.py")

# (n_rows, num_experts, topk)
TEST_SHAPES = [
    (32, 8, 2),
    (64, 16, 2),
    (128, 32, 2),
    (256, 64, 4),
    (512, 64, 2),
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


def reference_pack_bitmatrix(topk_ids, num_experts):
    """Independent CPU expert-membership oracle; padding is not an assignment."""
    import torch
    n_rows, topk = topk_ids.shape
    bitmatrix = torch.zeros(n_rows, (num_experts + 31) // 32, dtype=torch.uint32)
    for row in range(n_rows):
        for eid in topk_ids[row].tolist():
            if not 0 <= eid < num_experts:
                raise ValueError("Expert ID is outside the declared expert range")
            col, bit = divmod(eid, 32)
            bitmatrix[row, col] = bitmatrix[row, col].item() | (1 << bit)
    return bitmatrix


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "pack_topk_to_bitmatrix"), "Missing pack_topk_to_bitmatrix"
        assert hasattr(mod, "pack_bitmatrix"), "Missing pack_bitmatrix"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness(*, case_index=None):
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, (n_rows, num_experts, topk) in enumerate(TEST_SHAPES):
        if case_index is not None and i != case_index:
            continue
        try:
            torch.manual_seed(42 + i)
            topk_ids = torch.randint(0, num_experts, (n_rows, topk), device=device, dtype=torch.int16)

            result = mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
            torch.cuda.synchronize()

            ref = reference_pack_bitmatrix(topk_ids.cpu(), num_experts).to(device)
            if not torch.equal(result, ref):
                diff_count = (result != ref).sum().item()
                return False, f"Shape {i+1}: {diff_count} mismatched elements"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (n_rows, num_experts, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            topk_ids = torch.randint(0, num_experts, (n_rows, topk), device=device, dtype=torch.int16)

            def _bench_fn():
                mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
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
                    "n_rows": n_rows,
                    "num_experts": num_experts,
                    "topk": topk
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "n_rows": n_rows,
                    "num_experts": num_experts,
                    "topk": topk
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
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
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
