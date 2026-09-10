#!/usr/bin/env python3
"""Task runner for triton2triton/triton_merge_16x16_to_32x32"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_merge_16x16_to_32x32"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_merge_16x16_to_32x32.py")

SEEDS = [42, 43, 44, 45, 46]
CORRECTNESS_CASES = [
    {"seed": seed, "B": 2, "T": 64, "H": 4, "dtype": "float32"}
    for seed in SEEDS
] + [
    {"seed": 47, "B": 1, "T": 31, "H": 1, "dtype": "float16"},
    {"seed": 48, "B": 3, "T": 33, "H": 2, "dtype": "bfloat16"},
    {"seed": 49, "B": 1, "T": 47, "H": 5, "dtype": "float32"},
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
PERF_SEED_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(A):
    import torch
    B, T, H, BT = A.shape
    assert BT == 32
    from math import ceil
    NT = ceil(T / BT)
    Ai = torch.zeros_like(A, dtype=torch.float32, device='cpu')
    A_cpu = A.float().cpu()
    for b in range(B):
        for h in range(H):
            for t_idx in range(NT):
                start = t_idx * BT
                end = min(start + BT, T)
                sz = end - start
                block = A_cpu[b, start:end, h, :sz]
                I_plus_A = torch.eye(sz) + torch.tril(block, diagonal=-1)
                inv_block = torch.linalg.inv(I_plus_A)
                Ai[b, start:end, h, :sz] = inv_block
    return Ai


def gen_inputs(seed, device, B=2, T=64, H=4, dtype="float32"):
    import torch
    torch.manual_seed(seed)
    BT = 32
    A = torch.randn(B, T, H, BT, device=device, dtype=getattr(torch, dtype)) * 0.1
    # The kernel expects strictly lower triangular blocks (within each BT x BT tile).
    # Zero out diagonal and upper triangular entries in the last dimension.
    idx = torch.arange(BT, device=device)
    t_in_block = torch.arange(T, device=device) % BT
    mask = t_in_block[:, None] > idx[None, :]  # [T, BT] strict lower
    A = A * mask[None, :, None, :]  # broadcast over B and H
    return (A,), {}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "merge_16x16_to_32x32"), "Missing merge_16x16_to_32x32"
        assert hasattr(mod, "merge_16x16_to_32x32_inverse_kernel"), "Missing merge_16x16_to_32x32_inverse_kernel"
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
    for i, case in enumerate(CORRECTNESS_CASES):
        try:
            args, kwargs = gen_inputs(device=device, **case)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result = mod.merge_16x16_to_32x32(*args, **kwargs)
            ref = reference(*args_cpu, **kwargs)
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=1e-2, rtol=1e-2):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return False, f"Case {i+1} {case}: max diff = {max_diff:.6f}"

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Case {i+1} {case}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, seed in enumerate(SEEDS):
        try:
            args, kwargs = gen_inputs(seed, device)

            def _bench_fn():
                mod.merge_16x16_to_32x32(*args, **kwargs)
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
                    "seed": seed
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seed": seed
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(CORRECTNESS_CASES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
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
