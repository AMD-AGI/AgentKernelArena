#!/usr/bin/env python3
"""Task runner for triton2triton/triton_reshape_and_cache_flash_diffkv"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_reshape_and_cache_flash_diffkv"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_reshape_and_cache_flash_diffkv.py")

# Test configurations: (num_tokens, num_heads, head_size_k, head_size_v, num_blocks, block_size)
TEST_SHAPES = [
    (32, 8, 64, 32, 16, 16),
    (64, 16, 128, 64, 32, 16),
    (128, 32, 64, 64, 64, 16),
    (256, 8, 128, 128, 32, 32),
    (48, 16, 64, 128, 24, 8),
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


def reference_reshape_and_cache_diffkv(key, value, kv_cache, slot_mapping):
    """CPU/PyTorch reference for reshape_and_cache_flash_diffkv."""
    import torch
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        for h in range(num_heads):
            kv_cache[block_idx, block_offset, h, :head_size_k] = key[i, h]
            kv_cache[block_idx, block_offset, h, head_size_k:head_size_k + head_size_v] = value[i, h]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "reshape_and_cache_flash_diffkv"), "Missing reshape_and_cache_flash_diffkv"
        assert hasattr(mod, "reshape_and_cache_kernel_flash_diffkv"), "Missing reshape_and_cache_kernel_flash_diffkv"
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
    dtype = torch.float16

    for i, (num_tokens, num_heads, hk, hv, num_blocks, block_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            key = torch.randn(num_tokens, num_heads, hk, device=device, dtype=dtype)
            value = torch.randn(num_tokens, num_heads, hv, device=device, dtype=dtype)

            kv_cache = torch.zeros(num_blocks, block_size, num_heads, hk + hv, device=device, dtype=dtype)
            kv_cache_ref = kv_cache.clone()

            total_slots = num_blocks * block_size
            perm = torch.randperm(total_slots, device=device)[:num_tokens]
            slot_mapping = perm.to(torch.int64)

            mod.reshape_and_cache_flash_diffkv(key, value, kv_cache, slot_mapping)
            torch.cuda.synchronize()

            reference_reshape_and_cache_diffkv(key, value, kv_cache_ref, slot_mapping)

            if not torch.allclose(kv_cache, kv_cache_ref, atol=1e-3, rtol=1e-3):
                max_diff = (kv_cache - kv_cache_ref).abs().max().item()
                return False, f"Shape {i+1}: kv_cache max diff = {max_diff:.6f}"
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
    dtype = torch.float16
    test_cases = []

    for test_idx, (num_tokens, num_heads, hk, hv, num_blocks, block_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            key = torch.randn(num_tokens, num_heads, hk, device=device, dtype=dtype)
            value = torch.randn(num_tokens, num_heads, hv, device=device, dtype=dtype)
            kv_cache = torch.zeros(num_blocks, block_size, num_heads, hk + hv, device=device, dtype=dtype)
            total_slots = num_blocks * block_size
            slot_mapping = torch.randperm(total_slots, device=device)[:num_tokens].to(torch.int64)
            k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            def _bench_fn():
                mod.reshape_and_cache_flash_diffkv(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=kv_cache.zero_,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "num_tokens": num_tokens,
                    "num_heads": num_heads,
                    "head_size_k": hk,
                    "head_size_v": hv,
                    "num_blocks": num_blocks,
                    "block_size": block_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "num_tokens": num_tokens,
                    "num_heads": num_heads,
                    "head_size_k": hk,
                    "head_size_v": hv,
                    "num_blocks": num_blocks,
                    "block_size": block_size
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
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
