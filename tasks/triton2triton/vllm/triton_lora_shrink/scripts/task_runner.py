#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lora_shrink"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_lora_shrink"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lora_shrink.py")

# (M, hidden_size, lora_rank, num_loras, num_slices)
TEST_SHAPES = [
    (16, 64, 8, 2, 1),
    (32, 128, 16, 4, 1),
    (64, 256, 16, 4, 2),
    (128, 512, 32, 8, 1),
    (256, 1024, 32, 8, 2),
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


def reference_lora_shrink(inputs, lora_a_weights, token_indices, num_tokens_per_lora,
                          lora_token_start_loc, lora_ids, scaling):
    """CPU reference for LoRA shrink (A) operation."""
    import torch
    num_slices = len(lora_a_weights)
    M = inputs.shape[0]
    lora_rank = lora_a_weights[0].shape[-2]
    output = torch.zeros(num_slices, M, lora_rank, device=inputs.device, dtype=torch.float32)

    for lora_idx in range(lora_ids.shape[0]):
        lora_id = lora_ids[lora_idx].item()
        if lora_id == -1:
            continue
        n_tokens = num_tokens_per_lora[lora_idx].item()
        start = lora_token_start_loc[lora_idx].item()
        for t in range(n_tokens):
            token_id = token_indices[start + t].item()
            for s in range(num_slices):
                w = lora_a_weights[s]
                if w.ndim == 4:
                    w = w.squeeze(1)
                # w shape: [num_loras, lora_rank, hidden_size]
                inp = inputs[token_id].float()  # [hidden_size]
                weight = w[lora_id].float()  # [lora_rank, hidden_size]
                out_row = inp @ weight.T  # [lora_rank]
                output[s, token_id] = out_row * scaling

    return output.to(inputs.dtype)


def make_test_data(M, hidden_size, lora_rank, num_loras, num_slices, device, seed):
    import torch
    torch.manual_seed(seed)

    inputs = torch.randn(M, hidden_size, device=device, dtype=torch.float16) * 0.1

    lora_a_weights = []
    for _ in range(num_slices):
        w = torch.randn(num_loras, lora_rank, hidden_size, device=device, dtype=torch.float16) * 0.1
        lora_a_weights.append(w)

    output_tensor = torch.zeros(num_slices, M, lora_rank, device=device, dtype=torch.float32)

    token_lora_mapping = torch.randint(0, num_loras, (M,), device=device, dtype=torch.int64)

    lora_ids_list = list(range(num_loras))
    lora_ids = torch.tensor(lora_ids_list, device=device, dtype=torch.int64)

    sorted_indices = []
    num_tokens_list = []
    for lid in lora_ids_list:
        mask = (token_lora_mapping == lid)
        indices = mask.nonzero(as_tuple=True)[0]
        sorted_indices.append(indices)
        num_tokens_list.append(len(indices))

    token_indices_sorted = torch.cat(sorted_indices).to(device)
    num_tokens_per_lora = torch.tensor(num_tokens_list, device=device, dtype=torch.int64)
    cumsum = [0]
    for n in num_tokens_list:
        cumsum.append(cumsum[-1] + n)
    lora_token_start_loc = torch.tensor(cumsum, device=device, dtype=torch.int64)

    num_active_loras = num_loras
    scaling = 0.5

    return (inputs, lora_a_weights, output_tensor, token_lora_mapping,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
            lora_ids, num_active_loras, scaling)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lora_shrink"), "Missing lora_shrink"
        assert hasattr(mod, "_lora_shrink_kernel"), "Missing _lora_shrink_kernel"
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
    for i, (M, hidden_size, lora_rank, num_loras, num_slices) in enumerate(TEST_SHAPES):
        try:
            (inputs, lora_a_weights, output_tensor, token_lora_mapping,
             token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
             lora_ids, num_active_loras, scaling) = make_test_data(
                M, hidden_size, lora_rank, num_loras, num_slices, device, 42 + i)

            mod.lora_shrink(
                inputs, lora_a_weights, output_tensor, token_lora_mapping,
                token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
                lora_ids, num_active_loras, scaling,
            )
            torch.cuda.synchronize()

            ref = reference_lora_shrink(
                inputs, lora_a_weights, token_indices_sorted, num_tokens_per_lora,
                lora_token_start_loc, lora_ids, scaling).to(device)

            if not torch.allclose(output_tensor.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (output_tensor.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1} (M={M}): max diff = {max_diff:.6f}"
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

    for test_idx, (M, hidden_size, lora_rank, num_loras, num_slices) in enumerate(TEST_SHAPES):
        try:
            (inputs, lora_a_weights, output_tensor, token_lora_mapping,
             token_indices_sorted, num_tokens_per_lora, lora_token_start_loc,
             lora_ids, num_active_loras, scaling) = make_test_data(
                M, hidden_size, lora_rank, num_loras, num_slices, device, 0)

            # Avoid the wrapper's per-call CUDA pointer tensor construction.
            # The split-K kernel accumulates atomically, so reset its output via
            # prepare_fn. Preparation is stream-ordered but excluded from both
            # Graph and Event timing, matching other stateful tasks.
            (
                lora_ptr_tensor,
                lora_stride0,
                lora_stride1,
                lora_stride2,
            ) = mod._get_lora_a_ptr(lora_a_weights, inputs.device)
            split_k = 64 if M < 128 else 8
            block_m = 32
            block_n = 16
            block_k = 256 if M < 128 else 32
            even_k = hidden_size % (block_k * split_k) == 0
            grid = (
                split_k
                * mod.triton.cdiv(M, block_m)
                * mod.triton.cdiv(lora_rank, block_n),
                num_slices,
                num_active_loras,
            )
            input_stride0, input_stride1 = inputs.stride()
            output_stride0, output_stride1, output_stride2 = output_tensor.stride()

            def _bench_fn():
                mod._lora_shrink_kernel[grid](
                    inputs,
                    lora_ptr_tensor,
                    output_tensor,
                    M,
                    lora_rank,
                    hidden_size,
                    token_indices_sorted,
                    num_tokens_per_lora,
                    lora_token_start_loc,
                    lora_ids,
                    scaling,
                    input_stride0,
                    input_stride1,
                    lora_stride0,
                    lora_stride1,
                    lora_stride2,
                    output_stride0,
                    output_stride1,
                    output_stride2,
                    block_m,
                    block_n,
                    block_k,
                    even_k,
                    split_k,
                    8,  # GROUP_SIZE_M
                    num_slices,
                    False,  # USE_GDC
                    num_warps=4,
                    num_stages=2,
                    launch_pdl=False,
                )

            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=output_tensor.zero_,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "M": M,
                    "hidden_size": hidden_size,
                    "lora_rank": lora_rank,
                    "num_loras": num_loras,
                    "num_slices": num_slices
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance_case_exception",
                "params": {
                    "M": M,
                    "hidden_size": hidden_size,
                    "lora_rank": lora_rank,
                    "num_loras": num_loras,
                    "num_slices": num_slices
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
