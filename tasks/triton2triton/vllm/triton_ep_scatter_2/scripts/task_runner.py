#!/usr/bin/env python3
"""Task runner for triton2triton/triton_ep_scatter_2"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_ep_scatter_2"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ep_scatter_2.py")

# (num_tokens, hidden_size, num_experts, topk)
TEST_SHAPES = [
    (16, 64, 4, 2),
    (32, 128, 8, 2),
    (64, 256, 8, 2),
    (128, 512, 16, 2),
    (256, 512, 8, 2),
]

# Correctness-only coverage. Keep TEST_SHAPES above stable because it also defines
# the scored performance workload.
# (name, num_tokens, hidden_size, num_experts, topk, dtype, negative_assignments)
CORRECTNESS_CASES = [
    (f"baseline_{i + 1}", *shape, "float16", False)
    for i, shape in enumerate(TEST_SHAPES)
] + [
    ("single_token_boundary", 1, 1, 1, 1, "float32", False),
    ("non_power_of_two_topk_1", 37, 65, 5, 1, "float32", True),
    ("non_power_of_two_topk_3", 129, 257, 8, 3, "float16", True),
    ("above_program_cap", 8209, 33, 8, 4, "float16", True),
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


def round_up_128(x):
    return ((x + 127) // 128) * 128


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "ep_scatter_2"), "Missing ep_scatter_2"
        assert hasattr(mod, "_fwd_kernel_ep_scatter_2"), "Missing _fwd_kernel_ep_scatter_2"
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
        (case_name, num_tokens, hidden_size, num_experts, topk,
         dtype_name, negative_assignments) = case
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, dtype_name)
            recv_x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
            recv_topk = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
            if negative_assignments:
                # Deterministically exercise ignored assignments throughout the
                # tensor, including beyond the kernel's 8192-program grid cap.
                recv_topk.view(-1)[::5] = -1

            # Compute tokens per expert
            recv_topk_cpu = recv_topk.cpu()
            valid_assignments_cpu = recv_topk_cpu[recv_topk_cpu >= 0]
            counts = torch.bincount(
                valid_assignments_cpu.to(torch.int64), minlength=num_experts
            ).to(torch.int32)
            aligned = [round_up_128(c.item()) for c in counts]
            total = sum(aligned)

            starts = []
            s = 0
            for a in aligned:
                starts.append(s)
                s += a
            initial_expert_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)
            expert_start_loc = initial_expert_start_loc.clone()

            output_tensor = torch.zeros(total, hidden_size, device=device, dtype=dtype)
            output_index = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)

            mod.ep_scatter_2(recv_x, recv_topk, expert_start_loc, output_tensor, output_index)
            torch.cuda.synchronize()

            valid_mask = recv_topk >= 0
            invalid_mask = ~valid_mask
            if invalid_mask.any() and not torch.all(output_index[invalid_mask] == -1):
                return False, f"Case {case_name}: negative assignment modified output_index"

            valid_indices = output_index[valid_mask].to(torch.int64)
            valid_experts = recv_topk[valid_mask].to(torch.int64)
            region_starts = initial_expert_start_loc[valid_experts].to(torch.int64)
            counts_device = counts.to(device=device, dtype=torch.int64)
            region_ends = region_starts + counts_device[valid_experts]
            in_expert_region = ((valid_indices >= region_starts)
                                & (valid_indices < region_ends))
            if not torch.all(in_expert_region):
                return False, f"Case {case_name}: output_index outside assigned expert region"

            if torch.unique(valid_indices).numel() != valid_indices.numel():
                return False, f"Case {case_name}: duplicate output_index values"

            token_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)
            token_ids = token_ids[:, None].expand(num_tokens, topk)[valid_mask]
            if not torch.equal(output_tensor[valid_indices], recv_x[token_ids]):
                return False, f"Case {case_name}: scattered token data mismatch"

            expected_final_counters = initial_expert_start_loc + counts.to(device)
            if not torch.equal(expert_start_loc, expected_final_counters):
                return False, f"Case {case_name}: final expert counters mismatch"
        except Exception as e:
            return False, f"Case {case_name}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (num_tokens, hidden_size, num_experts, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            recv_x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
            recv_topk = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

            counts = torch.zeros(num_experts, dtype=torch.int32)
            for e in range(num_experts):
                counts[e] = (recv_topk.cpu() == e).sum().item()
            aligned = [round_up_128(c.item()) for c in counts]
            total = sum(aligned)
            starts = []
            s = 0
            for a in aligned:
                starts.append(s)
                s += a

            expert_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)
            initial_expert_start_loc = expert_start_loc.clone()
            output_tensor = torch.zeros(total, hidden_size, device=device, dtype=torch.float16)
            output_index = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)

            grid = (min(num_tokens, 1024 * 8),)
            recv_x_stride0, recv_x_stride1 = recv_x.stride()
            recv_topk_stride0, recv_topk_stride1 = recv_topk.stride()
            output_stride0, output_stride1 = output_tensor.stride()
            output_index_stride0, output_index_stride1 = output_index.stride()
            hidden_size_pad = mod.triton.next_power_of_2(hidden_size)

            def _bench_fn():
                mod._fwd_kernel_ep_scatter_2[grid](
                    num_tokens,
                    expert_start_loc,
                    recv_x, recv_x_stride0, recv_x_stride1,
                    recv_topk, recv_topk_stride0, recv_topk_stride1,
                    output_tensor, output_stride0, output_stride1,
                    output_index, output_index_stride0, output_index_stride1,
                    topk_num=topk,
                    num_warps=8,
                    HIDDEN_SIZE=hidden_size,
                    HIDDEN_SIZE_PAD=hidden_size_pad,
                )

            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=lambda: expert_start_loc.copy_(initial_expert_start_loc),
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "num_experts": num_experts,
                    "topk": topk
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance_case_exception",
                "params": {
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(CORRECTNESS_CASES)}
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
