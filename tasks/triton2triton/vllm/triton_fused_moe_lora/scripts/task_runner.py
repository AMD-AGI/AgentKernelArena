#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fused_moe_lora"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_moe_lora"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe_lora.py")

# (M, K, num_experts, lora_rank, out_dim, num_loras, top_k)
TEST_SHAPES = [
    (8, 64, 4, 8, 32, 2, 2),
    (16, 128, 4, 16, 64, 2, 2),
    (32, 256, 8, 16, 128, 4, 2),
    (64, 512, 8, 32, 256, 4, 2),
    (128, 1024, 8, 32, 512, 4, 2),
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


def reference_fused_moe_lora(qcurr_hidden_states, lora_a_stacked, lora_b_stacked,
                              topk_weights, expert_ids, token_lora_mapping,
                              top_k_num, adapter_enabled, mul_routed_weight):
    """CPU reference: per-token shrink then expand with MoE expert routing.
    Uses naive (non-sorted) assignment for simplicity."""
    import torch
    M = qcurr_hidden_states.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_slices = len(lora_a_stacked)
    max_lora_rank = lora_a_stacked[0].shape[2]
    w1_out_dim = lora_b_stacked[0].shape[2]
    out_dim = num_slices * w1_out_dim
    num_tokens = M * top_k_num

    # intermediate: [num_slices, M, top_k, max_lora_rank]
    intermediate = torch.zeros(num_slices, M, top_k_num, max_lora_rank,
                               dtype=torch.float32, device=qcurr_hidden_states.device)

    # Shrink: for each token, for each top_k, do input @ lora_a[expert].T
    for token_idx in range(M):
        lora_id = token_lora_mapping[token_idx].item()
        if lora_id == -1:
            continue
        if adapter_enabled[lora_id].item() == 0:
            continue
        for k in range(top_k_num):
            flat_idx = token_idx * top_k_num + k
            exp_id = expert_ids[flat_idx].item()
            if exp_id == -1:
                continue
            inp = qcurr_hidden_states[token_idx].float()
            for s in range(num_slices):
                # lora_a: [max_loras, num_experts, max_lora_rank, K]
                wa = lora_a_stacked[s][lora_id, exp_id].float()  # [rank, K]
                intermediate[s, token_idx, k] = inp @ wa.T

    # Expand: intermediate @ lora_b[expert].T -> output
    output = torch.zeros(M, top_k_num, out_dim,
                         dtype=qcurr_hidden_states.dtype,
                         device=qcurr_hidden_states.device).float()

    for token_idx in range(M):
        lora_id = token_lora_mapping[token_idx].item()
        if lora_id == -1:
            continue
        if adapter_enabled[lora_id].item() == 0:
            continue
        for k in range(top_k_num):
            flat_idx = token_idx * top_k_num + k
            exp_id = expert_ids[flat_idx].item()
            if exp_id == -1:
                continue
            for s in range(num_slices):
                inter = intermediate[s, token_idx, k].float()
                # lora_b: [max_loras, num_experts, out_dim_per_slice, max_lora_rank]
                wb = lora_b_stacked[s][lora_id, exp_id].float()  # [out_dim_per_slice, rank]
                result = inter @ wb.T
                if mul_routed_weight:
                    result *= topk_weights[token_idx, k].item()
                col_start = s * w1_out_dim
                col_end = col_start + w1_out_dim
                output[token_idx, k, col_start:col_end] += result

    return output.to(qcurr_hidden_states.dtype)


def make_test_data(M, K, num_experts, lora_rank, out_dim, num_loras, top_k, device, seed):
    import torch
    torch.manual_seed(seed)
    num_slices = 1

    qcurr = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
    topk_weights = torch.randn(M, top_k, device=device, dtype=torch.float32).abs()

    # lora_a: [num_loras, num_experts, lora_rank, K]
    lora_a = [torch.randn(num_loras, num_experts, lora_rank, K,
                           device=device, dtype=torch.float16) * 0.1]
    # lora_b: [num_loras, num_experts, out_dim, lora_rank]
    lora_b = [torch.randn(num_loras, num_experts, out_dim, lora_rank,
                           device=device, dtype=torch.float16) * 0.1]

    # Naive assignment: expert_ids is flat [M * top_k]
    expert_ids = torch.randint(0, num_experts, (M * top_k,), device=device, dtype=torch.int64)
    token_lora_mapping = torch.randint(0, num_loras, (M,), device=device, dtype=torch.int64)
    lora_ids = torch.arange(num_loras, device=device, dtype=torch.int64)
    adapter_enabled = torch.ones(num_loras, device=device, dtype=torch.int32)

    output = torch.zeros(M, top_k, out_dim * num_slices, device=device, dtype=torch.float16)

    return (output, qcurr, lora_a, lora_b, topk_weights,
            None, expert_ids, None, token_lora_mapping,
            lora_rank, top_k, lora_ids, num_loras, adapter_enabled)


def prepare_direct_launch(mod, output, qcurr_hidden_states, lora_a_stacked,
                          lora_b_stacked, topk_weights, sorted_token_ids,
                          expert_ids, num_tokens_post_padded,
                          token_lora_mapping, max_lora_rank, top_k_num,
                          lora_ids, num_active_loras, adapter_enabled,
                          mul_routed_weight=False, offset=0):
    """Prepare stable shrink/expand launches for graph-first benchmarking.

    The public wrapper allocates pointer tables and an intermediate tensor on
    every call. Those operations are deliberately hoisted here, along with all
    views, grids, strides, scalar arguments, and Triton meta-parameters. The
    returned callables only enqueue already-prepared device operations.
    """
    import torch

    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_a_stacked = lora_a_stacked[0]
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = w1_lora_a_stacked.shape[1]
    shrink_n = max_lora_rank
    num_tokens_base = topk_weights.shape[0]
    shrink_k = qcurr_hidden_states.shape[1]
    num_tokens = num_tokens_base * top_k_num
    output_dim = w1_lora_b_stacked.shape[2]

    shrink_block_size_m = 64
    shrink_block_size_n = min(64, mod._next_power_of_2(shrink_n))
    shrink_block_size_k = 32
    shrink_group_size_m = 8
    shrink_num_warps = 4
    shrink_num_stages = 3
    shrink_split_k = 1

    expand_block_size_m = 64
    expand_block_size_n = 64
    expand_block_size_k = max(16, min(32, mod._next_power_of_2(shrink_n)))
    expand_group_size_m = 8
    expand_num_warps = 4
    expand_num_stages = 3

    em = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens * shrink_block_size_m
    )
    grid_lora_dim, stride_tl, stride_el = mod._adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    grid_lora_dim2, stride_tl2, stride_el2 = mod._adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    # Both pointer tables and every tensor/view referenced by a captured graph
    # must retain a stable address for the graph's full lifetime.
    lora_a_ptrs = mod._get_ptr(lora_a_stacked, device)
    lora_b_ptrs = mod._get_ptr(lora_b_stacked, device)
    intermediate = torch.zeros(
        (num_slices, num_tokens_base, top_k_num, max_lora_rank),
        dtype=output.dtype,
        device=device,
    )
    intermediate_flat = intermediate.view(-1, intermediate.shape[3])
    out_view = output[:, :, offset: offset + num_slices * output_dim]

    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    shrink_grid = (
        shrink_split_k
        * _ceil_div(em, shrink_block_size_m)
        * _ceil_div(shrink_n, shrink_block_size_n),
        num_slices,
        grid_lora_dim,
    )
    expand_grid = (
        _ceil_div(em, expand_block_size_m)
        * _ceil_div(output_dim, expand_block_size_n),
        num_slices,
        grid_lora_dim2,
    )

    shrink_args = (
        qcurr_hidden_states,
        lora_a_ptrs,
        intermediate,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        shrink_n,
        shrink_k,
        em,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        w1_lora_a_stacked.shape[0],
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        intermediate.stride(2),
        intermediate.stride(3),
        stride_tl,
        stride_el,
    )
    shrink_meta = {
        "slice_a_size": qcurr_hidden_states.numel(),
        "slice_c_size": intermediate.numel() // num_slices,
        "num_slice_a": 1,
        "num_slice_c": num_slices,
        "token_mapping_factor": 1 if mul_routed_weight else top_k_num,
        "naive_block_assignment": sorted_token_ids is None,
        "MUL_ROUTED_WEIGHT": False,
        "ADD_INPUTS": False,
        "USE_B_L2_CACHE": True,
        "IS_PRIMARY": True,
        "BLOCK_SIZE_M": shrink_block_size_m,
        "BLOCK_SIZE_N": shrink_block_size_n,
        "BLOCK_SIZE_K": shrink_block_size_k,
        "GROUP_SIZE_M": shrink_group_size_m,
        "SPLIT_K": shrink_split_k,
        "USE_GDC": False,
        "launch_pdl": False,
        "num_warps": shrink_num_warps,
        "num_stages": shrink_num_stages,
    }

    expand_args = (
        intermediate_flat,
        lora_b_ptrs,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        output_dim,
        max_lora_rank,
        em,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        w1_lora_b_stacked.shape[0],
        intermediate_flat.stride(0),
        intermediate_flat.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        stride_tl2,
        stride_el2,
    )
    expand_meta = {
        "slice_a_size": intermediate_flat.numel() // num_slices,
        "slice_c_size": output_dim * out_view.stride(2),
        "num_slice_a": num_slices,
        "num_slice_c": num_slices,
        "token_mapping_factor": 1,
        "naive_block_assignment": sorted_token_ids is None,
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "ADD_INPUTS": True,
        "USE_B_L2_CACHE": True,
        "IS_PRIMARY": False,
        "BLOCK_SIZE_M": expand_block_size_m,
        "BLOCK_SIZE_N": expand_block_size_n,
        "BLOCK_SIZE_K": expand_block_size_k,
        "GROUP_SIZE_M": expand_group_size_m,
        "SPLIT_K": 1,
        "USE_GDC": False,
        "launch_pdl": False,
        "num_warps": expand_num_warps,
        "num_stages": expand_num_stages,
    }

    shrink_launcher = mod.fused_moe_lora_kernel[shrink_grid]
    expand_launcher = mod.fused_moe_lora_kernel[expand_grid]
    reset_output = output.zero_

    def launch_shrink():
        shrink_launcher(*shrink_args, **shrink_meta)

    def launch_expand():
        expand_launcher(*expand_args, **expand_meta)

    def launch_reset_expand():
        reset_output()
        launch_expand()

    def launch_fused_no_reset():
        # Shrink uses SPLIT_K=1 and ADD_INPUTS=False, so it overwrites every
        # intermediate row consumed by expand; no workspace reset is required.
        launch_shrink()
        launch_expand()

    def launch_fused():
        reset_output()
        launch_fused_no_reset()

    return {
        "fused": launch_fused,
        "fused_no_reset": launch_fused_no_reset,
        "shrink": launch_shrink,
        "reset_expand": launch_reset_expand,
        "reset_output": reset_output,
        "output": output,
        "intermediate": intermediate,
        "lora_a_ptrs": lora_a_ptrs,
        "lora_b_ptrs": lora_b_ptrs,
    }


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_moe_lora"), "Missing fused_moe_lora"
        assert hasattr(mod, "fused_moe_lora_kernel"), "Missing fused_moe_lora_kernel"
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
    for i, (M, K, num_experts, lora_rank, out_dim, num_loras, top_k) in enumerate(TEST_SHAPES):
        try:
            (output, qcurr, lora_a, lora_b, topk_weights,
             sorted_token_ids, expert_ids, num_tokens_post_padded,
             token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
             num_active_loras, adapter_enabled) = make_test_data(
                M, K, num_experts, lora_rank, out_dim, num_loras, top_k, device, 42 + i)

            mod.fused_moe_lora(
                output, qcurr, lora_a, lora_b, topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
                num_active_loras, adapter_enabled,
                mul_routed_weight=False, offset=0,
            )
            torch.cuda.synchronize()

            ref = reference_fused_moe_lora(
                qcurr, lora_a, lora_b, topk_weights, expert_ids,
                token_lora_mapping, top_k_num, adapter_enabled,
                mul_routed_weight=False).to(device)

            if not torch.allclose(output.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1} (M={M},K={K}): max diff = {max_diff:.6f}"

            direct = prepare_direct_launch(
                mod, output, qcurr, lora_a, lora_b, topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
                num_active_loras, adapter_enabled,
                mul_routed_weight=False, offset=0,
            )
            direct["fused"]()
            torch.cuda.synchronize()
            if not torch.allclose(output.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return False, (
                    f"Direct shape {i+1} (M={M},K={K}): "
                    f"max diff = {max_diff:.6f}"
                )
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

    for test_idx, (M, K, num_experts, lora_rank, out_dim, num_loras, top_k) in enumerate(TEST_SHAPES):
        try:
            (output, qcurr, lora_a, lora_b, topk_weights,
             sorted_token_ids, expert_ids, num_tokens_post_padded,
             token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
             num_active_loras, adapter_enabled) = make_test_data(
                M, K, num_experts, lora_rank, out_dim, num_loras, top_k, device, 0)

            direct = prepare_direct_launch(
                mod, output, qcurr, lora_a, lora_b, topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
                num_active_loras, adapter_enabled,
                mul_routed_weight=False, offset=0,
            )

            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                direct["fused_no_reset"],
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                prepare_fn=direct["reset_output"],
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "M": M,
                    "K": K,
                    "num_experts": num_experts,
                    "lora_rank": lora_rank,
                    "out_dim": out_dim,
                    "num_loras": num_loras,
                    "top_k": top_k
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "num_experts": num_experts,
                    "lora_rank": lora_rank,
                    "out_dim": out_dim,
                    "num_loras": num_loras,
                    "top_k": top_k
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
