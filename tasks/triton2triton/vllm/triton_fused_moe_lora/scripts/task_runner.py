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
PERF_SHAPE_IDX = 4


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
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    M, K, num_experts, lora_rank, out_dim, num_loras, top_k = TEST_SHAPES[PERF_SHAPE_IDX]
    (output, qcurr, lora_a, lora_b, topk_weights,
     sorted_token_ids, expert_ids, num_tokens_post_padded,
     token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
     num_active_loras, adapter_enabled) = make_test_data(
        M, K, num_experts, lora_rank, out_dim, num_loras, top_k, device, 0)

    for _ in range(5):
        output.zero_()
        mod.fused_moe_lora(
            output, qcurr, lora_a, lora_b, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
            num_active_loras, adapter_enabled,
            mul_routed_weight=False, offset=0,
        )
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        output.zero_()
        start_events[j].record()
        mod.fused_moe_lora(
            output, qcurr, lora_a, lora_b, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            token_lora_mapping, max_lora_rank, top_k_num, lora_ids,
            num_active_loras, adapter_enabled,
            mul_routed_weight=False, offset=0,
        )
        end_events[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


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
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
