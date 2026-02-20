#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fused_moe"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_moe"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe.py")

# (M, K, E, N, topk)
TEST_SHAPES = [
    (16, 64, 4, 64, 2),
    (32, 128, 8, 128, 2),
    (64, 256, 8, 256, 2),
    (128, 512, 16, 512, 2),
    (256, 1024, 8, 1024, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_fused_moe(input, expert_weights, topk_ids, topk_weights, mul_routed_weight):
    """CPU reference: per-token expert GEMM with optional weight scaling."""
    import torch
    M, K = input.shape
    E, N, _ = expert_weights.shape
    topk = topk_ids.shape[1]
    num_valid = M * topk
    output = torch.zeros(num_valid, N, device=input.device, dtype=torch.float32)

    for token_idx in range(M):
        for k in range(topk):
            flat_idx = token_idx * topk + k
            expert_id = topk_ids[token_idx, k].item()
            if expert_id < 0 or expert_id >= E:
                continue
            # C = A @ B^T where B is [N, K]
            row = input[token_idx].float() @ expert_weights[expert_id].float().T
            if mul_routed_weight:
                row *= topk_weights[flat_idx].item()
            output[flat_idx] = row
    return output.to(input.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_moe"), "Missing fused_moe"
        assert hasattr(mod, "fused_moe_kernel"), "Missing fused_moe_kernel"
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
    for i, (M, K, E, N, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            expert_weights = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

            result = mod.fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, mul_routed_weight=True)
            torch.cuda.synchronize()

            ref = reference_fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, True).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1} (M={M},K={K},E={E},N={N}): max diff = {max_diff:.6f}"
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
    M, K, E, N, topk = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
    expert_weights = torch.randn(E, N, K, device=device, dtype=torch.float16) * 0.1
    topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
    topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

    for _ in range(5):
        mod.fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, True)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.fused_moe(input_tensor, expert_weights, topk_ids, topk_weights_flat, True)
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
