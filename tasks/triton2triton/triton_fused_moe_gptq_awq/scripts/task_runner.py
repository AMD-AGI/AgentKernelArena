#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fused_moe_gptq_awq"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_moe_gptq_awq"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe_gptq_awq.py")

# (M, K, E, N, topk, group_size)
TEST_SHAPES = [
    (16, 64, 4, 64, 2, 32),
    (32, 128, 4, 128, 2, 64),
    (64, 128, 8, 128, 2, 64),
    (64, 256, 8, 256, 2, 128),
    (128, 256, 8, 256, 2, 128),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_moe_gptq_awq"), "Missing fused_moe_gptq_awq"
        assert hasattr(mod, "fused_moe_kernel_gptq_awq"), "Missing fused_moe_kernel_gptq_awq"
        return True, None
    except Exception as e:
        return False, str(e)


def reference_fused_moe_int4(input_t, qweight, scales, zeros, topk_ids,
                             topk_weights, mul_routed_weight, group_size):
    """CPU reference for INT4 (GPTQ/AWQ) weight-only MoE.

    The Triton kernel packs 2 x int4 values per uint8 element along the K
    dimension:
        qweight: [E, K//2, N]  uint8  – low nibble = even k, high nibble = odd k
        scales:  [E, K//group_size, N] fp16
        zeros:   [E, K//group_size, N//2] uint8 (packed 4-bit zero points per N)
                 or None (default zp = 8)
    """
    import torch
    M, K = input_t.shape
    E = qweight.shape[0]
    N = scales.shape[2]
    topk = topk_ids.shape[1]
    num_valid = M * topk
    output = torch.zeros(num_valid, N, device="cpu", dtype=torch.float32)

    qw_cpu = qweight.cpu().to(torch.int16)  # promote to avoid sign issues
    scales_cpu = scales.cpu().float()

    # Unpack weights: 2 x int4 per uint8 along K dim  -> [E, K, N]
    K_packed = qweight.shape[1]  # K // 2
    w_lo = (qw_cpu & 0xF).float()          # even k indices
    w_hi = ((qw_cpu >> 4) & 0xF).float()   # odd k indices
    # Interleave: w_unpacked[:, 0::2, :] = lo, w_unpacked[:, 1::2, :] = hi
    w_unpacked = torch.zeros(E, K, N, dtype=torch.float32)
    w_unpacked[:, 0::2, :] = w_lo
    w_unpacked[:, 1::2, :] = w_hi

    # Unpack zero points
    num_groups = K // group_size
    if zeros is not None:
        zp_cpu = zeros.cpu().to(torch.int16)
        # zeros: [E, K//group_size, N//2] – packed along N dim
        zp_lo = (zp_cpu & 0xF).float()
        zp_hi = ((zp_cpu >> 4) & 0xF).float()
        zp_unpacked = torch.zeros(E, num_groups, N, dtype=torch.float32)
        zp_unpacked[:, :, 0::2] = zp_lo
        zp_unpacked[:, :, 1::2] = zp_hi
    else:
        zp_unpacked = torch.full((E, num_groups, N), 8.0)

    # Dequantize: w_float = (w_int4 - zp) * scale
    w_deq = torch.zeros(E, K, N, dtype=torch.float32)
    for gi in range(num_groups):
        k_start = gi * group_size
        k_end = k_start + group_size
        w_deq[:, k_start:k_end, :] = (
            (w_unpacked[:, k_start:k_end, :] - zp_unpacked[:, gi:gi + 1, :])
            * scales_cpu[:, gi:gi + 1, :]
        )

    for token_idx in range(M):
        x = input_t[token_idx].cpu().float()
        for k_idx in range(topk):
            flat_idx = token_idx * topk + k_idx
            expert_id = topk_ids[token_idx, k_idx].item()
            if expert_id < 0 or expert_id >= E:
                continue
            row = x @ w_deq[expert_id]
            if mul_routed_weight:
                row *= topk_weights[flat_idx].item()
            output[flat_idx] = row
    return output


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, (M, K, E, N, topk, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1

            # INT4 quantized weights packed 2 per uint8 along K dim: [E, K//2, N]
            qweight = torch.randint(0, 255, (E, K // 2, N), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            num_groups = K // group_size
            scales_t = (torch.randn(E, num_groups, N, device=device,
                                    dtype=torch.float16).abs() * 0.01 + 0.001)

            # Zero points packed 2 per uint8 along N dim: [E, K//group_size, N//2]
            zeros_t = torch.randint(0, 255, (E, num_groups, N // 2), device=device,
                                    dtype=torch.int32).to(torch.uint8)

            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device,
                                            dtype=torch.float32).abs()

            result = mod.fused_moe_gptq_awq(
                input_tensor, qweight, scales_t, zeros_t, topk_ids,
                topk_weights_flat, mul_routed_weight=True,
                group_size=group_size, use_int4=True,
            )
            torch.cuda.synchronize()

            ref = reference_fused_moe_int4(
                input_tensor, qweight, scales_t, zeros_t, topk_ids,
                topk_weights_flat, True, group_size,
            ).to(device).to(torch.float16)

            if not torch.allclose(result.float(), ref.float(), atol=1.0, rtol=0.5):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"
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
    M, K, E, N, topk, group_size = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1

    # INT4 packed weights: [E, K//2, N] uint8
    qweight = torch.randint(0, 255, (E, K // 2, N), device=device,
                            dtype=torch.int32).to(torch.uint8)
    num_groups = K // group_size
    scales_t = (torch.randn(E, num_groups, N, device=device,
                            dtype=torch.float16).abs() * 0.01 + 0.001)
    zeros_t = torch.randint(0, 255, (E, num_groups, N // 2), device=device,
                            dtype=torch.int32).to(torch.uint8)
    topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
    topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

    for _ in range(5):
        mod.fused_moe_gptq_awq(input_tensor, qweight, scales_t, zeros_t,
                                topk_ids, topk_weights_flat,
                                True, group_size, use_int4=True)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.fused_moe_gptq_awq(input_tensor, qweight, scales_t, zeros_t,
                                topk_ids, topk_weights_flat,
                                True, group_size, use_int4=True)
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
