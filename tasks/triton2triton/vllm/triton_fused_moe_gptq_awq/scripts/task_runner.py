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
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
CORRECTNESS_ATOL = 1.0
CORRECTNESS_RTOL = 0.5

# (use_int4, has_zero_points, mul_routed_weight, pass_topk_weights)
CORRECTNESS_VARIANTS = [
    (True, True, True, True),
    (True, False, True, True),
    (False, True, True, True),
    (False, False, False, False),
    (True, True, False, True),
]


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


def reference_fused_moe(input_t, qweight, scales, zeros, topk_ids,
                        topk_weights, mul_routed_weight, group_size,
                        use_int4):
    """CPU reference for INT4/INT8 (GPTQ/AWQ) weight-only MoE.

    In INT4 mode the Triton kernel packs two values per uint8 element along K:
        qweight: [E, K//2, N] uint8, low nibble = even k, high = odd k
        scales:  [E, K//group_size, N] fp16
        zeros:   [E, K//group_size, N//2] uint8 (packed 4-bit zero points per N)
                 or None (default zp = 8)

    In INT8 mode qweight is [E, K, N], zeros (when present) is
    [E, K//group_size, N], and the default zero point is 128.
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

    if use_int4:
        # Unpack weights: 2 x int4 per uint8 along K dim -> [E, K, N].
        w_lo = (qw_cpu & 0xF).float()
        w_hi = ((qw_cpu >> 4) & 0xF).float()
        w_unpacked = torch.zeros(E, K, N, dtype=torch.float32)
        w_unpacked[:, 0::2, :] = w_lo
        w_unpacked[:, 1::2, :] = w_hi
    else:
        w_unpacked = qw_cpu.float()

    # Unpack zero points
    num_groups = K // group_size
    if zeros is not None:
        zp_cpu = zeros.cpu().to(torch.int16)
        if use_int4:
            # INT4 zero points are packed along N.
            zp_lo = (zp_cpu & 0xF).float()
            zp_hi = ((zp_cpu >> 4) & 0xF).float()
            zp_unpacked = torch.zeros(E, num_groups, N, dtype=torch.float32)
            zp_unpacked[:, :, 0::2] = zp_lo
            zp_unpacked[:, :, 1::2] = zp_hi
        else:
            zp_unpacked = zp_cpu.float()
    else:
        default_zp = 8.0 if use_int4 else 128.0
        zp_unpacked = torch.full((E, num_groups, N), default_zp)

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
                if topk_weights is not None:
                    row *= topk_weights[flat_idx].item()
            output[flat_idx] = row
    return output


def _outputs_match(result, reference):
    """Apply the task tolerance while explicitly rejecting omitted writes."""
    import torch

    if not torch.count_nonzero(result).item():
        return False
    return torch.allclose(
        result.float(), reference.float(),
        atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL,
    )


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, ((M, K, E, N, topk, group_size), variant) in enumerate(
        zip(TEST_SHAPES, CORRECTNESS_VARIANTS)
    ):
        try:
            use_int4, has_zp, mul_routed_weight, pass_topk_weights = variant
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5

            packed_k = K // 2 if use_int4 else K
            qweight = torch.randint(0, 256, (E, packed_k, N), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            num_groups = K // group_size
            scales_t = (torch.randn(E, num_groups, N, device=device,
                                    dtype=torch.float16).abs() * 0.05 + 0.02)

            zeros_t = None
            if has_zp:
                zero_n = N // 2 if use_int4 else N
                zeros_t = torch.randint(
                    0, 256, (E, num_groups, zero_n), device=device,
                    dtype=torch.int32,
                ).to(torch.uint8)

            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device,
                                            dtype=torch.float32).abs()
            topk_weights_arg = topk_weights_flat if pass_topk_weights else None

            result = mod.fused_moe_gptq_awq(
                input_tensor, qweight, scales_t, zeros_t, topk_ids,
                topk_weights_arg, mul_routed_weight=mul_routed_weight,
                group_size=group_size, use_int4=use_int4,
            )
            torch.cuda.synchronize()

            ref = reference_fused_moe(
                input_tensor, qweight, scales_t, zeros_t, topk_ids,
                topk_weights_arg, mul_routed_weight, group_size, use_int4,
            ).to(device).to(torch.float16)

            if torch.allclose(
                torch.zeros_like(ref).float(), ref.float(),
                atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL,
            ):
                return False, f"Shape {i+1}: reference signal is too small"

            if not _outputs_match(result, ref):
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
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (M, K, E, N, topk, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5

            # INT4 packed weights: [E, K//2, N] uint8
            qweight = torch.randint(0, 256, (E, K // 2, N), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            num_groups = K // group_size
            scales_t = (torch.randn(E, num_groups, N, device=device,
                                    dtype=torch.float16).abs() * 0.05 + 0.02)
            zeros_t = torch.randint(0, 256, (E, num_groups, N // 2), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()

            # Build the independent reference before any measured invocation.
            reference = reference_fused_moe(
                input_tensor, qweight, scales_t, zeros_t, topk_ids,
                topk_weights_flat, True, group_size, True,
            ).to(device).to(torch.float16)
            if torch.allclose(
                torch.zeros_like(reference).float(), reference.float(),
                atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL,
            ):
                raise RuntimeError("performance reference signal is too small")
            timed_output = [None]

            def _bench_fn():
                result = mod.fused_moe_gptq_awq(
                    input_tensor, qweight, scales_t, zeros_t,
                    topk_ids, topk_weights_flat,
                    True, group_size, use_int4=True,
                )
                timed_output[0] = result
                return result

            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                use_cuda_graph=False,
                fallback_reason="fused_moe_host_routing_and_dynamic_allocations",
            )
            if timed_output[0] is None or not _outputs_match(
                timed_output[0], reference
            ):
                raise RuntimeError("timed invocation output failed validation")

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "M": M,
                    "K": K,
                    "E": E,
                    "N": N,
                    "topk": topk,
                    "group_size": group_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "E": E,
                    "N": N,
                    "topk": topk,
                    "group_size": group_size
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
