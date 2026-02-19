#!/usr/bin/env python3
"""Task runner for triton2triton/triton_silu_mul_quant_fp8"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_silu_mul_quant_fp8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_silu_mul_quant_fp8.py")

# Test configs: (M, N) where N must be divisible by 256 (GROUP_SIZE*2), M by 128
TEST_SHAPES = [
    (128, 256),
    (128, 512),
    (256, 512),
    (256, 1024),
    (512, 1024),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_silu_mul_quant_fp8(input_t, fp8_dtype):
    """CPU reference: silu(x[:,:N/2]) * x[:,N/2:], then quantize per group."""
    import torch
    GROUP_SIZE = 128
    M, N = input_t.shape
    N_2 = N // 2

    x = input_t.cpu().float()
    gate = x[:, :N_2]
    up = x[:, N_2:]

    # SiLU + mul
    silu_out = gate / (1.0 + torch.exp(-gate))
    y = silu_out * up

    if fp8_dtype == torch.float8_e4m3fnuz:
        fp8_min, fp8_max = -240.0, 240.0
    else:
        finfo = torch.finfo(fp8_dtype)
        fp8_min, fp8_max = finfo.min, finfo.max

    num_groups = N_2 // GROUP_SIZE
    y_q = torch.zeros_like(y)
    # Column-major scales: shape [M, num_groups]
    y_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * GROUP_SIZE
            end = start + GROUP_SIZE
            group = y[row, start:end]
            absmax = max(group.abs().max().item(), 1e-10)
            scale = absmax / fp8_max
            y_s[row, g] = scale
            y_q[row, start:end] = (group / scale).clamp(fp8_min, fp8_max)

    return y_q.to(fp8_dtype), y_s


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing silu_mul_per_token_group_quant_fp8_colmajor"
        assert hasattr(mod, "_silu_mul_per_token_group_quant_fp8_colmajor"), \
            "Missing _silu_mul_per_token_group_quant_fp8_colmajor"
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
    fp8_dtype = mod._get_fp8_dtype()

    for i, (M, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16)

            y_q, y_s = mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_silu_mul_quant_fp8(x, fp8_dtype)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            # Check scales
            if not torch.allclose(y_s, ref_s, atol=1e-2, rtol=1e-1):
                max_diff = (y_s - ref_s).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}): scale max diff = {max_diff:.6f}"
                )

            # Check quantized via dequant
            N_2 = N // 2
            GROUP_SIZE = 128
            y_dq = y_q.float() * y_s.repeat_interleave(GROUP_SIZE, dim=-1)
            ref_dq = ref_q.float().to(device) * ref_s.repeat_interleave(GROUP_SIZE, dim=-1)

            if not torch.allclose(y_dq, ref_dq, atol=5e-1, rtol=1e-1):
                max_diff = (y_dq - ref_dq).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}): dequant max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, N={N}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    M, N = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float16)

    for _ in range(5):
        mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.silu_mul_per_token_group_quant_fp8_colmajor(x)
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
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
