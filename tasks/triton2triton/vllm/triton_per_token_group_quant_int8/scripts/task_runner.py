#!/usr/bin/env python3
"""Task runner for triton2triton/triton_per_token_group_quant_int8"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_per_token_group_quant_int8"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_per_token_group_quant_int8.py")

# Test configs: (M, N, group_size)
TEST_SHAPES = [
    (32, 128, 128),
    (64, 256, 128),
    (128, 512, 128),
    (256, 1024, 128),
    (64, 512, 64),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_per_token_group_quant_int8(x, group_size, eps=1e-10):
    """CPU reference for per-token-group INT8 quantization."""
    import torch
    M, N = x.shape
    x_cpu = x.cpu().float()

    int8_max = 127
    int8_min = -128

    num_groups = N // group_size
    x_q = torch.zeros((M, N), dtype=torch.int8)
    x_s = torch.zeros(M, num_groups, dtype=torch.float32)

    for row in range(M):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = x_cpu[row, start:end]
            absmax = max(group.abs().max().item(), eps)
            scale = absmax / int8_max
            x_s[row, g] = scale
            x_q[row, start:end] = (group / scale).clamp(int8_min, int8_max).to(torch.int8)

    return x_q, x_s


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "per_token_group_quant_int8"), "Missing per_token_group_quant_int8"
        assert hasattr(mod, "_per_token_group_quant_int8"), "Missing _per_token_group_quant_int8"
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

    for i, (M, N, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()

            x_q, x_s = mod.per_token_group_quant_int8(x, group_size)
            torch.cuda.synchronize()

            ref_q, ref_s = reference_per_token_group_quant_int8(x, group_size)
            ref_q = ref_q.to(device)
            ref_s = ref_s.to(device)

            # Check scales
            if not torch.allclose(x_s, ref_s, atol=1e-4, rtol=1e-3):
                max_diff = (x_s - ref_s).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}, G={group_size}): "
                    f"scale max diff = {max_diff:.6f}"
                )

            # Check quantized values
            if not torch.allclose(x_q.float(), ref_q.float(), atol=1.0, rtol=0.0):
                max_diff = (x_q.float() - ref_q.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}, G={group_size}): "
                    f"quant max diff = {max_diff:.1f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (M={M}, N={N}, G={group_size}): exception: {e}"
            )

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    M, N, group_size = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()

    for _ in range(5):
        mod.per_token_group_quant_int8(x, group_size)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.per_token_group_quant_int8(x, group_size)
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
