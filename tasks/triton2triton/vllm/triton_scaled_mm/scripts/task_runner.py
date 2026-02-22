#!/usr/bin/env python3
"""Task runner for triton2triton/triton_scaled_mm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_scaled_mm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_scaled_mm.py")

# Test configs: (M, K, N, per_token_scale_a, per_channel_scale_b, has_bias)
TEST_SHAPES = [
    (32, 64, 64, True, True, False),
    (64, 128, 128, True, True, True),
    (128, 256, 256, False, False, False),
    (256, 512, 512, True, True, True),
    (64, 256, 128, True, False, False),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_scaled_mm(input_t, weight, scale_a, scale_b, out_dtype, bias=None):
    """CPU reference: (input * scale_a) @ (weight * scale_b) + bias"""
    import torch
    a = input_t.float()
    b = weight.float()
    sa = scale_a.float()
    sb = scale_b.float()

    # scale_a is [M,1] or [1,1], scale_b is [N,1] or [1,1]
    # We need: (a * sa) @ (b * sb.T) but sb applies to columns of b (i.e. rows of result)
    result = (sa * a) @ b
    result = result * sb.reshape(1, -1)

    if bias is not None:
        result = result + bias.float()

    return result.to(out_dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_scaled_mm"), "Missing triton_scaled_mm"
        assert hasattr(mod, "scaled_mm_kernel"), "Missing scaled_mm_kernel"
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

    for i, (M, K, N, per_tok_a, per_ch_b, has_bias) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            input_t = torch.randn(M, K, device=device, dtype=dtype) * 0.1
            weight = torch.randn(K, N, device=device, dtype=dtype) * 0.1

            if per_tok_a:
                scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_a = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5

            if per_ch_b:
                scale_b = torch.rand(N, 1, device=device, dtype=torch.float32) * 2 + 0.5
            else:
                scale_b = torch.rand(1, 1, device=device, dtype=torch.float32) * 2 + 0.5

            bias = torch.randn(N, device=device, dtype=dtype) * 0.1 if has_bias else None

            result = mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)
            torch.cuda.synchronize()

            ref = reference_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)

            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, K={K}, N={N}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, K={K}, N={N}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    M, K, N, per_tok_a, per_ch_b, has_bias = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    input_t = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(K, N, device=device, dtype=dtype)
    scale_a = torch.rand(M, 1, device=device, dtype=torch.float32) + 0.5
    scale_b = torch.rand(N, 1, device=device, dtype=torch.float32) + 0.5
    bias = torch.randn(N, device=device, dtype=dtype) if has_bias else None

    for _ in range(5):
        mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.triton_scaled_mm(input_t, weight, scale_a, scale_b, dtype, bias=bias)
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
