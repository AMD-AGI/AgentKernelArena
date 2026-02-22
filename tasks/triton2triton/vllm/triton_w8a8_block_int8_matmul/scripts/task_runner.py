#!/usr/bin/env python3
"""Task runner for triton2triton/triton_w8a8_block_int8_matmul"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_w8a8_block_int8_matmul"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_w8a8_block_int8_matmul.py")

# Test configs: (M, N, K, block_n, block_k)
TEST_SHAPES = [
    (64, 128, 128, 128, 128),
    (128, 256, 256, 128, 128),
    (64, 128, 256, 128, 128),
    (256, 512, 512, 128, 128),
    (128, 256, 512, 128, 128),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    """CPU reference: block-wise dequantize INT8 then matmul."""
    import torch
    block_n, block_k = block_size
    M, K = A.shape
    N = B.shape[0]

    A_f = A.cpu().float()
    B_f = B.cpu().float()
    As_f = As.cpu().float()
    Bs_f = Bs.cpu().float()

    # Dequantize A
    A_dq = torch.zeros_like(A_f)
    for m in range(M):
        for kg in range(As_f.shape[1]):
            start_k = kg * block_k
            end_k = min(start_k + block_k, K)
            A_dq[m, start_k:end_k] = A_f[m, start_k:end_k] * As_f[m, kg]

    # Dequantize B
    B_dq = torch.zeros_like(B_f)
    for ng in range(Bs_f.shape[0]):
        for kg in range(Bs_f.shape[1]):
            start_n = ng * block_n
            end_n = min(start_n + block_n, N)
            start_k = kg * block_k
            end_k = min(start_k + block_k, K)
            B_dq[start_n:end_n, start_k:end_k] = B_f[start_n:end_n, start_k:end_k] * Bs_f[ng, kg]

    result = A_dq @ B_dq.T
    return result.to(output_dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "w8a8_block_int8_matmul"), "Missing w8a8_block_int8_matmul"
        assert hasattr(mod, "_w8a8_block_int8_matmul"), "Missing _w8a8_block_int8_matmul"
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

    for i, (M, N, K, block_n, block_k) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            import triton as _triton

            # Create INT8 tensors
            A = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
            B = torch.randint(-128, 127, (N, K), device=device, dtype=torch.int8)

            # Scales
            As = torch.rand(M, _triton.cdiv(K, block_k), device=device, dtype=torch.float32) * 0.1 + 0.01
            Bs = torch.rand(_triton.cdiv(N, block_n), _triton.cdiv(K, block_k),
                           device=device, dtype=torch.float32) * 0.1 + 0.01

            result = mod.w8a8_block_int8_matmul(
                A, B, As, Bs, [block_n, block_k], output_dtype=torch.float16
            )
            torch.cuda.synchronize()

            ref = reference_w8a8_block_int8_matmul(
                A, B, As, Bs, [block_n, block_k], torch.float16
            ).to(device)

            if not torch.allclose(result, ref, atol=1e-1, rtol=1e-1):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (M={M}, N={N}, K={K}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (M={M}, N={N}, K={K}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    M, N, K, block_n, block_k = TEST_SHAPES[PERF_SHAPE_IDX]
    import triton as _triton

    torch.manual_seed(0)
    A = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    B = torch.randint(-128, 127, (N, K), device=device, dtype=torch.int8)
    As = torch.rand(M, _triton.cdiv(K, block_k), device=device, dtype=torch.float32) + 0.01
    Bs = torch.rand(_triton.cdiv(N, block_n), _triton.cdiv(K, block_k),
                    device=device, dtype=torch.float32) + 0.01

    for _ in range(5):
        mod.w8a8_block_int8_matmul(A, B, As, Bs, [block_n, block_k])
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.w8a8_block_int8_matmul(A, B, As, Bs, [block_n, block_k])
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
