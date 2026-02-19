#!/usr/bin/env python3
"""Task runner for triton2triton/triton_mean"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_mean"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_mean.py")

# Test configurations: (shape, dim)
TEST_SHAPES = [
    ((32, 128), 1),
    ((64, 256, 64), 1),
    ((16, 512), 1),
    ((8, 1024, 32), 1),
    ((128, 64), 0),
]
PERF_SHAPE_IDX = 3


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
        assert hasattr(mod, "mean_dim"), "Missing mean_dim"
        assert hasattr(mod, "mean_kernel"), "Missing mean_kernel"
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

    for i, (shape, dim) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(shape, device=device, dtype=dtype)

            result = mod.mean_dim(x, dim=dim)
            torch.cuda.synchronize()

            ref = x.float().mean(dim=dim).to(dtype)

            if not torch.allclose(result, ref, atol=1e-2, rtol=1e-2):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i + 1} ({shape}, dim={dim}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i + 1} ({shape}, dim={dim}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    shape, dim = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=dtype)

    for _ in range(5):
        mod.mean_dim(x, dim=dim)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.mean_dim(x, dim=dim)
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
        shape, dim = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {"execution_time_ms": elapsed_ms, "shape": {"shape": list(shape), "dim": dim}}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
