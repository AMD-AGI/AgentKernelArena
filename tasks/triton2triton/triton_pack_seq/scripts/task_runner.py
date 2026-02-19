#!/usr/bin/env python3
"""Task runner for triton2triton/triton_pack_seq"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_pack_seq"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_pack_seq.py")

# Test configurations: (B, lengths_list, D)
TEST_SHAPES = [
    (4, [8, 12, 6, 10], 64),
    (2, [32, 16], 128),
    (8, [4, 8, 2, 16, 6, 10, 3, 7], 64),
    (3, [64, 32, 48], 256),
    (6, [10, 20, 15, 5, 25, 12], 128),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_pack_seq(x, lengths, pad_value=-float("inf")):
    """CPU/PyTorch reference for pack_seq."""
    import torch
    N, D = x.shape
    B = len(lengths)
    Lmax = max(lengths)

    out = torch.full((B, Lmax, D), pad_value, device=x.device, dtype=x.dtype)
    offset = 0
    for b in range(B):
        seq_len = lengths[b]
        out[b, :seq_len, :] = x[offset:offset + seq_len]
        offset += seq_len
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "pack_seq"), "Missing pack_seq"
        assert hasattr(mod, "_pack_seq_kernel"), "Missing _pack_seq_kernel"
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

    for i, (B, lengths_list, D) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            N = sum(lengths_list)
            x = torch.randn(N, D, device=device, dtype=dtype)
            lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)

            result = mod.pack_seq(x, lengths, pad_value=0.0)
            torch.cuda.synchronize()

            ref = reference_pack_seq(x, lengths_list, pad_value=0.0)

            if not torch.allclose(result, ref, atol=1e-3, rtol=1e-3):
                max_diff = (result - ref).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, D={D}): max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (B={B}, D={D}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    B, lengths_list, D = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    N = sum(lengths_list)
    x = torch.randn(N, D, device=device, dtype=dtype)
    lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)

    for _ in range(5):
        mod.pack_seq(x, lengths, pad_value=0.0)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.pack_seq(x, lengths, pad_value=0.0)
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
        B, lengths_list, D = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {"B": B, "D": D, "lengths": lengths_list},
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
