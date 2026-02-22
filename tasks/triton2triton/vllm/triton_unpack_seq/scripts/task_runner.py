#!/usr/bin/env python3
"""Task runner for triton2triton/triton_unpack_seq"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_unpack_seq"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_unpack_seq.py")

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


def reference_unpack_seq(packed, lengths_list):
    """CPU/PyTorch reference for unpack_seq."""
    import torch
    B, Lmax, D = packed.shape
    N = sum(lengths_list)
    out = torch.empty(N, D, device=packed.device, dtype=packed.dtype)
    offset = 0
    for b in range(B):
        seq_len = lengths_list[b]
        out[offset:offset + seq_len] = packed[b, :seq_len]
        offset += seq_len
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "unpack_seq"), "Missing unpack_seq"
        assert hasattr(mod, "_unpack_seq_triton_kernel"), "Missing _unpack_seq_triton_kernel"
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

            Lmax = max(lengths_list)
            packed = torch.randn(B, Lmax, D, device=device, dtype=dtype)
            lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)

            result = mod.unpack_seq(packed, lengths)
            torch.cuda.synchronize()

            ref = reference_unpack_seq(packed, lengths_list)

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
    Lmax = max(lengths_list)
    packed = torch.randn(B, Lmax, D, device=device, dtype=dtype)
    lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)

    for _ in range(5):
        mod.unpack_seq(packed, lengths)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.unpack_seq(packed, lengths)
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
