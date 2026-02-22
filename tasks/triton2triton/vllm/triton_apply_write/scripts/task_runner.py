#!/usr/bin/env python3
"""Task runner for triton2triton/triton_apply_write"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_apply_write"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_apply_write.py")

# (num_writes, row_size, avg_content_len)
TEST_SHAPES = [
    (4, 128, 8),
    (8, 256, 16),
    (16, 512, 32),
    (32, 1024, 64),
    (64, 2048, 128),
]
PERF_SHAPE_IDX = 3


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_apply_write(output, write_indices, write_starts, write_contents, write_cu_lens):
    import torch
    output = output.clone()
    n = write_indices.shape[0]
    for i in range(n):
        row_idx = write_indices[i].item()
        start_idx = write_starts[i].item()
        cu_start = write_cu_lens[i - 1].item() if i > 0 else 0
        cu_end = write_cu_lens[i].item()
        content_len = cu_end - cu_start
        for j in range(content_len):
            output[row_idx, start_idx + j] = write_contents[cu_start + j]
    return output


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "apply_write")
        assert hasattr(mod, "_apply_write_kernel")
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
    for i, (num_writes, row_size, avg_content_len) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            num_rows = num_writes + 8
            output = torch.zeros(num_rows, row_size, dtype=torch.int32, device=device)

            # Use unique row indices to avoid race conditions in parallel writes
            write_indices = torch.randperm(num_rows, device=device, dtype=torch.int32)[:num_writes]
            write_starts = torch.randint(0, row_size // 2, (num_writes,), dtype=torch.int32, device=device)

            # Generate content lengths and cumulative lengths
            content_lens = torch.randint(1, avg_content_len + 1, (num_writes,), dtype=torch.int32)
            # Ensure content fits in row
            for w in range(num_writes):
                max_len = row_size - write_starts[w].item()
                content_lens[w] = min(content_lens[w].item(), max_len)
            cu_lens = torch.cumsum(content_lens, dim=0).to(torch.int32).to(device)
            total_content = int(cu_lens[-1].item())
            write_contents = torch.randint(1, 10000, (total_content,), dtype=torch.int32, device=device)

            output_gpu = output.clone()
            mod.apply_write(output_gpu, write_indices, write_starts, write_contents, cu_lens)
            torch.cuda.synchronize()

            ref = reference_apply_write(
                output.cpu(), write_indices.cpu(), write_starts.cpu(),
                write_contents.cpu(), cu_lens.cpu(),
            )

            if not torch.equal(output_gpu.cpu(), ref):
                return False, f"Shape {i+1}: mismatch"
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
    num_writes, row_size, avg_content_len = TEST_SHAPES[PERF_SHAPE_IDX]
    num_rows = num_writes + 8

    torch.manual_seed(0)
    output = torch.zeros(num_rows, row_size, dtype=torch.int32, device=device)
    write_indices = torch.arange(num_writes, dtype=torch.int32, device=device)
    write_starts = torch.zeros(num_writes, dtype=torch.int32, device=device)
    cu_lens = torch.arange(1, num_writes + 1, dtype=torch.int32, device=device) * avg_content_len
    total_content = int(cu_lens[-1].item())
    write_contents = torch.randint(1, 10000, (total_content,), dtype=torch.int32, device=device)

    for _ in range(5):
        mod.apply_write(output.clone(), write_indices, write_starts, write_contents, cu_lens)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.apply_write(output.clone(), write_indices, write_starts, write_contents, cu_lens)
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
