#!/usr/bin/env python3
"""Task runner for triton_batch_memcpy"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_batch_memcpy"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_batch_memcpy.py")

# (batch, size_per_copy_bytes)
TEST_SHAPES = [
    (4, 256),
    (8, 1024),
    (16, 4096),
    (32, 2048),
    (64, 8192),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(batch, size_bytes, device="cuda"):
    import torch
    torch.manual_seed(42)
    # Allocate source and destination as flat byte tensors
    srcs = []
    dsts = []
    src_ptrs_list = []
    dst_ptrs_list = []
    sizes_list = []

    for i in range(batch):
        # Use a slightly varying size for more realistic testing
        sz = size_bytes + i * 16
        src = torch.randint(0, 256, (sz,), dtype=torch.uint8, device=device)
        dst = torch.zeros(sz, dtype=torch.uint8, device=device)
        srcs.append(src)
        dsts.append(dst)
        src_ptrs_list.append(src.data_ptr())
        dst_ptrs_list.append(dst.data_ptr())
        sizes_list.append(sz)

    src_ptrs = torch.tensor(src_ptrs_list, dtype=torch.int64, device=device)
    dst_ptrs = torch.tensor(dst_ptrs_list, dtype=torch.int64, device=device)
    sizes = torch.tensor(sizes_list, dtype=torch.int32, device=device)

    return srcs, dsts, src_ptrs, dst_ptrs, sizes


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "batch_memcpy"), "Missing batch_memcpy"
        assert hasattr(mod, "batch_memcpy_kernel"), "Missing kernel"
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
    for i, (batch, sz) in enumerate(TEST_SHAPES):
        try:
            srcs, dsts, src_ptrs, dst_ptrs, sizes = make_inputs(batch, sz, device)
            mod.batch_memcpy(src_ptrs, dst_ptrs, sizes)
            torch.cuda.synchronize()

            for j in range(batch):
                if not torch.equal(dsts[j], srcs[j]):
                    return False, f"Shape {i+1}: copy {j} mismatch"
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
    batch, sz = TEST_SHAPES[PERF_SHAPE_IDX]
    srcs, dsts, src_ptrs, dst_ptrs, sizes = make_inputs(batch, sz, device)

    for _ in range(5):
        mod.batch_memcpy(src_ptrs, dst_ptrs, sizes)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.batch_memcpy(src_ptrs, dst_ptrs, sizes)
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
