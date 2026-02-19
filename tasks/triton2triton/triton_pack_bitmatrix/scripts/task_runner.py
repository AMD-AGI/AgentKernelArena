#!/usr/bin/env python3
"""Task runner for triton2triton/triton_pack_bitmatrix"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_pack_bitmatrix"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_pack_bitmatrix.py")

# (n_rows, num_experts, topk)
TEST_SHAPES = [
    (32, 8, 2),
    (64, 16, 2),
    (128, 32, 2),
    (256, 64, 4),
    (512, 64, 2),
]
PERF_SHAPE_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_pack_bitmatrix(topk_ids, num_experts):
    """CPU reference: pack topk_ids into bitmatrix.

    Mirrors the Triton kernel exactly, including its treatment of padding
    slots.  The kernel loads BLOCK_SIZE_K (=32) entries per row; positions
    beyond the real topk are filled with -1 (the ``other`` value of
    ``tl.load``).  Because Triton uses C-style truncated integer division,
    -1 // 32 == 0 and -1 % 32 == -1.  The hardware shift ``1u << -1``
    (i.e. ``1u << 31``) then sets bit-31 in column 0 for every row.  The
    reference must reproduce this behaviour to match the kernel output.
    """
    import torch
    n_rows, topk = topk_ids.shape
    BLOCK_SIZE_K = 32
    bm_cols = (num_experts + 31) // 32
    bitmatrix = torch.zeros(n_rows, bm_cols, dtype=torch.uint32)
    for row in range(n_rows):
        for k in range(BLOCK_SIZE_K):
            eid = topk_ids[row, k].item() if k < topk else -1
            if eid >= 0:
                col = eid // 32
                bit = eid % 32
            else:
                # C-style truncated division: -1 / 32 == 0, -1 % 32 == -1
                # Hardware: uint32(1) << -1 wraps to 1 << 31
                col = 0
                bit = 31
            if 0 <= col < bm_cols:
                bitmatrix[row, col] = bitmatrix[row, col].item() | (1 << bit)
    return bitmatrix


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "pack_topk_to_bitmatrix"), "Missing pack_topk_to_bitmatrix"
        assert hasattr(mod, "pack_bitmatrix"), "Missing pack_bitmatrix"
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
    for i, (n_rows, num_experts, topk) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            topk_ids = torch.randint(0, num_experts, (n_rows, topk), device=device, dtype=torch.int16)

            result = mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
            torch.cuda.synchronize()

            ref = reference_pack_bitmatrix(topk_ids.cpu(), num_experts).to(device)
            if not torch.equal(result, ref):
                diff_count = (result != ref).sum().item()
                return False, f"Shape {i+1}: {diff_count} mismatched elements"
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
    n_rows, num_experts, topk = TEST_SHAPES[PERF_SHAPE_IDX]
    torch.manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (n_rows, topk), device=device, dtype=torch.int16)

    for _ in range(5):
        mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        start_events[j].record()
        mod.pack_topk_to_bitmatrix(topk_ids, num_experts)
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
