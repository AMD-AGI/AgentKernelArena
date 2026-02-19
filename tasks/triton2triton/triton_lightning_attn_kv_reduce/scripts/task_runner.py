#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lightning_attn_kv_reduce"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_lightning_attn_kv_reduce"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lightning_attn_kv_reduce.py")

# Test configurations: (batch, heads, seq, d_model, e_model)
TEST_SHAPES = [
    (1, 4, 64, 64, 64),
    (2, 8, 128, 64, 64),
    (1, 4, 256, 128, 128),
    (2, 4, 128, 64, 32),
    (4, 2, 64, 32, 32),
]
PERF_SHAPE_IDX = 2


def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_kv_reduce(s, kv, kv_history, n, BLOCK=256):
    """
    PyTorch reference for KV reduce (prefix sum with decay).
    s: [H], kv: [B, H, NUM_BLOCK, D, E], kv_history: [B, H, D, E]
    Returns: (kv_reduced, kv_history_updated) -- both modified
    """
    import torch
    B, H, NUM_BLOCK, D, E = kv.shape
    kv_out = kv.clone()
    kv_hist_out = kv_history.clone()

    for b_idx in range(B):
        for h_idx in range(H):
            slope = s[h_idx].item()
            kv_pre = kv_hist_out[b_idx, h_idx].float().clone()

            for blk in range(NUM_BLOCK):
                block_size = min(n - blk * BLOCK, BLOCK)
                block_decay = float(torch.exp(torch.tensor(-slope * block_size)))

                kv_cur = kv_out[b_idx, h_idx, blk].float().clone()
                kv_out[b_idx, h_idx, blk] = kv_pre
                kv_pre = block_decay * kv_pre + kv_cur

            kv_hist_out[b_idx, h_idx] = kv_pre

    return kv_out, kv_hist_out


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lightning_attn_kv_reduce_forward"), "Missing lightning_attn_kv_reduce_forward"
        assert hasattr(mod, "_fwd_kv_reduce"), "Missing _fwd_kv_reduce"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    """Run correctness checks against PyTorch reference."""
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"

    for i, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            BLOCK = 256
            NUM_BLOCK = (N + BLOCK - 1) // BLOCK
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01

            kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32)
            kv_history = torch.randn(B, H, D, E, device=device, dtype=torch.float32) * 0.1

            # Reference (on clones)
            ref_kv, ref_hist = reference_kv_reduce(s, kv.clone(), kv_history.clone(), N, BLOCK)

            # Run Triton kernel (modifies in-place)
            kv_triton = kv.clone()
            hist_triton = kv_history.clone()
            mod.lightning_attn_kv_reduce_forward(s, kv_triton, hist_triton, N, BLOCK)
            torch.cuda.synchronize()

            if not torch.allclose(kv_triton, ref_kv, atol=1e-2, rtol=1e-2):
                max_diff = (kv_triton - ref_kv).abs().max().item()
                return False, (
                    f"Shape {i+1} KV mismatch (B={B}, H={H}, N={N}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )
            if not torch.allclose(hist_triton, ref_hist, atol=1e-2, rtol=1e-2):
                max_diff = (hist_triton - ref_hist).abs().max().item()
                return False, (
                    f"Shape {i+1} History mismatch (B={B}, H={H}, N={N}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (B={B}, H={H}, N={N}, D={D}, E={E}): exception: {e}"
            )

    return True, None


def run_performance():
    """Measure kernel execution time."""
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    B, H, N, D, E = TEST_SHAPES[PERF_SHAPE_IDX]
    BLOCK = 256
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    torch.manual_seed(0)
    s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
    kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32)
    kv_history = torch.zeros(B, H, D, E, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        kv_c = kv.clone()
        h_c = kv_history.clone()
        mod.lightning_attn_kv_reduce_forward(s, kv_c, h_c, N, BLOCK)
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        kv_c = kv.clone()
        h_c = kv_history.clone()
        start_events[j].record()
        mod.lightning_attn_kv_reduce_forward(s, kv_c, h_c, N, BLOCK)
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        elapsed_ms = run_performance()
        B, H, N, D, E = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {"batch": B, "heads": H, "seq": N, "d_model": D, "e_model": E},
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
