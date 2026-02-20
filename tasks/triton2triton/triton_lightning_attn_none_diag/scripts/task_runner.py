#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lightning_attn_none_diag"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_lightning_attn_none_diag"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lightning_attn_none_diag.py")

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


def reference_none_diag(q, o_diag, s, kv, BLOCK=256, CBLOCK=64):
    """
    PyTorch reference for non-diagonal (cross-block) attention.
    q: [B, H, N, D], o_diag: [B, H, N, E], s: [H], kv: [B, H, NUM_BLOCK, D, E]
    Returns: o [B, H, N, E] = o_diag + cross-block contribution
    """
    import torch
    B, H, N, D = q.shape
    E = o_diag.shape[-1]
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK
    NUM_CBLOCK = BLOCK // CBLOCK

    o = o_diag.clone().float()

    for b_idx in range(B):
        for h_idx in range(H):
            slope = s[h_idx].float().item()
            for blk in range(NUM_BLOCK):
                for ci in range(NUM_CBLOCK):
                    block_offset = blk * BLOCK + ci * CBLOCK
                    q_end = min(block_offset + CBLOCK, N)
                    q_len = q_end - block_offset
                    if q_len <= 0:
                        break

                    q_slice = q[b_idx, h_idx, block_offset:q_end].float()  # [q_len, D]
                    kv_slice = kv[b_idx, h_idx, blk].float()  # [D, E]

                    c_array = torch.arange(q_len, device=q.device, dtype=torch.float32)
                    q_decay = torch.exp(-slope * (ci * CBLOCK + c_array)).unsqueeze(1)  # [q_len, 1]

                    contrib = (q_slice @ kv_slice) * q_decay  # [q_len, E]
                    o[b_idx, h_idx, block_offset:q_end] += contrib

    return o


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lightning_attn_none_diag_forward"), "Missing lightning_attn_none_diag_forward"
        assert hasattr(mod, "_fwd_none_diag_kernel"), "Missing _fwd_none_diag_kernel"
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
    dtype = torch.float16

    for i, (B, H, N, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            BLOCK = 256
            CBLOCK = 64
            NUM_BLOCK = (N + BLOCK - 1) // BLOCK
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01

            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            # Simulate diagonal attention output
            o_diag = torch.randn(B, H, N, E, device=device, dtype=dtype) * 0.1
            # Simulate prefix-summed KV state
            kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32) * 0.01

            # Reference
            ref = reference_none_diag(q, o_diag, s, kv, BLOCK, CBLOCK)
            ref = ref.to(dtype)

            # Run Triton kernel (modifies o in-place)
            o_triton = o_diag.clone()
            mod.lightning_attn_none_diag_forward(q, o_triton, s, kv, BLOCK, CBLOCK)
            torch.cuda.synchronize()

            if not torch.allclose(o_triton, ref, atol=1e-2, rtol=1e-2):
                max_diff = (o_triton.float() - ref.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, N={N}, D={D}, E={E}): "
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
    dtype = torch.float16
    B, H, N, D, E = TEST_SHAPES[PERF_SHAPE_IDX]
    BLOCK = 256
    CBLOCK = 64
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    torch.manual_seed(0)
    s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    o = torch.randn(B, H, N, E, device=device, dtype=dtype) * 0.1
    kv = torch.randn(B, H, NUM_BLOCK, D, E, device=device, dtype=torch.float32) * 0.01

    # Warmup
    for _ in range(5):
        o_c = o.clone()
        mod.lightning_attn_none_diag_forward(q, o_c, s, kv, BLOCK, CBLOCK)
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        o_c = o.clone()
        start_events[j].record()
        mod.lightning_attn_none_diag_forward(q, o_c, s, kv, BLOCK, CBLOCK)
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
