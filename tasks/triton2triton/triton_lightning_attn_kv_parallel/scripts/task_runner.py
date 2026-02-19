#!/usr/bin/env python3
"""Task runner for triton2triton/triton_lightning_attn_kv_parallel"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_lightning_attn_kv_parallel"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_lightning_attn_kv_parallel.py")

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


def reference_kv_parallel(k, v, s, BLOCK=256, CBLOCK=64):
    """
    PyTorch reference for parallel KV outer product with decay.
    k: [B, H, N, D], v: [B, H, N, E], s: [H]
    Returns: kv [B, H, NUM_BLOCK, D, E]
    """
    import torch
    B, H, N, D = k.shape
    E = v.shape[-1]
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    kv_out = torch.zeros(B, H, NUM_BLOCK, D, E, dtype=torch.float32, device=k.device)

    # Precompute decay factors
    array = torch.arange(0, BLOCK, device=k.device).float() + 1
    k_decay_all = torch.exp(-s.view(-1, 1) * (BLOCK - array.view(1, -1)))  # [H, BLOCK]

    NUM_CBLOCK = BLOCK // CBLOCK

    for b_idx in range(B):
        for h_idx in range(H):
            for blk in range(NUM_BLOCK):
                block_start = blk * BLOCK
                block_end = min(block_start + BLOCK, N)
                split_n = block_end - block_start

                left_shift = ((split_n + CBLOCK - 1) // CBLOCK) * CBLOCK - split_n
                num_cblocks = min((split_n + CBLOCK - 1) // CBLOCK, NUM_CBLOCK)

                acc = torch.zeros(D, E, dtype=torch.float32, device=k.device)
                decay_offset = (NUM_CBLOCK - num_cblocks) * CBLOCK

                for j in range(num_cblocks):
                    left_bound = (1 - j) * left_shift if j == 0 else 0
                    # Actual positions in the block
                    pos_start = block_start + j * CBLOCK - left_shift
                    pos_end = pos_start + CBLOCK

                    k_slice = torch.zeros(CBLOCK, D, dtype=torch.float32, device=k.device)
                    v_slice = torch.zeros(CBLOCK, E, dtype=torch.float32, device=k.device)

                    for idx in range(CBLOCK):
                        if idx >= left_bound:
                            src_pos = pos_start + idx
                            if 0 <= src_pos < N:
                                k_slice[idx] = k[b_idx, h_idx, src_pos].float()
                                v_slice[idx] = v[b_idx, h_idx, src_pos].float()

                    decay_idx = decay_offset + j * CBLOCK
                    k_d = k_decay_all[h_idx, decay_idx:decay_idx + CBLOCK]

                    # k_trans [D, CBLOCK] * decay [1, CBLOCK] -> weighted k_trans
                    k_trans = k_slice.T  # [D, CBLOCK]
                    k_trans_weighted = k_trans * k_d.unsqueeze(0)
                    acc += k_trans_weighted @ v_slice  # [D, E]

                kv_out[b_idx, h_idx, blk] = acc

    return kv_out


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "lightning_attn_kv_parallel_forward"), "Missing lightning_attn_kv_parallel_forward"
        assert hasattr(mod, "_fwd_kv_parallel"), "Missing _fwd_kv_parallel"
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
            s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01

            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, E, device=device, dtype=dtype)

            # Run Triton kernel
            out = mod.lightning_attn_kv_parallel_forward(k, v, s, N)
            torch.cuda.synchronize()

            # Run reference
            ref = reference_kv_parallel(k, v, s)

            if not torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
                max_diff = (out - ref).abs().max().item()
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

    torch.manual_seed(0)
    s = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, E, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        mod.lightning_attn_kv_parallel_forward(k, v, s, N)
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.lightning_attn_kv_parallel_forward(k, v, s, N)
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
