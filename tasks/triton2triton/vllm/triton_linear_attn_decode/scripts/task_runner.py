#!/usr/bin/env python3
"""Task runner for triton2triton/triton_linear_attn_decode"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_linear_attn_decode"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_linear_attn_decode.py")

# Test configurations: (batch, heads, d_model, e_model)
TEST_SHAPES = [
    (1, 4, 64, 64),
    (4, 8, 64, 64),
    (2, 16, 128, 128),
    (8, 4, 64, 32),
    (1, 32, 64, 64),
]
PERF_SHAPE_IDX = 2


def load_module():
    """Dynamically load the source module."""
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_linear_attn_decode(q, k, v, kv_caches, slope_rate, slot_idx):
    """
    PyTorch reference for linear attention decode.
    q, k, v: [B, H, 1, D]
    kv_caches: [num_slots, H, D, D_v]
    slope_rate: [H]
    slot_idx: [B]
    Returns: output [B, H, 1, D_v], updated kv_caches
    """
    import torch
    B, H, _, D = q.shape
    D_v = v.shape[-1]

    output = torch.zeros(B, H, 1, D_v, dtype=torch.float32, device=q.device)

    for b_idx in range(B):
        sid = slot_idx[b_idx].item()
        if sid == -1:
            continue

        for h_idx in range(H):
            ratio = torch.exp(-slope_rate[h_idx])

            q_vec = q[b_idx, h_idx, 0].float()   # [D]
            k_vec = k[b_idx, h_idx, 0].float()   # [D]
            v_vec = v[b_idx, h_idx, 0].float()   # [D_v]

            kv_outer = k_vec[:, None] * v_vec[None, :]  # [D, D_v]
            kv_old = kv_caches[sid, h_idx].float()       # [D, D_v]
            kv_new = kv_outer + ratio * kv_old

            out = (q_vec[:, None] * kv_new).sum(dim=0)   # [D_v]

            kv_caches[sid, h_idx] = kv_new
            output[b_idx, h_idx, 0] = out

    return output


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "linear_attn_decode_forward"), "Missing linear_attn_decode_forward"
        assert hasattr(mod, "_linear_attn_decode_kernel"), "Missing _linear_attn_decode_kernel"
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

    for i, (B, H, D, E) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            slope_rate = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
            slot_idx = torch.arange(B, device=device, dtype=torch.int32)

            q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
            k = torch.randn(B, H, 1, D, device=device, dtype=dtype)
            v = torch.randn(B, H, 1, E, device=device, dtype=dtype)

            num_slots = B
            kv_caches_triton = torch.randn(num_slots, H, D, E, device=device, dtype=dtype) * 0.1
            kv_caches_ref = kv_caches_triton.clone().float()

            # Reference
            ref_out = reference_linear_attn_decode(q, k, v, kv_caches_ref, slope_rate, slot_idx)

            # Triton kernel
            triton_out = mod.linear_attn_decode_forward(
                q, k, v, kv_caches_triton, slope_rate, slot_idx
            )
            torch.cuda.synchronize()

            # Compare output
            if not torch.allclose(triton_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
                max_diff = (triton_out.float() - ref_out.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} output (B={B}, H={H}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )

            # Compare updated KV cache
            if not torch.allclose(kv_caches_triton.float(), kv_caches_ref.float(), atol=1e-2, rtol=1e-2):
                max_diff = (kv_caches_triton.float() - kv_caches_ref.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} kv_cache (B={B}, H={H}, D={D}, E={E}): "
                    f"max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, (
                f"Shape {i+1} (B={B}, H={H}, D={D}, E={E}): exception: {e}"
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
    B, H, D, E = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    slope_rate = torch.rand(H, device=device, dtype=torch.float32) * 0.1 + 0.01
    slot_idx = torch.arange(B, device=device, dtype=torch.int32)
    q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    k = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    v = torch.randn(B, H, 1, E, device=device, dtype=dtype)
    kv_caches = torch.randn(B, H, D, E, device=device, dtype=dtype) * 0.1

    # Warmup
    for _ in range(5):
        kv_c = kv_caches.clone()
        mod.linear_attn_decode_forward(q, k, v, kv_c, slope_rate, slot_idx)
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        kv_c = kv_caches.clone()
        start_events[j].record()
        mod.linear_attn_decode_forward(q, k, v, kv_c, slope_rate, slot_idx)
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
        B, H, D, E = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {"batch": B, "heads": H, "d_model": D, "e_model": E},
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
