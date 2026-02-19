#!/usr/bin/env python3
"""Task runner for triton2triton/triton_correct_attn_cp_out"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_correct_attn_cp_out"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_correct_attn_cp_out.py")

# Test configurations: (B, H, D, N)
# B=batch, H=heads, D=head_dim, N=num_ranks
TEST_SHAPES = [
    (4, 8, 64, 2),
    (8, 16, 64, 4),
    (16, 32, 128, 2),
    (2, 8, 128, 8),
    (32, 16, 64, 4),
]
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_correct_attn_cp_out(out, lses, lse_idx, is_base_e=True):
    """CPU/PyTorch reference for correct_attn_cp_out."""
    import torch
    import math

    B, H, D = out.shape
    N = lses.shape[0]

    # Compute global LSE over N ranks for each (B, H)
    # lses: [N, B, H]
    lses_float = lses.float()

    # Handle NaN and inf
    lses_clean = torch.where(
        torch.isnan(lses_float) | (lses_float == float("inf")),
        torch.tensor(-float("inf"), device=lses.device, dtype=torch.float32),
        lses_float,
    )

    lse_max = lses_clean.max(dim=0).values  # [B, H]
    lse_max = torch.where(lse_max == -float("inf"), torch.zeros_like(lse_max), lse_max)

    shifted = lses_clean - lse_max.unsqueeze(0)

    if is_base_e:
        exp_vals = torch.exp(shifted)
        acc = exp_vals.sum(dim=0)
        global_lse = torch.log(acc) + lse_max
    else:
        exp_vals = torch.pow(2.0, shifted)
        acc = exp_vals.sum(dim=0)
        global_lse = torch.log2(acc) + lse_max

    # Compute correction factor
    local_lse = lses_float[lse_idx]  # [B, H]
    lse_diff = local_lse - global_lse

    # Clean up
    lse_diff = torch.where(
        torch.isnan(lse_diff) | (lse_diff == float("inf")),
        torch.tensor(-float("inf"), device=lse_diff.device, dtype=torch.float32),
        lse_diff,
    )

    if is_base_e:
        factor = torch.exp(lse_diff)
    else:
        factor = torch.pow(2.0, lse_diff)

    corrected = out.float() * factor.unsqueeze(-1)
    return corrected.to(out.dtype), global_lse


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "correct_attn_cp_out"), "Missing correct_attn_cp_out"
        assert hasattr(mod, "_correct_attn_cp_out_kernel"), "Missing _correct_attn_cp_out_kernel"
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

    for i, (B, H, D, N) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)

            out = torch.randn(B, H, D, device=device, dtype=torch.float16)
            lses = torch.randn(N, B, H, device=device, dtype=torch.float32)
            lse_idx = 0

            corrected, final_lse = mod.correct_attn_cp_out(out, lses, lse_idx, is_base_e=True)
            torch.cuda.synchronize()

            ref_corrected, ref_lse = reference_correct_attn_cp_out(out, lses, lse_idx, is_base_e=True)

            if not torch.allclose(final_lse.float(), ref_lse.float(), atol=1e-2, rtol=1e-2):
                max_diff = (final_lse.float() - ref_lse.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): "
                    f"lse max diff = {max_diff:.6f}"
                )

            if not torch.allclose(corrected.float(), ref_corrected.float(), atol=1e-2, rtol=1e-2):
                max_diff = (corrected.float() - ref_corrected.float()).abs().max().item()
                return False, (
                    f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): "
                    f"output max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i+1} (B={B}, H={H}, D={D}, N={N}): exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    B, H, D, N = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    out = torch.randn(B, H, D, device=device, dtype=torch.float16)
    lses = torch.randn(N, B, H, device=device, dtype=torch.float32)

    for _ in range(5):
        mod.correct_attn_cp_out(out, lses, 0, is_base_e=True)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.correct_attn_cp_out(out, lses, 0, is_base_e=True)
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
        B, H, D, N = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {"B": B, "H": H, "D": D, "N": N},
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
