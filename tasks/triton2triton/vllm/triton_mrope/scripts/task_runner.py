#!/usr/bin/env python3
"""Task runner for triton2triton/triton_mrope"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_mrope"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_mrope.py")

# Test configurations: (num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim, mrope_section)
TEST_SHAPES = [
    (32, 8, 8, 64, 64, [16, 8, 8]),
    (64, 16, 4, 64, 64, [16, 8, 8]),
    (128, 32, 8, 128, 64, [16, 8, 8]),
    (256, 16, 16, 64, 64, [16, 8, 8]),
    (16, 8, 2, 128, 64, [16, 8, 8]),
]
PERF_SHAPE_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim):
    """CPU/PyTorch reference for MRoPE.

    q: [num_tokens, num_q_heads * head_size]
    k: [num_tokens, num_kv_heads * head_size]
    cos: [3, num_tokens, rotary_dim // 2]
    sin: [3, num_tokens, rotary_dim // 2]
    mrope_section: [t, h, w]
    """
    import torch

    num_tokens = q.shape[0]
    n_q_head = q.shape[1] // head_size
    n_kv_head = k.shape[1] // head_size
    half_rd = rotary_dim // 2

    # Build combined cos/sin from sections (non-interleaved)
    t_sec, h_sec, w_sec = mrope_section
    # cos/sin shape: [3, num_tokens, rotary_dim // 2]
    # Section t: indices [0, t_sec), from cos[0]
    # Section h: indices [t_sec, t_sec+h_sec), from cos[1]
    # Section w: indices [t_sec+h_sec, half_rd), from cos[2]
    combined_cos = torch.zeros(num_tokens, half_rd, device=q.device, dtype=cos.dtype)
    combined_sin = torch.zeros(num_tokens, half_rd, device=q.device, dtype=sin.dtype)

    combined_cos[:, :t_sec] = cos[0, :, :t_sec]
    combined_sin[:, :t_sec] = sin[0, :, :t_sec]
    combined_cos[:, t_sec:t_sec + h_sec] = cos[1, :, t_sec:t_sec + h_sec]
    combined_sin[:, t_sec:t_sec + h_sec] = sin[1, :, t_sec:t_sec + h_sec]
    combined_cos[:, t_sec + h_sec:half_rd] = cos[2, :, t_sec + h_sec:half_rd]
    combined_sin[:, t_sec + h_sec:half_rd] = sin[2, :, t_sec + h_sec:half_rd]

    # Apply rotary embedding to q
    q_out = q.clone()
    for h in range(n_q_head):
        offset = h * head_size
        x1 = q_out[:, offset:offset + half_rd].float()
        x2 = q_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        q_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(q.dtype)
        q_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(q.dtype)

    # Apply rotary embedding to k
    k_out = k.clone()
    for h in range(n_kv_head):
        offset = h * head_size
        x1 = k_out[:, offset:offset + half_rd].float()
        x2 = k_out[:, offset + half_rd:offset + rotary_dim].float()
        c = combined_cos.float()
        s = combined_sin.float()
        k_out[:, offset:offset + half_rd] = (x1 * c - x2 * s).to(k.dtype)
        k_out[:, offset + half_rd:offset + rotary_dim] = (x2 * c + x1 * s).to(k.dtype)

    return q_out, k_out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_mrope"), "Missing triton_mrope"
        assert hasattr(mod, "_triton_mrope_forward"), "Missing _triton_mrope_forward"
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

    for i, (num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
            k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
            cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
            sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)

            # Clone for reference
            q_ref = q.clone()
            k_ref = k.clone()

            # Triton kernel (in-place)
            q_triton = q.clone()
            k_triton = k.clone()
            mod.triton_mrope(
                q_triton, k_triton, cos, sin, mrope_section,
                head_size, rotary_dim, False
            )
            torch.cuda.synchronize()

            # Reference
            q_expected, k_expected = reference_mrope(
                q_ref, k_ref, cos, sin, mrope_section, head_size, rotary_dim
            )

            if not torch.allclose(q_triton, q_expected, atol=1e-2, rtol=1e-2):
                max_diff = (q_triton - q_expected).abs().max().item()
                return False, (
                    f"Shape {i + 1} q mismatch: max diff = {max_diff:.6f}"
                )
            if not torch.allclose(k_triton, k_expected, atol=1e-2, rtol=1e-2):
                max_diff = (k_triton - k_expected).abs().max().item()
                return False, (
                    f"Shape {i + 1} k mismatch: max diff = {max_diff:.6f}"
                )
        except Exception as e:
            return False, f"Shape {i + 1}: exception: {e}"

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    dtype = torch.float16
    num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section = TEST_SHAPES[PERF_SHAPE_IDX]

    torch.manual_seed(0)
    q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
    k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
    cos = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(3, num_tokens, rotary_dim // 2, device=device, dtype=dtype)

    for _ in range(5):
        q_tmp = q.clone()
        k_tmp = k.clone()
        mod.triton_mrope(q_tmp, k_tmp, cos, sin, mrope_section, head_size, rotary_dim, False)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        q_tmp = q.clone()
        k_tmp = k.clone()
        start_events[j].record()
        mod.triton_mrope(q_tmp, k_tmp, cos, sin, mrope_section, head_size, rotary_dim, False)
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
        num_tokens, n_qh, n_kh, head_size, rotary_dim, mrope_section = TEST_SHAPES[PERF_SHAPE_IDX]
        report = {
            "execution_time_ms": elapsed_ms,
            "shape": {
                "num_tokens": num_tokens,
                "num_q_heads": n_qh,
                "num_kv_heads": n_kh,
                "head_size": head_size,
            },
        }
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
