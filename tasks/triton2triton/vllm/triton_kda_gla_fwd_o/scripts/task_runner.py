#!/usr/bin/env python3
"""Task runner for triton2triton/triton_kda_gla_fwd_o"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_kda_gla_fwd_o"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_kda_gla_fwd_o.py")

SEEDS = [42, 43, 44, 45, 46]
PERF_SEED_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(q, v, g, A, h, scale, chunk_size=64):
    import torch
    from math import ceil
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = ceil(T / BT)
    q_c, v_c, g_c, A_c, h_c = q.float().cpu(), v.float().cpu(), g.float().cpu(), A.float().cpu(), h.float().cpu()
    o = torch.zeros(B, T, H, V)
    for b in range(B):
        for hh in range(H):
            for t in range(NT):
                s = t * BT
                e = min(s + BT, T)
                bq = q_c[b, s:e, hh] * scale
                bg = g_c[b, s:e, hh]
                bqg = bq * torch.exp(bg)
                bh = h_c[b, t, hh]  # [K, V]
                o[b, s:e, hh] = bqg @ bh
                bA = A_c[b, s:e, hh, :e-s]
                mask = torch.tril(torch.ones(e-s, e-s))
                bA = bA * mask
                o[b, s:e, hh] += bA @ v_c[b, s:e, hh]
    return o


def gen_inputs(seed, device):
    import torch
    torch.manual_seed(seed)
    B, T, H, K, V, BT = 1, 64, 2, 32, 32, 64
    from math import ceil
    NT = ceil(T / BT)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32) * 0.1
    g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.01
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32) * 0.01
    h = torch.randn(B, NT, H, K, V, device=device, dtype=torch.float32) * 0.1
    return (q, v, g, A, h, 0.125), {"chunk_size": BT}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "kda_gla_fwd_o"), "Missing kda_gla_fwd_o"
        assert hasattr(mod, "chunk_gla_fwd_kernel_o"), "Missing chunk_gla_fwd_kernel_o"
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
    for i, seed in enumerate(SEEDS):
        try:
            args, kwargs = gen_inputs(seed, device)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result = mod.kda_gla_fwd_o(*args, **kwargs)
            ref = reference(*args_cpu, **kwargs)
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=5e-2, rtol=5e-2):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"

            torch.cuda.synchronize()
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
    args, kwargs = gen_inputs(SEEDS[PERF_SEED_IDX], device)

    for _ in range(5):
        mod.kda_gla_fwd_o(*args, **kwargs)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.kda_gla_fwd_o(*args, **kwargs)
        end_events[j].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(SEEDS)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
