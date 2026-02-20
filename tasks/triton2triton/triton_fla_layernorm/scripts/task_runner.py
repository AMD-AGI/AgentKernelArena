#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fla_layernorm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_fla_layernorm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fla_layernorm.py")

TEST_CASES = [
    # (M, N, has_bias, has_z, norm_before_gate, is_rms_norm)
    (256, 128, True, False, True, False),
    (192, 96, False, False, True, True),
    (128, 64, True, True, True, False),
    (128, 64, True, True, False, False),
    (96, 80, False, False, True, False),
]
PERF_CASE_IDX = 0


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(x, weight, bias=None, eps=1e-5, z=None, norm_before_gate=True, is_rms_norm=False):
    import torch
    x_f = x.float().cpu()
    if z is not None and not norm_before_gate:
        z_f = z.float().cpu()
        x_f = x_f * z_f * torch.sigmoid(z_f)
    if is_rms_norm:
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x_f - mean) * torch.rsqrt(var + eps)
    y = x_hat * weight.float().cpu()
    if bias is not None:
        y = y + bias.float().cpu()
    if z is not None and norm_before_gate:
        z_f = z.float().cpu()
        y = y * z_f * torch.sigmoid(z_f)
    return y


def gen_inputs(seed, case_idx, device):
    import torch
    torch.manual_seed(seed)
    M, N, has_bias, has_z, norm_before_gate, is_rms_norm = TEST_CASES[case_idx]
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    w = torch.ones(N, device=device, dtype=torch.float32)
    b = torch.zeros(N, device=device, dtype=torch.float32) if has_bias else None
    z = torch.randn(M, N, device=device, dtype=torch.float32) if has_z else None
    kwargs = {"z": z, "norm_before_gate": norm_before_gate, "is_rms_norm": is_rms_norm}
    return (x, w, b), kwargs


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "layer_norm_fwd"), "Missing layer_norm_fwd"
        assert hasattr(mod, "layer_norm_fwd_kernel"), "Missing layer_norm_fwd_kernel"
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
    for i, test_case in enumerate(TEST_CASES):
        try:
            args, kwargs = gen_inputs(42 + i, i, device)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result_tuple = mod.layer_norm_fwd(*args, **kwargs)
            result = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
            ref = reference(
                args_cpu[0], args_cpu[1], args_cpu[2],
                z=kwargs["z"], norm_before_gate=kwargs["norm_before_gate"], is_rms_norm=kwargs["is_rms_norm"]
            )
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=1e-3, rtol=1e-3):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return False, f"Case {i+1} {test_case}: max diff = {max_diff:.6f}"

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Case {i+1} {test_case}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0

    device = "cuda"
    args, kwargs = gen_inputs(42, PERF_CASE_IDX, device)

    for _ in range(5):
        mod.layer_norm_fwd(*args, **kwargs)
    torch.cuda.synchronize()

    n_iter = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    for j in range(n_iter):
        start_events[j].record()
        mod.layer_norm_fwd(*args, **kwargs)
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
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_CASES)}
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
