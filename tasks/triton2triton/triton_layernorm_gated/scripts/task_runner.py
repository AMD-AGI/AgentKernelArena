#!/usr/bin/env python3
"""Task runner for triton_layernorm_gated"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_layernorm_gated.py")

# (M, N, is_rms, has_bias, has_z)
TEST_SHAPES = [
    (32, 128, False, True, False),
    (64, 256, True, False, False),
    (128, 512, True, True, True),
    (256, 1024, False, True, True),
    (512, 2048, True, False, True),
]
PERF_IDX = 4


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(x, weight, bias, eps, z, is_rms, norm_before_gate=True):
    import torch
    x_f = x.float()
    if z is not None and not norm_before_gate:
        z_f = z.float()
        x_f = x_f * z_f * torch.sigmoid(z_f)
    if is_rms:
        var = (x_f ** 2).mean(-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)
    else:
        mean = x_f.mean(-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(-1, keepdim=True)
        x_hat = (x_f - mean) * torch.rsqrt(var + eps)
    y = x_hat * weight.float()
    if bias is not None:
        y = y + bias.float()
    if z is not None and norm_before_gate:
        z_f = z.float()
        y = y * z_f * torch.sigmoid(z_f)
    return y.to(x.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_layer_norm_fwd_1pass_kernel")
        assert hasattr(mod, "layer_norm_fwd")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Load failed: {e}"
    device = "cuda"
    for i, (M, N, is_rms, has_bias, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device=device, dtype=torch.float16)
            w = torch.randn(N, device=device, dtype=torch.float16)
            b = torch.randn(N, device=device, dtype=torch.float16) if has_bias else None
            z = torch.randn(M, N, device=device, dtype=torch.float16) if has_z else None
            eps = 1e-5
            out, _, _ = mod.layer_norm_fwd(x, w, b, eps, z=z, is_rms_norm=is_rms)
            ref = reference(x, w, b, eps, z, is_rms)
            if not torch.allclose(out, ref, atol=1e-2, rtol=1e-2):
                diff = (out - ref).abs().max().item()
                return False, f"Shape {i}: max diff={diff}"
        except Exception as e:
            return False, f"Shape {i}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return -1.0
    device = "cuda"
    M, N, is_rms, has_bias, has_z = TEST_SHAPES[PERF_IDX]
    torch.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    w = torch.randn(N, device=device, dtype=torch.float16)
    b = torch.randn(N, device=device, dtype=torch.float16) if has_bias else None
    z = torch.randn(M, N, device=device, dtype=torch.float16) if has_z else None
    for _ in range(5):
        mod.layer_norm_fwd(x, w, b, 1e-5, z=z, is_rms_norm=is_rms)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.layer_norm_fwd(x, w, b, 1e-5, z=z, is_rms_norm=is_rms)
        ends[j].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
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
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        ms = run_performance()
        report = {"execution_time_ms": ms}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
