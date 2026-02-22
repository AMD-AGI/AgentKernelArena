#!/usr/bin/env python3
"""Task runner for triton_selective_scan_update"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_selective_scan_update.py")

# (batch, nheads, dim, dstate, ngroups, has_D, has_z)
TEST_SHAPES = [
    (4, 8, 64, 16, 4, True, False),
    (8, 4, 128, 32, 2, True, True),
    (2, 16, 64, 64, 8, False, False),
    (4, 8, 128, 16, 4, True, True),
    (16, 4, 64, 32, 2, True, False),
]
PERF_IDX = 1


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(state, x, dt, A, B, C, D, z):
    import torch
    batch, nheads, dim = x.shape
    dstate = state.shape[-1]
    ngroups = B.shape[1]
    nheads_per_group = nheads // ngroups
    state_f = state.float()
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    out = torch.zeros(batch, nheads, dim, dtype=torch.float32)
    for b in range(batch):
        for h in range(nheads):
            g = h // nheads_per_group
            for d in range(dim):
                for s in range(dstate):
                    dA_val = torch.exp(A_f[h, d, s] * dt_f[b, h, d])
                    dB_val = B_f[b, g, s] * dt_f[b, h, d]
                    state_f[b, h, d, s] = state_f[b, h, d, s] * dA_val + dB_val * x_f[b, h, d]
            for d in range(dim):
                val = 0.0
                for s in range(dstate):
                    val += state_f[b, h, d, s] * C_f[b, g, s]
                if D is not None:
                    val += x_f[b, h, d] * D.float()[h, d]
                if z is not None:
                    zv = z.float()[b, h, d]
                    val *= zv * torch.sigmoid(torch.tensor(zv))
                out[b, h, d] = val
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_selective_scan_update_kernel")
        assert hasattr(mod, "selective_state_update")
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
    for i, (batch, nheads, dim, dstate, ngroups, has_D, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=torch.float32) * 0.1
            x = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32)
            dt = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) * 0.1
            A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32)
            B = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            C = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
            D = torch.randn(nheads, dim, device=device, dtype=torch.float32) if has_D else None
            z = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) if has_z else None
            out = torch.empty(batch, nheads, dim, device=device, dtype=torch.float32)
            state_copy = state.clone()
            ref = reference(state_copy.cpu(), x.cpu(), dt.cpu(), A.cpu(), B.cpu(), C.cpu(),
                          D.cpu() if D is not None else None,
                          z.cpu() if z is not None else None)
            mod.selective_state_update(state, x, dt, A, B, C, D=D, z=z, out=out)
            torch.cuda.synchronize()
            if not torch.allclose(out.cpu(), ref, atol=1e-2, rtol=1e-2):
                diff = (out.cpu() - ref).abs().max().item()
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
    batch, nheads, dim, dstate, ngroups, has_D, has_z = TEST_SHAPES[PERF_IDX]
    torch.manual_seed(0)
    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=torch.float32) * 0.1
    x = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=torch.float32)
    D = torch.randn(nheads, dim, device=device, dtype=torch.float32) if has_D else None
    z = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32) if has_z else None
    out = torch.empty(batch, nheads, dim, device=device, dtype=torch.float32)
    for _ in range(5):
        mod.selective_state_update(state.clone(), x, dt, A, B, C, D=D, z=z, out=out)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.selective_state_update(state.clone(), x, dt, A, B, C, D=D, z=z, out=out)
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
