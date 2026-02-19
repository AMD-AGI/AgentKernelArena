#!/usr/bin/env python3
"""Task runner for triton_ssd_chunk_state"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_chunk_state.py")

# (seqlen, nheads, headdim, ngroups, dstate, chunk_size)
TEST_SHAPES = [
    (128, 4, 32, 2, 16, 64),
    (256, 8, 64, 4, 32, 64),
    (512, 4, 32, 2, 16, 128),
    (256, 8, 64, 2, 64, 64),
    (384, 4, 128, 2, 32, 128),
]
PERF_IDX = 1


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(B, x, dt, dA_cumsum, cu):
    import torch
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    states = torch.zeros(nchunks, nheads, headdim, dstate, dtype=torch.float32, device="cpu")
    x_cpu = x.cpu().float()
    B_cpu = B.cpu().float()
    dt_cpu = dt.cpu().float()
    dA_cpu = dA_cumsum.cpu().float()
    cu_cpu = cu.cpu()
    nheads_ngroups_ratio = nheads // ngroups
    for c in range(nchunks):
        s, e = cu_cpu[c].item(), cu_cpu[c+1].item()
        clen = e - s
        dA_cs_last = dA_cpu[:, c, chunk_size - 1]
        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            for t in range(clen):
                scale = torch.exp(dA_cs_last[h] - dA_cpu[h, c, t]) * dt_cpu[h, c, t]
                states[c, h] += x_cpu[s + t, h, :, None] * (B_cpu[s + t, g, None, :] * scale)
    return states


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_state_fwd_kernel")
        assert hasattr(mod, "chunk_state_fwd")
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
    for i, (seqlen, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            nchunks = seqlen // chunk_size
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
            B = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            result = mod.chunk_state_fwd(B, x, dt, dA_cumsum, cu)
            ref = reference(B, x, dt, dA_cumsum, cu).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=5e-1, rtol=1e-1):
                diff = (result.float() - ref.float()).abs().max().item()
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
    seqlen, nheads, headdim, ngroups, dstate, chunk_size = TEST_SHAPES[PERF_IDX]
    nchunks = seqlen // chunk_size
    torch.manual_seed(0)
    x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
    B = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
    dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
    dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
    cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
    for _ in range(5):
        mod.chunk_state_fwd(B, x, dt, dA_cumsum, cu)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.chunk_state_fwd(B, x, dt, dA_cumsum, cu)
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
