#!/usr/bin/env python3
"""Task runner for triton_ssd_chunk_cumsum"""
import sys, os, json, argparse, importlib.util
import math

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_chunk_cumsum.py")

# (seqlen, nheads, chunk_size, has_bias, softplus)
TEST_SHAPES = [
    (128, 8, 64, False, False),
    (256, 16, 64, True, False),
    (512, 8, 128, False, True),
    (256, 32, 64, True, True),
    (384, 16, 64, False, False),
]
PERF_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_softplus(x):
    import torch
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def reference(dt, A, chunk_size, cu, dt_bias, softplus):
    import torch
    seqlen, nheads = dt.shape
    nchunks = len(cu) - 1
    dt_f = dt.cpu().float()
    A_f = A.cpu().float()
    dt_out = torch.zeros(nheads, nchunks, chunk_size, dtype=torch.float32)
    dA_cumsum = torch.zeros(nheads, nchunks, chunk_size, dtype=torch.float32)
    cu_cpu = cu.cpu()
    for c in range(nchunks):
        s, e = cu_cpu[c].item(), cu_cpu[c+1].item()
        clen = e - s
        for h in range(nheads):
            dt_chunk = dt_f[s:e, h].clone()
            if dt_bias is not None:
                dt_chunk += dt_bias.cpu().float()[h]
            if softplus:
                dt_chunk = ref_softplus(dt_chunk)
            dt_chunk = dt_chunk.clamp(min=0.0)
            dt_out[h, c, :clen] = dt_chunk
            dA = dt_chunk * A_f[h]
            dA_cumsum[h, c, :clen] = torch.cumsum(dA, 0)
    return dA_cumsum, dt_out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_cumsum_fwd_kernel")
        assert hasattr(mod, "chunk_cumsum_fwd")
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
    for i, (seqlen, nheads, chunk_size, has_bias, softplus) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            nchunks = seqlen // chunk_size
            dt = torch.randn(seqlen, nheads, device=device, dtype=torch.float32) * 0.1
            A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
            dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.01 if has_bias else None
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            dA_cs, dt_out = mod.chunk_cumsum_fwd(dt, A, chunk_size, cu, dt_bias=dt_bias, dt_softplus=softplus)
            ref_dA, ref_dt = reference(dt, A, chunk_size, cu, dt_bias, softplus)
            ref_dA = ref_dA.to(device)
            ref_dt = ref_dt.to(device)
            if not torch.allclose(dA_cs, ref_dA, atol=1e-3, rtol=1e-3):
                diff = (dA_cs - ref_dA).abs().max().item()
                return False, f"Shape {i} dA_cumsum: max diff={diff}"
            if not torch.allclose(dt_out, ref_dt, atol=1e-3, rtol=1e-3):
                diff = (dt_out - ref_dt).abs().max().item()
                return False, f"Shape {i} dt_out: max diff={diff}"
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
    seqlen, nheads, chunk_size, has_bias, softplus = TEST_SHAPES[PERF_IDX]
    nchunks = seqlen // chunk_size
    torch.manual_seed(0)
    dt = torch.randn(seqlen, nheads, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.01 if has_bias else None
    cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
    for _ in range(5):
        mod.chunk_cumsum_fwd(dt, A, chunk_size, cu, dt_bias=dt_bias, dt_softplus=softplus)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.chunk_cumsum_fwd(dt, A, chunk_size, cu, dt_bias=dt_bias, dt_softplus=softplus)
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
