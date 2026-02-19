#!/usr/bin/env python3
"""Task runner for triton_ssd_chunk_scan"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_chunk_scan.py")

# (seqlen, nheads, headdim, ngroups, dstate, chunk_size)
TEST_SHAPES = [
    (128, 4, 32, 2, 16, 64),
    (256, 8, 64, 4, 32, 64),
    (512, 4, 32, 2, 16, 128),
    (256, 4, 64, 2, 16, 64),
    (384, 8, 32, 4, 32, 128),
]
PERF_IDX = 1


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, chunk_size):
    import torch

    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    ratio = nheads // ngroups
    nchunks = seqlen // chunk_size

    cb_c = cb.float().cpu()
    x_c = x.float().cpu()
    dt_c = dt.float().cpu()
    dA_c = dA_cumsum.float().cpu()
    C_c = C.float().cpu()
    states_c = states.float().cpu()
    seq_c = seq_idx.cpu()

    out = torch.zeros(seqlen, nheads, headdim, dtype=torch.float32)
    for c in range(nchunks):
        for h in range(nheads):
            g = h // ratio
            if c == 0 or seq_c[c].item() != seq_c[c - 1].item():
                prev_state = torch.zeros(headdim, dstate, dtype=torch.float32)
            else:
                prev_state = states_c[c - 1, h]
            for t in range(chunk_size):
                tok = c * chunk_size + t
                if tok >= seqlen:
                    break
                dA_t = dA_c[h, c, t]
                acc = torch.matmul(prev_state, C_c[tok, g]) * torch.exp(dA_t)
                for k in range(t + 1):
                    tok_k = c * chunk_size + k
                    coeff = cb_c[c, g, t, k] * torch.exp(dA_t - dA_c[h, c, k]) * dt_c[h, c, k]
                    acc = acc + coeff * x_c[tok_k, h]
                out[tok, h] = acc
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_scan_fwd_kernel")
        assert hasattr(mod, "chunk_scan_fwd")
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
            cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.float16) * 0.01
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
            C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
            seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
            mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
            torch.cuda.synchronize()
            ref = reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, chunk_size).to(device)
            if not torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2):
                diff = (out.float() - ref.float()).abs().max().item()
                return False, f"Shape {i}: max diff = {diff:.6f}"
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
    cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.float16) * 0.01
    x = torch.randn(seqlen, nheads, headdim, device=device, dtype=torch.float16)
    C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=torch.float16)
    dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
    dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
    states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
    seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
    cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
    out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
    for _ in range(5):
        mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
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
