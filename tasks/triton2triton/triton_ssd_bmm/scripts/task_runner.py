#!/usr/bin/env python3
"""Task runner for triton_ssd_bmm"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_bmm.py")

# (seqlen, ngroups, k, chunk_size, causal)
TEST_SHAPES = [
    (128, 2, 32, 64, False),
    (256, 4, 64, 64, True),
    (512, 2, 32, 128, False),
    (256, 8, 16, 64, True),
    (384, 4, 64, 128, False),
]
PERF_IDX = 2


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_bmm(a, b, chunk_size, cu_chunk_seqlens, causal):
    import torch
    nchunks = len(cu_chunk_seqlens) - 1
    ngroups, k = a.shape[1], a.shape[2]
    out = torch.zeros(nchunks, ngroups, chunk_size, chunk_size, device="cpu", dtype=torch.float32)
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    cu = cu_chunk_seqlens.cpu()
    for c in range(nchunks):
        s, e = cu[c].item(), cu[c + 1].item()
        clen = e - s
        for g in range(ngroups):
            ag = a_cpu[s:e, g, :]  # (clen, k)
            bg = b_cpu[s:e, g, :]  # (clen, k)
            block = ag @ bg.T  # (clen, clen)
            out[c, g, :clen, :clen] = block
    return out.to(a.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_bmm_chunk_fwd_kernel")
        assert hasattr(mod, "bmm_chunk_fwd")
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
    for i, (seqlen, ngroups, k, chunk_size, causal) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            nchunks = seqlen // chunk_size
            a = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            b = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            result = mod.bmm_chunk_fwd(a, b, chunk_size, cu, causal=causal)
            ref = reference_bmm(a, b, chunk_size, cu, causal).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=1e-1, rtol=1e-1):
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
    seqlen, ngroups, k, chunk_size, causal = TEST_SHAPES[PERF_IDX]
    nchunks = seqlen // chunk_size
    torch.manual_seed(0)
    a = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
    b = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
    cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
    for _ in range(5):
        mod.bmm_chunk_fwd(a, b, chunk_size, cu, causal=causal)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.bmm_chunk_fwd(a, b, chunk_size, cu, causal=causal)
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
