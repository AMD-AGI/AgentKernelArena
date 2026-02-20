#!/usr/bin/env python3
"""Task runner for triton_ssd_state_passing"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_state_passing.py")

# (nchunks, nheads, dim, chunk_size)
TEST_SHAPES = [
    (4, 8, 64, 64),
    (8, 16, 128, 64),
    (4, 8, 256, 128),
    (16, 4, 64, 64),
    (8, 32, 128, 128),
]
PERF_IDX = 1


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(states, dA_cumsum, seq_idx):
    import torch
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    out = torch.zeros_like(states, dtype=torch.float32)
    s_cpu = states.cpu().float()
    dA_cpu = dA_cumsum.cpu().float()
    si_cpu = seq_idx.cpu()
    running = torch.zeros(nheads, dim, dtype=torch.float32)
    prev_seq = 0
    for c in range(nchunks):
        si = si_cpu[c].item()
        if si != prev_seq:
            running.zero_()
        prev_seq = si
        for h in range(nheads):
            dA_val = dA_cpu[h, c, chunk_size - 1].item()
            running[h] = running[h] * torch.exp(torch.tensor(dA_val)) + s_cpu[c, h]
            out[c, h] = running[h]
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_state_passing_fwd_kernel")
        assert hasattr(mod, "state_passing_fwd")
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
    for i, (nchunks, nheads, dim, chunk_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            states = torch.randn(nchunks, nheads, dim, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(torch.randn(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.01, dim=-1)
            seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            result = mod.state_passing_fwd(states, dA_cumsum, cu, seq_idx)
            ref = reference(states, dA_cumsum, seq_idx).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=1e-2, rtol=1e-2):
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
    nchunks, nheads, dim, chunk_size = TEST_SHAPES[PERF_IDX]
    torch.manual_seed(0)
    states = torch.randn(nchunks, nheads, dim, device=device, dtype=torch.float32) * 0.1
    dA_cumsum = torch.cumsum(torch.randn(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.01, dim=-1)
    seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
    cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
    for _ in range(5):
        mod.state_passing_fwd(states, dA_cumsum, cu, seq_idx)
    torch.cuda.synchronize()
    n_iter = 20
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for j in range(n_iter):
        starts[j].record()
        mod.state_passing_fwd(states, dA_cumsum, cu, seq_idx)
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
