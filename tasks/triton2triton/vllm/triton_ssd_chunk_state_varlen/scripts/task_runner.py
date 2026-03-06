#!/usr/bin/env python3
"""Task runner for triton_ssd_chunk_state_varlen"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_chunk_state_varlen.py")

# (total_seqlen, batch, nheads, headdim, ngroups, dstate, chunk_size)
TEST_SHAPES = [
    (128, 2, 4, 32, 2, 16, 64),
    (256, 4, 8, 32, 4, 16, 64),
    (256, 2, 8, 64, 2, 32, 64),
    (384, 3, 4, 32, 2, 16, 128),
    (256, 2, 4, 64, 2, 32, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, chunk_size, initial_states=None):
    import torch

    total_seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    ratio = nheads // ngroups
    batch = cu_seqlens.shape[0] - 1

    B_c = B.float().cpu()
    x_c = x.float().cpu()
    dt_c = dt.float().cpu()
    dA_c = dA_cumsum.float().cpu()
    chunk_states_c = chunk_states.float().cpu()
    cu_c = cu_seqlens.cpu()
    init_c = initial_states.float().cpu() if initial_states is not None else None

    out = torch.zeros(batch, nheads, headdim, dstate, dtype=torch.float32)
    for b in range(batch):
        start_idx = cu_c[b].item()
        end_idx = cu_c[b + 1].item()
        pid_c = (end_idx - 1) // chunk_size
        chunk_start = pid_c * chunk_size
        chunk_size_limit = end_idx - chunk_start
        start_idx_cur = max(start_idx - chunk_start, 0)
        for h in range(nheads):
            g = h // ratio
            dA_last = dA_c[h, pid_c, chunk_size_limit - 1]
            acc = torch.zeros(headdim, dstate, dtype=torch.float32)
            for k in range(start_idx_cur, chunk_size_limit):
                tok = chunk_start + k
                scale = torch.exp(dA_last - dA_c[h, pid_c, k]) * dt_c[h, pid_c, k]
                acc += torch.outer(x_c[tok, h], B_c[tok, g]) * scale

            if start_idx < chunk_start or init_c is not None:
                dA_boundary = 0.0
                if init_c is None:
                    past = chunk_states_c[pid_c, h]
                else:
                    if start_idx < chunk_start:
                        past = chunk_states_c[pid_c, h]
                    else:
                        past = init_c[b, h]
                        if start_idx > chunk_start:
                            dA_boundary = dA_c[h, pid_c, start_idx - chunk_start - 1]
                acc += past * torch.exp(dA_last - dA_boundary)

            out[b, h] = acc
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_state_varlen_kernel")
        assert hasattr(mod, "chunk_state_varlen")
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
    for i, (total_seqlen, batch, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            nchunks = total_seqlen // chunk_size
            # Create uniform cu_seqlens
            seq_per_batch = total_seqlen // batch
            cu_seqlens = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_per_batch
            x = torch.randn(total_seqlen, nheads, headdim, device=device, dtype=torch.float16)
            B = torch.randn(total_seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            chunk_states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32)
            result = mod.chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
            ref = reference_chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, chunk_size).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=5e-2, rtol=5e-2):
                diff = (result.float() - ref.float()).abs().max().item()
                return False, f"Shape {i}: max diff={diff:.6f}"
        except Exception as e:
            return False, f"Shape {i}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (total_seqlen, batch, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
            nchunks = total_seqlen // chunk_size
            seq_per_batch = total_seqlen // batch
            torch.manual_seed(42 + test_idx)
            cu_seqlens = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * seq_per_batch
            x = torch.randn(total_seqlen, nheads, headdim, device=device, dtype=torch.float16)
            B = torch.randn(total_seqlen, ngroups, dstate, device=device, dtype=torch.float16)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            chunk_states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32)
            for _ in range(WARMUP_ITERATIONS):
                mod.chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record()
                mod.chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
                ends[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            elapsed_ms = sum(times) / len(times)

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                "params": {
                    "total_seqlen": total_seqlen,
                    "batch": batch,
                    "nheads": nheads,
                    "headdim": headdim,
                    "ngroups": ngroups,
                    "dstate": dstate,
                    "chunk_size": chunk_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "total_seqlen": total_seqlen,
                    "batch": batch,
                    "nheads": nheads,
                    "headdim": headdim,
                    "ngroups": ngroups,
                    "dstate": dstate,
                    "chunk_size": chunk_size
                }
            })
    return test_cases


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
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
