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

# Correctness-only cases for contract paths that are intentionally absent from
# the fixed performance workload. ``chunk_lengths`` describes the packed token
# range assigned to each chunk and may be shorter than ``chunk_size``.
CORRECTNESS_CASES = [
    {
        "name": f"regular_{i}",
        "shape": shape,
    }
    for i, shape in enumerate(TEST_SHAPES)
] + [
    {
        "name": "irregular_boundary_large_dstate",
        "shape": (94, 4, 40, 2, 129, 48),
        "chunk_lengths": (29, 48, 17),
        "seq_idx": (0, 0, 1),
        "D_shape": "head",
    },
    {
        "name": "irregular_boundary_initial_state",
        "shape": (53, 4, 35, 2, 31, 24),
        "chunk_lengths": (24, 11, 18),
        "seq_idx": (0, 1, 1),
        "initial_states": True,
        "disable_rocm_buffer_ops": True,
    },
    {
        "name": "bf16_irregular_boundary_D_hdim_z",
        "shape": (76, 6, 37, 3, 21, 33),
        "chunk_lengths": (33, 16, 27),
        "seq_idx": (0, 1, 1),
        "dtype": "bfloat16",
        "D_shape": "head_hdim",
        "z": True,
    },
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_chunk_scan(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    import torch

    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    ratio = nheads // ngroups
    nchunks = cb.shape[0]

    cb_c = cb.float().cpu()
    x_c = x.float().cpu()
    dt_c = dt.float().cpu()
    dA_c = dA_cumsum.float().cpu()
    C_c = C.float().cpu()
    states_c = states.float().cpu()
    cu_c = cu_chunk_seqlens.cpu()
    seq_c = seq_idx.cpu()
    D_c = D.float().cpu() if D is not None else None
    z_c = z.float().cpu() if z is not None else None
    initial_states_c = (
        initial_states.float().cpu() if initial_states is not None else None
    )

    out = torch.zeros(seqlen, nheads, headdim, dtype=torch.float32)
    for c in range(nchunks):
        chunk_start = cu_c[c].item()
        chunk_end = cu_c[c + 1].item()
        for h in range(nheads):
            g = h // ratio
            if c == 0 or seq_c[c].item() != seq_c[c - 1].item():
                if initial_states_c is None:
                    prev_state = torch.zeros(headdim, dstate, dtype=torch.float32)
                else:
                    prev_state = initial_states_c[seq_c[c].item(), h]
            else:
                prev_state = states_c[c - 1, h]
            for t in range(chunk_end - chunk_start):
                tok = chunk_start + t
                dA_t = dA_c[h, c, t]
                acc = torch.matmul(prev_state, C_c[tok, g]) * torch.exp(dA_t)
                for k in range(t + 1):
                    tok_k = chunk_start + k
                    coeff = cb_c[c, g, t, k] * torch.exp(dA_t - dA_c[h, c, k]) * dt_c[h, c, k]
                    acc = acc + coeff * x_c[tok_k, h]
                if D_c is not None:
                    acc = acc + x_c[tok, h] * D_c[h]
                if z_c is not None:
                    z_tok = z_c[tok, h]
                    acc = acc * z_tok * torch.sigmoid(z_tok)
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
    for i, case in enumerate(CORRECTNESS_CASES):
        seqlen, nheads, headdim, ngroups, dstate, chunk_size = case["shape"]
        case_name = case["name"]
        try:
            torch.manual_seed(42 + i)
            chunk_lengths = case.get("chunk_lengths", (chunk_size,) * (seqlen // chunk_size))
            assert sum(chunk_lengths) == seqlen
            assert all(0 < length <= chunk_size for length in chunk_lengths)
            nchunks = len(chunk_lengths)
            input_dtype = getattr(torch, case.get("dtype", "float16"))
            cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=input_dtype) * 0.01
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=input_dtype)
            C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=input_dtype)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * (-0.1), dim=-1)
            states = torch.randn(
                nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32
            ) * 0.01
            seq_idx = torch.tensor(
                case.get("seq_idx", (0,) * nchunks), device=device, dtype=torch.int32
            )
            cu = torch.tensor(
                (0, *chunk_lengths), device=device, dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            initial_states = None
            if case.get("initial_states"):
                nseqs = max(case["seq_idx"]) + 1
                initial_states = torch.randn(
                    nseqs, nheads, headdim, dstate,
                    device=device, dtype=torch.float32,
                ) * 0.01
            D = None
            if case.get("D_shape") == "head":
                D = torch.linspace(-0.75, 0.75, nheads, device=device, dtype=torch.float32)
            elif case.get("D_shape") == "head_hdim":
                D = torch.linspace(
                    -0.5, 0.5, nheads * headdim, device=device, dtype=torch.float32
                ).reshape(nheads, headdim)
            z = torch.randn_like(x) if case.get("z") else None
            out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
            # ROCm's optional buffer-op lowering cannot currently canonicalize
            # this kernel's valid runtime selection between ``states`` and
            # ``initial_states`` pointers. Disable that lowering only for this
            # correctness specialization; CUDA and performance are unchanged.
            restore_buffer_ops = None
            if case.get("disable_rocm_buffer_ops") and torch.version.hip is not None:
                import triton
                restore_buffer_ops = triton.knobs.amd.use_buffer_ops
                triton.knobs.amd.use_buffer_ops = False
            try:
                mod.chunk_scan_fwd(
                    cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx,
                    D=D, z=z, initial_states=initial_states,
                )
            finally:
                if restore_buffer_ops is not None:
                    triton.knobs.amd.use_buffer_ops = restore_buffer_ops
            torch.cuda.synchronize()
            ref = reference_chunk_scan(
                cb, x, dt, dA_cumsum, C, states, cu, seq_idx,
                D=D, z=z, initial_states=initial_states,
            ).to(device)
            if not torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2):
                diff = (out.float() - ref.float()).abs().max().item()
                return False, f"Case {case_name}: max diff = {diff:.6f}"
        except Exception as e:
            return False, f"Case {case_name}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []
    for test_idx, (seqlen, nheads, headdim, ngroups, dstate, chunk_size) in enumerate(TEST_SHAPES):
        try:
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
            def _bench_fn():
                mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "seqlen": seqlen,
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
                    "seqlen": seqlen,
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
