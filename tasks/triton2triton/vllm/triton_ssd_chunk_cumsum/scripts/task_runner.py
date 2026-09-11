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

# Correctness-only cases that cover non-uniform chunk masks, non-power-of-two
# dimensions, clamp limits, lower precision, and non-contiguous strides without
# changing the performance workload.
CORRECTNESS_EDGE_CASES = [
    {
        "name": "variable_chunks_odd_heads_clamped",
        "seqlen": 121,
        "nheads": 7,
        "chunk_size": 48,
        "cu_chunk_seqlens": (0, 48, 79, 121),
        "has_bias": True,
        "softplus": False,
        "dt_limit": (0.01, 0.06),
        "dtype": "float32",
        "strided": False,
    },
    {
        "name": "short_chunks_strided_float16",
        "seqlen": 67,
        "nheads": 5,
        "chunk_size": 33,
        "cu_chunk_seqlens": (0, 1, 34, 58, 67),
        "has_bias": True,
        "softplus": True,
        "dt_limit": (0.0, float("inf")),
        "dtype": "float16",
        "strided": True,
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


def ref_softplus(x):
    import torch
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def reference(dt, A, chunk_size, cu, dt_bias, softplus, dt_limit=(0.0, float("inf"))):
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
            dt_chunk = dt_chunk.clamp(min=dt_limit[0], max=dt_limit[1])
            dt_out[h, c, :clen] = dt_chunk
            dA = dt_out[h, c] * A_f[h]
            dA_cumsum[h, c] = torch.cumsum(dA, 0)
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

    cases = []
    for i, (seqlen, nheads, chunk_size, has_bias, softplus) in enumerate(TEST_SHAPES):
        nchunks = seqlen // chunk_size
        cases.append({
            "name": f"baseline_{i}",
            "seqlen": seqlen,
            "nheads": nheads,
            "chunk_size": chunk_size,
            "cu_chunk_seqlens": tuple(j * chunk_size for j in range(nchunks + 1)),
            "has_bias": has_bias,
            "softplus": softplus,
            "dt_limit": (0.0, float("inf")),
            "dtype": "float32",
            "strided": False,
        })
    cases.extend(CORRECTNESS_EDGE_CASES)

    for i, case in enumerate(cases):
        try:
            seqlen = case["seqlen"]
            nheads = case["nheads"]
            chunk_size = case["chunk_size"]
            has_bias = case["has_bias"]
            softplus = case["softplus"]
            dt_limit = case["dt_limit"]
            dtype = getattr(torch, case["dtype"])
            torch.manual_seed(42 + i)
            if case["strided"]:
                dt_storage = torch.randn(
                    seqlen * 2, nheads * 2, device=device, dtype=dtype
                ) * 0.1
                dt = dt_storage[::2, ::2]
                A_storage = -torch.rand(
                    nheads * 2, device=device, dtype=torch.float32
                ) * 0.5
                A = A_storage[::2]
                if has_bias:
                    dt_bias_storage = torch.randn(
                        nheads * 2, device=device, dtype=torch.float32
                    ) * 0.01
                    dt_bias = dt_bias_storage[::2]
                else:
                    dt_bias = None
            else:
                dt = torch.randn(seqlen, nheads, device=device, dtype=dtype) * 0.1
                A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
                dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.01 if has_bias else None
            cu = torch.tensor(case["cu_chunk_seqlens"], device=device, dtype=torch.int32)
            dA_cs, dt_out = mod.chunk_cumsum_fwd(
                dt, A, chunk_size, cu, dt_bias=dt_bias,
                dt_softplus=softplus, dt_limit=dt_limit,
            )
            ref_dA, ref_dt = reference(
                dt, A, chunk_size, cu, dt_bias, softplus, dt_limit,
            )
            ref_dA = ref_dA.to(device)
            ref_dt = ref_dt.to(device)
            if not torch.allclose(dA_cs, ref_dA, atol=1e-3, rtol=1e-3):
                diff = (dA_cs - ref_dA).abs().max().item()
                return False, f"{case['name']} dA_cumsum: max diff={diff}"
            if not torch.allclose(dt_out, ref_dt, atol=1e-3, rtol=1e-3):
                diff = (dt_out - ref_dt).abs().max().item()
                return False, f"{case['name']} dt_out: max diff={diff}"
        except Exception as e:
            return False, f"{case['name']}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (seqlen, nheads, chunk_size, has_bias, softplus) in enumerate(TEST_SHAPES):
        try:
            nchunks = seqlen // chunk_size
            torch.manual_seed(42 + test_idx)
            dt = torch.randn(seqlen, nheads, device=device, dtype=torch.float32) * 0.1
            A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
            dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.01 if has_bias else None
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            def _bench_fn():
                mod.chunk_cumsum_fwd(dt, A, chunk_size, cu, dt_bias=dt_bias, dt_softplus=softplus)
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
                    "chunk_size": chunk_size,
                    "has_bias": has_bias,
                    "softplus": softplus
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seqlen": seqlen,
                    "nheads": nheads,
                    "chunk_size": chunk_size,
                    "has_bias": has_bias,
                    "softplus": softplus
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
