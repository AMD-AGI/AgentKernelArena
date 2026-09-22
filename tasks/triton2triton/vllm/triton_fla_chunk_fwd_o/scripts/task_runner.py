#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fla_chunk_fwd_o"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)
from scripts.contract_checks import InputSnapshot, check_outputs, validate_timed
from scripts import semantic_controls

TASK_NAME = "triton2triton/triton_fla_chunk_fwd_o"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fla_chunk_fwd_o.py")

SEEDS = [42, 43, 44, 45, 46]
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
PERF_SEED_IDX = 2


_LOADED_MODULES = []


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED_MODULES.append(mod)
    return mod


def reference(q, k, v, h, g=None, scale=None, chunk_size=64):
    import torch
    from math import ceil
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = ceil(T / BT)
    if scale is None:
        scale = K ** -0.5
    q_c, k_c, v_c, h_c = q.float().cpu(), k.float().cpu(), v.float().cpu(), h.float().cpu()
    g_c = g.float().cpu() if g is not None else None
    o = torch.zeros(B, T, H, V)
    for b in range(B):
        for hh in range(H):
            for t in range(NT):
                s = t * BT
                e = min(s + BT, T)
                bq = q_c[b, s:e, hh]
                bk = k_c[b, s:e, hh]
                bv = v_c[b, s:e, hh]
                bh = h_c[b, t, hh]  # [V, K]
                # inter: q @ h^T
                o_inter = bq @ bh.T
                # intra: (q @ k^T) * causal_mask @ v
                A = bq @ bk.T
                if g_c is not None:
                    bg = g_c[b, s:e, hh]
                    o_inter = o_inter * torch.exp(bg)[:, None]
                    A = A * torch.exp(bg[:, None] - bg[None, :])
                sz = e - s
                mask = torch.tril(torch.ones(sz, sz))
                A = A * mask
                o[b, s:e, hh] = o_inter * scale + (A @ bv) * scale
    return o


def gen_inputs(seed, device):
    import torch
    torch.manual_seed(seed)
    B, T, H, K, V, BT = 1, 64, 2, 32, 32, 64
    from math import ceil
    NT = ceil(T / BT)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32) * 0.1
    h = torch.randn(B, NT, H, V, K, device=device, dtype=torch.float32) * 0.1
    g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.01
    return (q, k, v, h, g), {"chunk_size": BT}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "chunk_fwd_o"), "Missing chunk_fwd_o"
        assert hasattr(mod, "chunk_fwd_kernel_o"), "Missing chunk_fwd_kernel_o"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness(*, case_index=None):
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    for i, seed in enumerate(SEEDS):
        if case_index is not None and i != case_index:
            continue
        try:
            args, kwargs = gen_inputs(seed, device)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            readonly = InputSnapshot({str(i): a for i,a in enumerate(args) if isinstance(a, torch.Tensor)})
            result = mod.chunk_fwd_o(*args, **kwargs)
            readonly.check()
            check_outputs(result, reference(*args, **kwargs).to(device), atol=5e-2, rtol=5e-2,
                          inputs=[e[1] for e in readonly.entries])
            ref = reference(*args_cpu, **kwargs)
            r_cpu = result.float().cpu()
            ref_f = ref.float()
            if not torch.allclose(r_cpu, ref_f, atol=5e-2, rtol=5e-2):
                max_diff = (r_cpu - ref_f).abs().max().item()
                return False, f"Shape {i+1}: max diff = {max_diff:.6f}"

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, seed in enumerate(SEEDS):
        try:
            args, kwargs = gen_inputs(seed, device)

            readonly = InputSnapshot({str(i): a for i,a in enumerate(args) if isinstance(a, torch.Tensor)})
            from _aka_benchmark import TimedRun
            timed = TimedRun()
            def _bench_fn():
                return mod.chunk_fwd_o(*args, **kwargs)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                timed_run=timed,
            )
            benchmark_metadata.update(validate_timed(
                timed, readonly, lambda: reference(*args, **kwargs).to(device),
                lambda: args[0].mul_(8.0), atol=5e-2, rtol=5e-2))

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "seed": seed,
                    "chunk_size": kwargs.get("chunk_size", 64)
                }
            })
        except Exception as exc:
            test_cases.append({
                "error": f"{type(exc).__name__}: {exc}",
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seed": seed,
                    "chunk_size": kwargs.get("chunk_size", 64)
                }
            })
    return test_cases


def run_reference_controls():
    return semantic_controls.reference_controls(sys.modules[__name__])


def run_semantic_controls():
    return semantic_controls.run_controls(load_module(), device="cuda")


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        run_semantic_controls()
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(SEEDS)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
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
