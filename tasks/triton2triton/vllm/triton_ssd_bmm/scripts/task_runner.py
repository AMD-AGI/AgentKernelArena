#!/usr/bin/env python3
"""Task runner for triton_ssd_bmm"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)
from scripts.contract_checks import InputSnapshot, check_outputs, validate_timed
from scripts import semantic_controls
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_ssd_bmm.py")

# (seqlen, ngroups, k, chunk_size, causal)
TEST_SHAPES = [
    (128, 2, 32, 64, False),
    (256, 4, 64, 64, True),
    (512, 2, 32, 128, False),
    (256, 8, 16, 64, True),
    (384, 4, 64, 128, False),
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

_LOADED_MODULES = []


def load_module():
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED_MODULES.append(mod)
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


def run_correctness(*, case_index=None):
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Load failed: {e}"
    device = "cuda"
    for i, (seqlen, ngroups, k, chunk_size, causal) in enumerate(TEST_SHAPES):
        if case_index is not None and i != case_index:
            continue
        try:
            torch.manual_seed(42 + i)
            nchunks = seqlen // chunk_size
            a = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            b = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            readonly = InputSnapshot(dict(a=a, b=b, cu=cu))
            result = mod.bmm_chunk_fwd(a, b, chunk_size, cu, causal=causal)
            readonly.check()
            check_outputs(result, reference_bmm(a, b, chunk_size, cu, causal).to(device), atol=1e-1, rtol=1e-1,
                          inputs=[e[1] for e in readonly.entries])
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
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (seqlen, ngroups, k, chunk_size, causal) in enumerate(TEST_SHAPES):
        try:
            nchunks = seqlen // chunk_size
            torch.manual_seed(42 + test_idx)
            a = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            b = torch.randn(seqlen, ngroups, k, device=device, dtype=torch.float16)
            cu = torch.arange(0, nchunks + 1, device=device, dtype=torch.int32) * chunk_size
            readonly = InputSnapshot(dict(a=a, b=b, cu=cu))
            from _aka_benchmark import TimedRun
            timed = TimedRun()
            def _bench_fn():
                return mod.bmm_chunk_fwd(a, b, chunk_size, cu, causal=causal)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
                timed_run=timed,
            )
            benchmark_metadata.update(validate_timed(
                timed, readonly, lambda: reference_bmm(a, b, chunk_size, cu, causal).to(device),
                lambda: a.neg_(), atol=1e-1, rtol=1e-1))

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "seqlen": seqlen,
                    "ngroups": ngroups,
                    "k": k,
                    "chunk_size": chunk_size,
                    "causal": causal
                }
            })
        except Exception as exc:
            test_cases.append({
                "error": f"{type(exc).__name__}: {exc}",
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "seqlen": seqlen,
                    "ngroups": ngroups,
                    "k": k,
                    "chunk_size": chunk_size,
                    "causal": causal
                }
            })
    return test_cases


def run_reference_controls():
    return semantic_controls.reference_controls(sys.modules[__name__])


def run_semantic_controls():
    return semantic_controls.run_controls(load_module(), device="cuda")


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
        run_semantic_controls()
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
