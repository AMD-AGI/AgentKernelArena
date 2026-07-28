#!/usr/bin/env python3
"""Harness for the vLLM Triton DeepSeek-V4 sparse-attention prefill kernel
``_sparse_attn_prefill_ragged_kernel`` (rocm_aiter_mla_sparse.py).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits take effect (Triton JIT recompiles on source change).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]

REPO_SUBDIR = "vllm_v1_attention_ops"
KERNEL_FILE = "rocm_aiter_mla_sparse.py"
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_rocm_aiter_mla_sparse"


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(fn, repetition):
    import time
    import torch

    repetition = max(1, int(repetition))
    if not torch.cuda.is_available():
        times_ms = []
        for _ in range(repetition):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
        return times_ms

    times_ms = []
    for _ in range(repetition):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=10,
    repetition=100,
    target_ms=1.0,
    n_retries=5,
    estimate_reps=5,
    max_graph_repeats=1000,
    use_cuda_graph=True,
    fallback_reason=None,
):
    import torch

    # A captured graph whose measured per-iteration time falls below this floor is
    # treated as empty (it recorded no device work) and rejected in favour of
    # per-launch event timing, so a graph that silently captured nothing is never
    # reported as a fabricated ~0 ms cuda_graph result. Real kernels measure far
    # above this floor; an empty-graph replay measures ~1e-5 ms.
    empty_graph_floor_ms = 1e-4

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    max_graph_repeats = max(1, int(max_graph_repeats))
    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }

    if not torch.cuda.is_available():
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_unavailable",
        })
        return sum(times) / len(times), metadata

    if not use_cuda_graph:
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_graph_disabled",
        })
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_reps = max(1, int(estimate_reps))
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(estimate_reps):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()

            estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
            if estimate_ms == 0:
                n_repeat = max_graph_repeats
            else:
                n_repeat = min(max_graph_repeats, max(1, int(float(target_ms) / estimate_ms)))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n_repeat):
                    fn()
            torch.cuda.synchronize()

            retry_times = []
            for _ in range(max(1, int(repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                retry_times.append(start_event.elapsed_time(end_event) / n_repeat)

        graph_mean = sum(retry_times) / len(retry_times)
        if graph_mean < empty_graph_floor_ms:
            times = _measure_cuda_event_fallback(fn, repetition)
            metadata.update({
                "benchmark_method": "cuda_event_fallback",
                "benchmark_effective_repeats": int(repetition),
                "benchmark_fallback_reason": fallback_reason or "empty_cuda_graph_capture",
            })
            return sum(times) / len(times), metadata

        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        return graph_mean, metadata
    except Exception as exc:
        # Isolate the aborted capture before re-measuring so the fallback timing is
        # not polluted by the failed attempt (a mid-capture failure can leave the
        # first few launches abnormally slow).
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for _ in range(min(3, max(1, int(warmup)))):
            fn()
        torch.cuda.synchronize()
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
        return sum(times) / len(times), metadata
# <<< AKA-GENERATED <<<


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def _load_kernel_module():
    # rocm_aiter_mla_sparse.py uses only absolute imports and registers no custom
    # ops at import time, so a straight file-path load of the edited workspace copy
    # is sufficient for the agent's edits to take effect.
    import vllm  # noqa: F401  (ensure platform init)

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _make(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    params = dict(case["params"])
    num_queries = min(params["num_queries"], 32) if correctness else params["num_queries"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    nope_head_dim = params["nope_head_dim"]
    rope_head_dim = params["rope_head_dim"]
    num_kv = min(params["num_kv"], 1024) if correctness else params["num_kv"]
    topk = min(params["topk"], 128) if correctness else params["topk"]
    per_q = min(topk, num_kv)
    dtype = torch.bfloat16
    scale = head_dim**-0.5

    torch.manual_seed(29)

    q = torch.randn(
        (num_queries, num_heads, head_dim), device="cuda", dtype=dtype
    )
    kv = torch.randn((num_kv, head_dim), device="cuda", dtype=dtype)

    # Ragged CSR sparse selection: each query attends to `per_q` KV positions.
    idx = torch.randint(
        0, num_kv, (num_queries, per_q), device="cuda"
    )
    idx, _ = idx.sort(dim=1)
    indices = idx.to(torch.int32).reshape(-1).contiguous()
    indptr = torch.arange(
        0, (num_queries + 1) * per_q, per_q, device="cuda", dtype=torch.int32
    )

    return {
        "cfg": case,
        "module": _load_kernel_module(),
        "q": q,
        "kv": kv,
        "indices": indices,
        "indptr": indptr,
        "scale": scale,
        "nope_head_dim": nope_head_dim,
        "rope_head_dim": rope_head_dim,
    }


def _run(inputs: dict):
    return inputs["module"]._rocm_sparse_attn_prefill_ragged_triton(
        inputs["q"],
        inputs["kv"],
        inputs["indices"],
        inputs["indptr"],
        inputs["scale"],
        None,  # attn_sink
        inputs["nope_head_dim"],
        inputs["rope_head_dim"],
    )


def _reference(inputs: dict):
    torch = _torch()
    q = inputs["q"].float()  # (sq, H, D)
    kv = inputs["kv"].float()  # (skv, D)
    indices = inputs["indices"]
    indptr = inputs["indptr"]
    scale = inputs["scale"]
    outputs = []
    for i in range(q.shape[0]):
        start = int(indptr[i].item())
        end = int(indptr[i + 1].item())
        sel = indices[start:end].long()
        kv_sel = kv[sel]  # (kv_len, D) — MLA latent used as both K and V
        scores = (q[i] @ kv_sel.t()) * scale  # (H, kv_len)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(probs @ kv_sel)  # (H, D)
    return torch.stack(outputs).to(torch.bfloat16)


def run_compile() -> None:
    inputs = _make(CASES[0], correctness=True)
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case, correctness=True)
        got = _run(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            got, _reference(inputs), atol=0.08, rtol=0.08
        )
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case, correctness=False)
        _run(inputs)
        _torch().cuda.synchronize()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
        )
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "session_id": case.get("session_id"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
            "benchmark_method": bench_meta.get("benchmark_method"),
        }
        metadata.update(
            {k: v for k, v in bench_meta.items() if k.startswith("benchmark_")}
        )
        rows.append(
            {
                "test_case_id": case["id"],
                "shape": case.get("trace_input_shapes"),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
        print(
            case["id"],
            f"{execution_time_ms:.6f} ms",
            bench_meta.get("benchmark_method"),
            bench_meta.get("benchmark_fallback_reason", ""),
        )
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["compile", "correctness", "performance", "manifest"]
    )
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
