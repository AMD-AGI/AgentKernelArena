#!/usr/bin/env python3
"""Harness for the vLLM Triton decode paged-attention kernel
``kernel_paged_attention_2d`` (chunked_prefill_paged_decode.py).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits to chunked_prefill_paged_decode.py take effect.
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
KERNEL_FILE = "chunked_prefill_paged_decode.py"
# Dotted name so the edited copy's relative `from .prefix_prefill import ...`
# resolves against the installed vllm package (prefill path is untouched here).
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_chunked_prefill_paged_decode"


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
    # Ensure the installed vllm package (and the parent of the edited module) is
    # importable so the edited copy's relative import resolves.
    import vllm.v1.attention.ops.prefix_prefill  # noqa: F401

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
    num_seqs = min(params["num_seqs"], 8) if correctness else params["num_seqs"]
    ctx_len = min(params["ctx_len"], 256) if correctness else params["ctx_len"]
    num_query_heads = params["num_query_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_size = params["head_size"]
    block_size = params["block_size"]
    dtype = torch.bfloat16
    scale = head_size**-0.5

    torch.manual_seed(23)

    # Decode: one query token per sequence.
    query = torch.randn(
        (num_seqs, num_query_heads, head_size), device="cuda", dtype=dtype
    )
    output = torch.empty_like(query)
    query_start_loc = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((num_seqs,), ctx_len, device="cuda", dtype=torch.int32)

    # Contiguous per-sequence context KV, used to fill the paged cache and to
    # compute the reference.
    key = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )
    value = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )

    pages_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = num_seqs * pages_per_seq + 1

    # Paged KV cache in the vLLM ROCm layout: (2, num_blocks, block_size,
    # num_kv_heads, head_size), split into 5D key / 4D value views.
    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size),
        device="cuda",
        dtype=dtype,
    )
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv_cache, num_kv_heads, head_size
    )

    block_table = torch.arange(
        num_seqs * pages_per_seq, device="cuda", dtype=torch.int32
    ).view(num_seqs, pages_per_seq)

    # slot = physical_block * block_size + offset, in (seq, pos) row-major order.
    seq_idx = torch.arange(num_seqs, device="cuda").view(-1, 1).expand(-1, ctx_len)
    pos = torch.arange(ctx_len, device="cuda").view(1, -1).expand(num_seqs, -1)
    phys_block = block_table[seq_idx, pos // block_size].long()
    slot_mapping = (phys_block * block_size + (pos % block_size)).reshape(-1)

    import vllm._custom_ops as ops

    one = torch.ones(1, device="cuda", dtype=torch.float32)
    ops.reshape_and_cache(
        key.reshape(-1, num_kv_heads, head_size),
        value.reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        one,
        one,
    )

    return {
        "cfg": case,
        "module": _load_kernel_module(),
        "query": query,
        "output": output,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "max_seq_len": ctx_len,
        "scale": scale,
        "one": one,
    }


def _run(inputs: dict):
    inputs["module"].chunked_prefill_paged_decode(
        query=inputs["query"],
        key=None,
        value=None,
        output=inputs["output"],
        kv_cache_dtype="auto",
        key_cache=inputs["key_cache"],
        value_cache=inputs["value_cache"],
        block_table=inputs["block_table"],
        query_start_loc=inputs["query_start_loc"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=inputs["max_seq_len"],
        max_query_len=1,
        k_scale=inputs["one"],
        v_scale=inputs["one"],
        sm_scale=inputs["scale"],
        causal=True,
    )
    return inputs["output"]


def _reference(inputs: dict):
    torch = _torch()
    query = inputs["query"].float()  # (S, num_query_heads, head_size)
    key = inputs["key"].float()  # (S, ctx, num_kv_heads, head_size)
    value = inputs["value"].float()
    scale = inputs["scale"]
    ratio = query.shape[1] // key.shape[2]
    outputs = []
    for s in range(query.shape[0]):
        k = key[s].repeat_interleave(ratio, dim=1)  # (ctx, num_query_heads, hs)
        v = value[s].repeat_interleave(ratio, dim=1)
        scores = torch.einsum("hd,khd->hk", query[s], k) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probs, v))
    return torch.stack(outputs).to(inputs["output"].dtype)


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
