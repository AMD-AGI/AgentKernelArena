#!/usr/bin/env python3
"""Harness for the vLLM Triton decode paged-attention kernel
``kernel_paged_attention_2d`` as MiniMax-M3-MXFP4 drives it (block_size=128).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits to chunked_prefill_paged_decode.py take effect.

Shapes, dtypes and context lengths come from the 20260815T100002Z session trace;
see session_cases.json. Correctness and performance run the SAME case list at the
SAME sizes, and performance is measured under a CUDA/HIP graph.
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
# resolves against the installed vllm package.
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_chunked_prefill_paged_decode"

# q/k/v are slices of one fused qkv buffer in the model, so the query row stride
# is 8*128 (q) + 128 (k) + 128 (v) = 1280, not 8*128. Reproduced here because a
# non-unit row stride changes the kernel's global-load pattern.
QKV_ROW_STRIDE = 1280


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.chdir(WORKSPACE)


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
    import vllm.v1.attention.ops.prefix_prefill  # noqa: F401

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _compile_smoke_case(case: dict) -> dict:
    """Small stand-in used ONLY by the compile smoke test.

    run_correctness and run_performance both iterate the full CASES list at the
    session sizes; this shrink exists so `compile` stays a few-second check.
    """
    smoke = {**case, "params": dict(case["params"])}
    smoke["params"]["num_seqs"] = min(case["params"]["num_seqs"], 8)
    smoke["params"]["num_padded_tokens"] = min(
        case["params"]["num_padded_tokens"], 8
    )
    smoke["params"]["ctx_len"] = min(case["params"]["ctx_len"], 512)
    smoke["params"]["num_blocks"] = 512
    return smoke


def _make(case: dict) -> dict:
    torch = _torch()
    p = dict(case["params"])
    num_seqs = p["num_seqs"]
    num_padded = max(p["num_padded_tokens"], num_seqs)
    num_query_heads = p["num_query_heads"]
    num_kv_heads = p["num_kv_heads"]
    head_size = p["head_size"]
    block_size = p["block_size"]
    ctx_len = p["ctx_len"]
    dtype = torch.bfloat16
    scale = head_size**-0.5

    torch.manual_seed(23)

    # Fused qkv buffer -> query/key/value are strided views, as in the model.
    row = max(QKV_ROW_STRIDE, (num_query_heads + 2 * num_kv_heads) * head_size)
    qkv = torch.randn((num_padded, row), device="cuda", dtype=dtype)
    query = qkv[:, : num_query_heads * head_size].view(
        num_padded, num_query_heads, head_size
    )
    out_buf = torch.empty(
        (num_padded, num_query_heads * head_size), device="cuda", dtype=dtype
    )
    output = out_buf.view(num_padded, num_query_heads, head_size)

    # Decode batch: one query token per sequence, occupying rows [0, num_seqs).
    query_start_loc = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((num_seqs,), ctx_len, device="cuda", dtype=torch.int32)

    # Contiguous per-sequence context KV. Drives both the paged cache the kernel
    # reads and the reference, so the two always describe the same workload.
    key = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )
    value = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )

    # Allocate the session's real cache size and scatter each sequence's pages
    # across it. A tight per-sequence range would let the 298 MB working set sit
    # in the 256 MB Infinity Cache across graph replays and understate the cost
    # by ~3x (measured: 0.38 ms contiguous vs 1.14 ms fragmented, against the
    # session's 1.168 ms).
    pages_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = max(p.get("num_blocks", 0), num_seqs * pages_per_seq + 1)

    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size),
        device="cuda",
        dtype=dtype,
    )
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv_cache, num_kv_heads, head_size
    )

    gen = torch.Generator(device="cpu").manual_seed(77)
    block_table = (
        torch.randperm(num_blocks, generator=gen)[: num_seqs * pages_per_seq]
        .view(num_seqs, pages_per_seq)
        .to("cuda", torch.int32)
        .contiguous()
    )

    seq_idx = torch.arange(num_seqs, device="cuda").view(-1, 1).expand(-1, ctx_len)
    pos = torch.arange(ctx_len, device="cuda").view(1, -1).expand(num_seqs, -1)
    phys_block = block_table[seq_idx, pos // block_size].long()
    slot_mapping = (phys_block * block_size + (pos % block_size)).reshape(-1)

    one = torch.ones(1, device="cuda", dtype=torch.float32)

    inputs = {
        "cfg": case,
        "module": _load_kernel_module(),
        "qkv": qkv,
        "query": query,
        "output": output,
        "out_buf": out_buf,
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
        "slot_mapping": slot_mapping,
        "num_seqs": num_seqs,
        "num_kv_heads": num_kv_heads,
        "num_query_heads": num_query_heads,
        "head_size": head_size,
    }
    _fill_kv_cache(inputs)
    return inputs


def _fill_kv_cache(inputs: dict) -> None:
    """Page the contiguous key/value into the cache the kernel reads."""
    import vllm._custom_ops as ops

    num_kv_heads = inputs["num_kv_heads"]
    head_size = inputs["head_size"]
    ops.reshape_and_cache(
        inputs["key"].reshape(-1, num_kv_heads, head_size),
        inputs["value"].reshape(-1, num_kv_heads, head_size),
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["slot_mapping"],
        "auto",
        inputs["one"],
        inputs["one"],
    )


def _perturb_inputs(inputs: dict) -> None:
    """Refresh data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes.
    """
    torch = _torch()
    torch.manual_seed(37)
    inputs["qkv"].normal_()
    inputs["key"].normal_()
    inputs["value"].normal_()
    _fill_kv_cache(inputs)


def _run(inputs: dict):
    inputs["module"].chunked_prefill_paged_decode(
        query=inputs["query"][: inputs["num_seqs"]],
        key=None,
        value=None,
        output=inputs["output"][: inputs["num_seqs"]],
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
    return inputs["output"][: inputs["num_seqs"]]


def _reference(inputs: dict):
    """Fully vectorized decode attention reference (no per-sequence loop).

    GQA is handled by folding the group width into the query tensor rather than
    broadcasting the KV heads, so the expanded KV is never materialized. Peak
    intermediate is the score tensor [S, Hkv, group, ctx] in fp32, e.g.
    64*1*8*9216*4 = 18.9 MB for the largest case.
    """
    torch = _torch()
    S = inputs["num_seqs"]
    Hkv = inputs["num_kv_heads"]
    D = inputs["head_size"]
    group = inputs["num_query_heads"] // Hkv

    q = inputs["query"][:S].float().view(S, Hkv, group, D)  # [S, Hkv, g, D]
    k = inputs["key"].float()  # [S, ctx, Hkv, D]
    v = inputs["value"].float()

    scores = torch.einsum("shgd,skhd->shgk", q, k) * inputs["scale"]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("shgk,skhd->shgd", probs, v)
    return out.reshape(S, Hkv * group, D).to(inputs["output"].dtype)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=0.08, rtol=0.08)


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed."""
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    inputs["out_buf"].fill_(float("nan"))
    _assert_close(inputs, timed.rerun())


def run_compile() -> None:
    inputs = _make(_compile_smoke_case(CASES[0]))
    _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        _assert_close(inputs, got)
        print("correctness PASS", case["id"])
        del inputs, got
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch.cuda.synchronize()
        timed = _TimedRun()
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=10,
            repetition=100,
            target_ms=1.0,
            max_graph_repeats=1000,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "session_id": case.get("session_id"),
            "phase": case.get("phase"),
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
        del inputs
        torch.cuda.empty_cache()
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
