#!/usr/bin/env python3
"""Harness for vLLM's Triton attention kernel ``kernel_unified_attention``
(``vllm/v1/attention/ops/triton_unified_attention.py``).

The kernel is loaded from the editable workspace copy of the in-image source tree
so an optimizing agent's edits to triton_unified_attention.py take effect; Triton
re-keys its JIT on the source, so no explicit rebuild step is needed.

This is vLLM's kernel, not AITER's ``kernel_unified_attention_2d``/``_3d`` - see
``session_cases.json`` ``provenance.kernel_ownership``.
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
KERNEL_FILE = "triton_unified_attention.py"
# Dotted name so the edited copy is importable under the installed vllm package;
# every import in the file is absolute, so it resolves against the install.
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_triton_unified_attention"

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs. The sliding/head_size-256 decode
# shape is the session's largest single leaf at 19.55% of GPU time. Correctness
# and performance still sweep every case in CASES.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]


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
    # Import the installed module first so the vllm.v1.attention.ops package (the
    # parent of the edited module) is initialized.
    import vllm.v1.attention.ops.triton_unified_attention  # noqa: F401

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    if not path.is_file():
        raise RuntimeError(f"seeded kernel source not found: {path}")
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def profile_case() -> dict:
    """The single case profiling runs against (see PROFILE_CASE_ID)."""
    for case in CASES:
        if case["id"] == PROFILE_CASE_ID:
            return case
    raise KeyError(
        f"profile_case {PROFILE_CASE_ID!r} is not present in session_cases.json"
    )


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one. It matters twice over on this operator: the
    sliding window only clips once ``ctx_len`` exceeds it, so a shortened context
    silently turns a windowed layer into a full-attention one.
    """
    torch = _torch()
    params = dict(case["params"])
    num_seqs = params["num_seqs"]
    ctx_len = params["ctx_len"]
    num_query_heads = params["num_query_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_size = params["head_size"]
    block_size = params["block_size"]
    sliding_window = params["sliding_window"]
    dtype = torch.bfloat16
    scale = head_size**-0.5

    # TritonAttentionImpl maps a causal SWA layer to (window-1, 0); a full
    # attention layer disables the window entirely.
    window_size = (sliding_window - 1, 0) if sliding_window else (-1, -1)

    torch.manual_seed(31)

    # Decode: one query token per sequence.
    query = torch.randn(
        (num_seqs, num_query_heads, head_size), device="cuda", dtype=dtype
    )
    output = torch.empty_like(query)

    # Contiguous per-sequence context KV, used both to fill the paged cache and
    # to compute the reference.
    key = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )
    value = torch.randn(
        (num_seqs, ctx_len, num_kv_heads, head_size), device="cuda", dtype=dtype
    )

    pages_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = num_seqs * pages_per_seq + 1

    # TritonAttentionBackend.get_kv_cache_shape:
    # (num_blocks, 2, block_size, num_kv_heads, head_size), unbound on dim 1.
    kv_cache = torch.zeros(
        (num_blocks, 2, block_size, num_kv_heads, head_size),
        device="cuda",
        dtype=dtype,
    )
    block_table = torch.arange(
        num_seqs * pages_per_seq, device="cuda", dtype=torch.int32
    ).view(num_seqs, pages_per_seq)

    seq_idx = torch.arange(num_seqs, device="cuda").view(-1, 1).expand(-1, ctx_len)
    pos = torch.arange(ctx_len, device="cuda").view(1, -1).expand(num_seqs, -1)
    phys_block = block_table[seq_idx, pos // block_size].long().reshape(-1)
    offset = (pos % block_size).reshape(-1)
    key_cache, value_cache = kv_cache.unbind(1)

    # Non-quantized KV still receives expanded per-(seq, kv_head) descales, the
    # same way TritonAttentionImpl.forward builds them from layer._k_scale.
    descale = torch.ones(
        (num_seqs, num_kv_heads), device="cuda", dtype=torch.float32
    )

    inputs = {
        "cfg": case,
        "module": _load_kernel_module(),
        "query": query,
        "output": output,
        "key": key,
        "value": value,
        "kv_cache": kv_cache,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "cu_seqlens_q": torch.arange(
            num_seqs + 1, device="cuda", dtype=torch.int32
        ),
        "seqused_k": torch.full(
            (num_seqs,), ctx_len, device="cuda", dtype=torch.int32
        ),
        "descale": descale,
        "ctx_len": ctx_len,
        "sliding_window": sliding_window,
        "window_size": window_size,
        "scale": scale,
        "phys_block": phys_block,
        "offset": offset,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
    }
    _fill_kv_cache(inputs)
    return inputs


def _fill_kv_cache(inputs: dict) -> None:
    """Page the contiguous key/value into the cache the kernel reads.

    The reference reads ``key``/``value`` while the kernel reads the paged cache,
    so the two have to be refreshed together or they stop describing the same
    workload.
    """
    num_kv_heads = inputs["num_kv_heads"]
    head_size = inputs["head_size"]
    kv_cache = inputs["kv_cache"]
    kv_cache[inputs["phys_block"], 0, inputs["offset"]] = inputs["key"].reshape(
        -1, num_kv_heads, head_size
    )
    kv_cache[inputs["phys_block"], 1, inputs["offset"]] = inputs["value"].reshape(
        -1, num_kv_heads, head_size
    )


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. Fresh values stop an output
    buffer that the kernel never wrote from matching the reference by accident.
    """
    torch = _torch()
    torch.manual_seed(67)
    inputs["query"].normal_()
    inputs["key"].normal_()
    inputs["value"].normal_()
    _fill_kv_cache(inputs)


def _run(inputs: dict):
    inputs["module"].unified_attention(
        q=inputs["query"],
        k=inputs["key_cache"],
        v=inputs["value_cache"],
        out=inputs["output"],
        cu_seqlens_q=inputs["cu_seqlens_q"],
        max_seqlen_q=1,
        seqused_k=inputs["seqused_k"],
        max_seqlen_k=inputs["ctx_len"],
        softmax_scale=inputs["scale"],
        causal=True,
        window_size=inputs["window_size"],
        block_table=inputs["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=inputs["descale"],
        v_descale=inputs["descale"],
    )
    return inputs["output"]


def _reference(inputs: dict):
    torch = _torch()
    query = inputs["query"].float()  # (S, num_query_heads, head_size)
    key = inputs["key"].float()  # (S, ctx, num_kv_heads, head_size)
    value = inputs["value"].float()
    ctx_len = inputs["ctx_len"]
    window = inputs["sliding_window"]
    # The single decode query sits at position ctx_len-1, so a (window-1, 0)
    # causal window admits exactly the last `window` context tokens.
    lo = max(0, ctx_len - window) if window else 0
    ratio = query.shape[1] // key.shape[2]
    outputs = []
    for s in range(query.shape[0]):
        k = key[s, lo:].repeat_interleave(ratio, dim=1)
        v = value[s, lo:].repeat_interleave(ratio, dim=1)
        scores = torch.einsum("hd,khd->hk", query[s], k) * inputs["scale"]
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probs, v))
    return torch.stack(outputs).to(inputs["output"].dtype)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=0.02, rtol=0.02)


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Correctness and performance must share one
    shape, otherwise the scored path is not the validated path.
    """
    params = dict(case["params"])
    params["num_seqs"] = min(params["num_seqs"], 8)
    params["ctx_len"] = min(params["ctx_len"], 256)
    return {**case, "params": params}


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against freshly perturbed
    inputs and checks the buffer it wrote, so work the scored path skips cannot
    hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    inputs["output"].fill_(float("nan"))
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
        # The output buffer is written, never accumulated: poison it so an
        # implementation that leaves rows untouched cannot pass on allocator
        # leftovers.
        inputs["output"].fill_(float("nan"))
        got = _run(inputs)
        torch.cuda.synchronize()
        _assert_close(inputs, got)
        print("correctness PASS", case["id"])


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        _torch().cuda.synchronize()
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
