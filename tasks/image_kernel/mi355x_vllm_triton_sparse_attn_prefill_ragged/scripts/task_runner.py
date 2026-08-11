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

# Queries processed per reference chunk. Bounds the gathered
# (chunk, per_q, head_dim) float buffer, which dominates reference memory.
_REFERENCE_QUERY_CHUNK = 512

REPO_SUBDIR = "vllm_v1_attention_ops"
KERNEL_FILE = "rocm_aiter_mla_sparse.py"
EDIT_MODULE_NAME = "vllm.v1.attention.ops._ka_rocm_aiter_mla_sparse"


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


def _make(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one.
    """
    torch = _torch()
    params = dict(case["params"])
    num_queries = params["num_queries"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    nope_head_dim = params["nope_head_dim"]
    rope_head_dim = params["rope_head_dim"]
    num_kv = params["num_kv"]
    topk = params["topk"]
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
    """Sparse MLA attention reference: gather the selected KV, then dense attend.

    ``_make`` emits a uniform CSR (every query selects the same ``per_q``
    positions), so this batches into gather + einsum instead of walking queries in
    Python. The loop it replaces synchronised on ``indptr`` once per query, which
    is what made a full-shape correctness run impractical.

    Queries are chunked because the gathered KV is ``(chunk, per_q, head_dim)``
    floats, which reaches tens of GB at the scored shape if done in one go.
    """
    torch = _torch()
    q = inputs["q"]  # (sq, H, D) bf16
    kv = inputs["kv"].float()  # (skv, D)
    indptr = inputs["indptr"]
    scale = inputs["scale"]

    sq = q.shape[0]
    widths = (indptr[1:] - indptr[:-1]).unique()
    assert widths.numel() == 1, "reference expects a uniform CSR from _make"
    per_q = int(widths.item())
    sel = inputs["indices"].view(sq, per_q).long()

    out = torch.empty_like(q)
    chunk = max(1, min(sq, _REFERENCE_QUERY_CHUNK))
    for start in range(0, sq, chunk):
        stop = min(start + chunk, sq)
        kv_sel = kv[sel[start:stop]]  # (c, per_q, D), latent is both K and V
        scores = (
            torch.einsum("qhd,qkd->qhk", q[start:stop].float(), kv_sel) * scale
        )
        probs = torch.softmax(scores, dim=-1)
        out[start:stop] = torch.einsum("qhk,qkd->qhd", probs, kv_sel).to(out.dtype)
    return out


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=0.08, rtol=0.08)


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. The CSR selection is structure
    rather than data, so it stays fixed.
    """
    torch = _torch()
    torch.manual_seed(47)
    inputs["q"].normal_()
    inputs["kv"].normal_()


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Correctness and performance must share one
    shape, otherwise the scored path is not the validated path.
    """
    smoke_case = {**case, "params": dict(case["params"])}
    smoke_case["params"]["num_queries"] = min(case["params"]["num_queries"], 32)
    smoke_case["params"]["num_kv"] = min(case["params"]["num_kv"], 1024)
    smoke_case["params"]["topk"] = min(case["params"]["topk"], 128)
    return smoke_case


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against freshly perturbed
    inputs and checks the buffer it wrote, so work that the scored path skips
    cannot hide behind a correctness call that took a different branch.
    """
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
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
