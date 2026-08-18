#!/usr/bin/env python3
"""Image-kernel harness for the GLM-5.2 DSA sparse-MLA TileLang kernel.

Reproduces the highest-E2E operator of Hyperloom session
GLM-5.2-MXFP4_20260814T163244Z on MI355X/gfx950.

The timed callable is ``tilelang_sparse_fwd`` -- one MLA layer's worth of DSA
attention, i.e. the ``sparse_mla_fwd_decode_partial`` + ``sparse_mla_fwd_decode_combine``
pair. Both TileLang prim_funcs are named ``main``, so roctracer records both as
``main_kernel``; in the session trace they alternate 66.43 us / 4.21 us at 78
pairs per decode step.

Both phases go through this same entry point. server.log line 12 of the scored
run reads:

    Set DSA backends for bfloat16 KV Cache: prefill=tilelang, decode=tilelang

so the prefill cases below are the real prefill path, not a decode kernel run at
a prefill shape.

Two things this harness is deliberate about:

  * **Index locality is part of the workload.** The kernel is gather-bound
    (34.4 TFLOP/s = 1.4% of peak; 2.27 TB/s of KV gather at the decode shape).
    Drawing top-k slots uniformly from the whole 1.81M-slot pool measures a
    regime the model never enters -- it came out ~52% slower than the
    session-faithful pattern at the prefill shape (11.13 vs 7.32 ms/layer).
    ``_make`` therefore draws every query's top-k causally from its own
    sequence's contiguous slot range, and sorts it, which is what sglang's
    ``topk_transform`` emits.
  * **Padding is exercised.** Queries at position p < 2048 get min(p+1, 2048)
    valid entries and -1 for the rest; the kernel masks negatives out of the
    softmax (tilelang_kernel.py:913). The prefill cases hit this on their first
    2048 positions, so a patch that drops the mask fails correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]
COMMON = SPEC["params_common"]

HEADS = COMMON["heads"]
DIM = COMMON["dim"]
TAIL_DIM = COMMON["tail_dim"]
TOPK = COMMON["topk"]
LATENT = DIM + TAIL_DIM
KV_POOL_SLOTS = COMMON["kv_pool_slots"]
SM_SCALE = LATENT**-0.5

# Queries per reference chunk. The gathered buffer is (chunk, TOPK, 576) in fp32,
# so 128 keeps it near 600 MB while staying wide enough that the reference is a
# handful of large matmuls rather than a Python loop.
_REFERENCE_QUERY_CHUNK = 128


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # The DSA backend selection in tilelang_sparse_fwd is gated on aiter being
    # active, exactly as in the session (EXTRA_ENV=SGLANG_USE_AITER=1).
    os.environ.setdefault("SGLANG_USE_AITER", "1")
    # Prefer the workspace-seeded copy of sglang so the agent's edits to
    # kernels/ops/attention/dsa/tilelang_kernel.py take effect. TileLang JIT
    # recompiles on source change.
    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python"))
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


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _write_report(rows: list[dict]) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _load_entry():
    """Resolve ``tilelang_sparse_fwd`` from the (possibly edited) workspace copy.

    Fails closed if the import resolved past the seeded copy: sglang is an
    editable install with a PEP 660 finder, and if the workspace copy ever stops
    winning, an agent's edits become invisible and every number this harness
    prints describes the original kernel.
    """
    import sglang

    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        resolved = Path(sglang.__file__).resolve()
        if seeded.resolve() not in resolved.parents:
            raise RuntimeError(
                f"sglang resolved to {resolved}, not the workspace copy under "
                f"{seeded}. Source edits would be ignored."
            )
    return tilelang_sparse_fwd


def _make(case: dict) -> dict:
    """Build one case at its scored shape.

    There is no correctness/performance switch: the shape that is timed is the
    shape that is validated. Only ``compile`` shrinks anything, via
    ``_compile_smoke_case``.
    """
    torch = _torch()
    p = case["params"]
    num_seqs = int(p["num_seqs"])
    q_per_seq = int(p["q_per_seq"])
    ctx_len = int(p["ctx_len"])
    pool = int(p.get("kv_pool_slots", KV_POOL_SLOTS))
    topk = int(p.get("topk", TOPK))
    seq = num_seqs * q_per_seq

    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(p["seed"]))

    q = torch.randn(
        (seq, HEADS, LATENT), device="cuda", dtype=torch.bfloat16, generator=gen
    )
    kv = torch.randn(
        (pool, 1, LATENT), device="cuda", dtype=torch.bfloat16, generator=gen
    )

    # Each sequence owns a contiguous slot range in the paged pool.
    base = torch.randint(
        0, max(1, pool - ctx_len), (num_seqs,), device="cuda", dtype=torch.int64,
        generator=gen,
    )
    sid = torch.arange(seq, device="cuda") // q_per_seq
    # Absolute position of each query inside its own sequence. A prefill chunk
    # covers the whole sequence; a decode step appends one token at the end.
    within = torch.arange(seq, device="cuda") % q_per_seq
    pos = within + (ctx_len - q_per_seq)
    n_valid = torch.clamp(pos + 1, max=topk)

    # Causal draw from [0, pos] of the query's own sequence, then sorted.
    #
    # The sort is not cosmetic. sglang's top-k selector
    # (kernels/aot/csrc/elementwise/topk.hip) emits its 2048 slots in ascending
    # order: `naive_topk_transform` (context <= 2048) copies the page table
    # straight through, and `fast_topk_cuda_tl` is a radix-select that appends
    # via atomicAdd while threads scan idx in strided order, so the output is
    # monotone at block granularity. Leaving the draw unsorted scatters each
    # 64-slot tile over the sequence's whole KV range and inflates the prefill
    # case by 13% (8.30 ms/layer vs 7.31); block-shuffling within 64 measures the
    # same as a full sort, so a plain sort is the faithful and simpler choice.
    u = torch.rand((seq, topk), device="cuda", generator=gen)
    sel_pos = (u * (pos + 1)[:, None].to(u.dtype)).long().clamp_(max=ctx_len - 1)
    sel_pos, _ = sel_pos.sort(dim=1)
    slots = (base[sid][:, None] + sel_pos).to(torch.int32)
    ar = torch.arange(topk, device="cuda")[None, :]
    indices = torch.where(
        ar < n_valid[:, None], slots, torch.full_like(slots, -1)
    ).unsqueeze(1).contiguous()

    return {
        "cfg": case,
        "entry": _load_entry(),
        "q": q,
        "kv": kv,
        "indices": indices,
        "sm_scale": SM_SCALE,
        "d_v": DIM,
    }


def _run(inputs: dict):
    return inputs["entry"](
        q=inputs["q"],
        kv=inputs["kv"],
        indices=inputs["indices"],
        sm_scale=inputs["sm_scale"],
        d_v=inputs["d_v"],
    )


def _reference(inputs: dict):
    """Dense-gather sparse MLA reference.

    Vectorized: the top-k set is gathered per chunk of queries and the two
    contractions run as batched einsums, so the whole reference is a few dozen
    large matmuls rather than a per-query Python loop. Chunking bounds the
    gathered (chunk, topk, 576) fp32 buffer.

    Matches the kernel semantics read off tilelang_kernel.py:900-968:
      * K is the full 576-dim latent, V is dims [0:512] of the SAME tensor;
      * ``sm_scale`` multiplies the raw QK dot before the softmax;
      * a negative index is masked (-inf) and, if a whole row is masked, the
        kernel divides by 1 so the output is 0 rather than NaN.
    """
    torch = _torch()
    q = inputs["q"]
    kv = inputs["kv"]
    idx = inputs["indices"]
    scale = inputs["sm_scale"]

    seq = q.shape[0]
    sel_all = idx[:, 0]  # (S, topk) int32, negatives = padding
    out = torch.empty((seq, HEADS, DIM), device=q.device, dtype=q.dtype)

    chunk = max(1, min(seq, _REFERENCE_QUERY_CHUNK))
    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        sel = sel_all[start:stop].long()
        valid = sel >= 0
        gathered = kv[sel.clamp_(min=0), 0].float()  # (c, topk, 576)
        scores = torch.einsum(
            "qhd,qkd->qhk", q[start:stop].float(), gathered
        ).mul_(scale)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # An all-masked row softmaxes to NaN; the kernel emits 0 there.
        probs = torch.nan_to_num(probs, nan=0.0)
        out[start:stop] = torch.einsum(
            "qhk,qkd->qhd", probs, gathered[:, :, :DIM]
        ).to(out.dtype)
    return out


def _assert_close(inputs: dict, got) -> None:
    torch = _torch()
    ref = _reference(inputs)
    # tilelang_sparse_fwd keeps the leading batch dim; the reference does not.
    got = got.reshape(ref.shape)
    torch.testing.assert_close(got.float(), ref.float(), atol=0.05, rtol=0.05)


def _perturb_inputs(inputs: dict) -> None:
    """Refresh data through the captured addresses.

    A replayed CUDA graph reads the buffers it captured, so writing through them
    changes what the scored kernel consumes. The index set is workload structure,
    not data, so it stays fixed -- re-drawing it would change the locality the
    benchmark is measuring.
    """
    torch = _torch()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(47)
    inputs["q"].normal_(generator=gen)
    inputs["kv"].normal_(generator=gen)


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Correctness and performance share one shape.
    ``topk`` and the 576-dim latent are left alone because they change which
    TileLang kernel gets specialized (topk must stay 2048 -- see the assert in
    tilelang_sparse_fwd).
    """
    smoke = {**case, "params": dict(case["params"])}
    smoke["params"]["num_seqs"] = min(int(case["params"]["num_seqs"]), 2)
    smoke["params"]["q_per_seq"] = min(int(case["params"]["q_per_seq"]), 16)
    smoke["params"]["ctx_len"] = min(int(case["params"]["ctx_len"]), 4096)
    smoke["params"]["kv_pool_slots"] = 65536
    return smoke


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    ``run_correctness`` checks a separate call, which a kernel can tell apart
    from the scored one. This re-runs the timed unit against freshly perturbed
    inputs and checks the buffer it wrote.
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
        del inputs, got
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inputs = _make(case)
        _run(inputs)
        torch.cuda.synchronize()
        timed = _TimedRun()  # noqa: F821 - injected with the AKA-GENERATED block
        execution_time_ms, bench_meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=5,
            repetition=30,
            target_ms=1.0,
            max_graph_repeats=200,
            timed_run=timed,
        )
        _assert_timed_outputs(inputs, timed)
        metadata = {
            **case["params"],
            "regime": case.get("regime"),
            "model": SPEC.get("model"),
            "session_id": SPEC.get("session_id"),
            "gpu_pct": case.get("gpu_pct"),
            "measured_session_us": case.get("measured_session_us"),
            "measured_replay_us": case.get("measured_replay_us"),
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
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[
        mode
    ]()


if __name__ == "__main__":
    main()
