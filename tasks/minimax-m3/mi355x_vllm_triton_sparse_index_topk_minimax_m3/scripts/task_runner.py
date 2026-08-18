#!/usr/bin/env python3
"""Harness for MiniMax-M3's sparse block-selection chain
(vllm/models/minimax_m3/amd/ops/index_topk.py).

Two scored paths, both driven through their public launchers:

  decode  : minimax_m3_index_decode()  -> _decode_index_score_kernel
                                          + _topk_index_partial_kernel
                                          + _topk_index_merge_kernel
  prefill : minimax_m3_index_score()   -> _index_block_score_kernel
            minimax_m3_index_topk()    -> _topk_index_kernel

Shapes, dtypes and sequence lengths come from the 20260815T100002Z session trace;
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
MODEL_CONFIG = SPEC["model_config"]

REPO_SUBDIR = "vllm_models_minimax_m3"
KERNEL_FILE = Path("amd") / "ops" / "index_topk.py"
EDIT_MODULE_NAME = "vllm.models.minimax_m3.amd.ops._ka_index_topk"

_FORCE_INIT = 1e30
_FORCE_LOCAL = 1e29
# Reference chunk over the query dimension for the prefill path; bounds the
# [chunk, max_block*128] fp32 score matrix.
_REF_Q_CHUNK = 4096


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
    import vllm.models.minimax_m3.common.ops.index_topk  # noqa: F401

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _compile_smoke_case(case: dict) -> dict:
    p = dict(case["params"])
    p["num_blocks"] = 512
    if p["path"] == "decode":
        p["num_reqs"] = min(p["num_reqs"], 8)
        p["total_q"] = p["num_reqs"]
        p["seq_len"] = min(p["seq_len"], 1024)
    else:
        p["query_lens"] = [min(q, 256) for q in p["query_lens"]]
        p["seq_lens"] = [min(s, 512) for s in p["seq_lens"]]
        p["prefix_lens"] = [
            min(pf, s - q)
            for pf, s, q in zip(p["prefix_lens"], p["seq_lens"], p["query_lens"])
        ]
    return {**case, "params": p}


def _make(case: dict) -> dict:
    torch = _torch()
    p = dict(case["params"])
    dev = "cuda"
    dtype = torch.bfloat16
    blk = p["sparse_block_size"]
    D = p["index_head_dim"]
    H = p["num_idx_heads"]
    max_blocks = (p["max_seq_len"] + blk - 1) // blk

    torch.manual_seed(23)

    if p["path"] == "decode":
        batch = p["num_reqs"]
        total_q = p["total_q"]
        seq_lens_list = [p["seq_len"]] * batch
        prefix_lens_list = [p["seq_len"] - 1] * batch  # query is the last token
        query_lens = [1] * batch
    else:
        batch = p["batch"]
        query_lens = list(p["query_lens"])
        seq_lens_list = list(p["seq_lens"])
        prefix_lens_list = list(p["prefix_lens"])
        total_q = sum(query_lens)

    # The page pool must cover every sequence's full block_table row.
    num_blocks = max(p["num_blocks"], batch * max_blocks + 1)

    # Scaled down so bf16 dot products stay in a sane range; the kernel and the
    # reference see the same values either way.
    idx_q = torch.randn((total_q, H, D), device=dev, dtype=dtype) * 0.25
    index_kv_cache = torch.randn((num_blocks, blk, D), device=dev, dtype=dtype) * 0.25

    gen = torch.Generator(device="cpu").manual_seed(77)
    perm = torch.randperm(num_blocks, generator=gen)[: batch * max_blocks]
    block_table = perm.view(batch, max_blocks).to(dev, torch.int32).contiguous()

    seq_lens_t = torch.tensor(seq_lens_list, device=dev, dtype=torch.int32)
    prefix_lens_t = torch.tensor(prefix_lens_list, device=dev, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(batch + 1, device=dev, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(query_lens, device=dev, dtype=torch.int32).cumsum(0)

    # Stable output buffer, as vLLM's persistent topk_indices_buffer provides.
    out_buf = torch.empty((H, total_q, p["topk"]), device=dev, dtype=torch.int32)

    return {
        "cfg": case,
        "params": p,
        "module": _load_kernel_module(),
        "idx_q": idx_q,
        "index_kv_cache": index_kv_cache,
        "block_table": block_table,
        "seq_lens": seq_lens_t,
        "prefix_lens": prefix_lens_t,
        "cu_seqlens_q": cu_seqlens_q,
        "out_buf": out_buf,
        "batch": batch,
        "total_q": total_q,
        "query_lens": query_lens,
        "seq_lens_list": seq_lens_list,
        "prefix_lens_list": prefix_lens_list,
        "max_query_len": max(query_lens),
    }


def _perturb_inputs(inputs: dict) -> None:
    torch = _torch()
    torch.manual_seed(37)
    inputs["idx_q"].normal_().mul_(0.25)
    inputs["index_kv_cache"].normal_().mul_(0.25)


def _run(inputs: dict):
    p = inputs["params"]
    m = inputs["module"]
    if p["path"] == "decode":
        return m.minimax_m3_index_decode(
            inputs["idx_q"],
            inputs["index_kv_cache"],
            inputs["block_table"],
            inputs["seq_lens"],
            p["max_seq_len"],
            p["topk"],
            p["init_blocks"],
            p["local_blocks"],
            p["num_idx_heads"],
            p["decode_query_len"],
            p["decode_query_len"],
            out=inputs["out_buf"],
        )
    score = m.minimax_m3_index_score(
        inputs["idx_q"],
        inputs["index_kv_cache"],
        inputs["block_table"],
        inputs["cu_seqlens_q"],
        inputs["seq_lens"],
        inputs["prefix_lens"],
        inputs["max_query_len"],
        p["max_seq_len"],
        p["num_idx_heads"],
    )
    return m.minimax_m3_index_topk(
        score,
        inputs["cu_seqlens_q"],
        inputs["prefix_lens"],
        inputs["max_query_len"],
        p["topk"],
        p["init_blocks"],
        p["local_blocks"],
        out=inputs["out_buf"],
    )


def _reference_scores(inputs: dict):
    """Per-(head, query, block) score, matching both kernels' semantics.

    score[h, q, blk] = max over the 128 positions of blk of (k . q), with
    positions masked out when ``pos > absolute_query_position`` (causal) or
    ``pos >= seq_len``. Blocks beyond the query's causal reach get -inf.
    Forced blocks then override: init -> 1e30, local -> 1e29
    (index_topk.py:390-392 for decode; _topk_index_kernel takes init/local
    directly on the prefill path -- same selection either way).

    Vectorized over (queries in a chunk) x blocks x positions x heads; the only
    Python loops are over the batch and over query chunks.
    """
    torch = _torch()
    p = inputs["params"]
    blk = p["sparse_block_size"]
    D = p["index_head_dim"]
    H = p["num_idx_heads"]
    cache = inputs["index_kv_cache"]
    max_block = (p["max_seq_len"] + blk - 1) // blk

    scores = torch.full(
        (H, inputs["total_q"], max_block), float("-inf"), device=cache.device,
        dtype=torch.float32,
    )
    nvis_all = torch.empty(inputs["total_q"], device=cache.device, dtype=torch.long)

    q_off = 0
    for b, q_len in enumerate(inputs["query_lens"]):
        seq_len = inputs["seq_lens_list"][b]
        prefix = inputs["prefix_lens_list"][b]
        n_blk = (seq_len + blk - 1) // blk
        pages = inputs["block_table"][b, :n_blk].long()
        # [n_blk*blk, D] contiguous view of this sequence's index-K
        k = cache[pages].reshape(n_blk * blk, D).float()
        pos = torch.arange(n_blk * blk, device=cache.device)

        for c0 in range(0, q_len, _REF_Q_CHUNK):
            c1 = min(c0 + _REF_Q_CHUNK, q_len)
            n = c1 - c0
            g0 = q_off + c0
            qpos = prefix + torch.arange(c0, c1, device=cache.device)  # [n]
            nvis = (qpos // blk) + 1
            nvis_all[g0 : g0 + n] = nvis

            qc = inputs["idx_q"][g0 : g0 + n].float()  # [n, H, D]
            s = torch.einsum("nhd,pd->hnp", qc, k)  # [H, n, n_blk*blk]
            ok = (pos.view(1, 1, -1) <= qpos.view(1, n, 1)) & (
                pos.view(1, 1, -1) < seq_len
            )
            s = s.masked_fill(~ok, float("-inf"))
            s = s.view(H, n, n_blk, blk).amax(dim=-1)  # [H, n, n_blk]

            blocks = torch.arange(n_blk, device=cache.device).view(1, 1, -1)
            visible = blocks < nvis.view(1, n, 1)
            is_init = (blocks < p["init_blocks"]) & visible
            is_local = (blocks >= (nvis.view(1, n, 1) - p["local_blocks"])) & visible
            s = torch.where(is_local, torch.full_like(s, _FORCE_LOCAL), s)
            s = torch.where(is_init, torch.full_like(s, _FORCE_INIT), s)
            s = s.masked_fill(~visible, float("-inf"))
            scores[:, g0 : g0 + n, :n_blk] = s
            del s, ok, qc
        del k
        q_off += q_len
    return scores, nvis_all


def _assert_close(inputs: dict, got) -> None:
    """Validate the selected block set, tie-safely.

    Top-k over floats has no unique answer when scores tie, so comparing indices
    element-by-element against torch.topk would be wrong. Instead we require, per
    (head, query): the right number of entries, all distinct and causally
    visible, and a selected-score multiset equal to torch's top-k multiset.
    """
    torch = _torch()
    p = inputs["params"]
    topk = p["topk"]
    scores, nvis = _reference_scores(inputs)
    H, Q, _ = scores.shape
    got = got[:, : inputs["total_q"], :].to(torch.int64)

    expect_n = torch.clamp(nvis, max=topk)  # [Q]
    n_sel = (got >= 0).sum(dim=-1)  # [H, Q]
    if not torch.equal(n_sel, expect_n.view(1, Q).expand(H, Q)):
        bad = (n_sel != expect_n.view(1, Q)).nonzero()[:5].tolist()
        raise AssertionError(
            f"wrong number of selected blocks at (head,query) {bad}; "
            f"got {n_sel.flatten()[:8].tolist()} expected "
            f"{expect_n[:8].tolist()}"
        )

    safe = got.clamp(min=0)
    if (safe >= nvis.view(1, Q, 1)).logical_and(got >= 0).any():
        raise AssertionError("selected a block outside the causally visible range")
    # distinctness among the valid entries
    marked = torch.where(got >= 0, safe, torch.arange(topk, device=got.device) + 10**6)
    if (marked.sort(dim=-1).values.diff(dim=-1) == 0).any():
        raise AssertionError("duplicate block ids in the selection")

    got_scores = torch.gather(scores, 2, safe)
    got_scores = torch.where(
        got >= 0, got_scores, torch.full_like(got_scores, float("-inf"))
    )
    ref_scores = scores.topk(topk, dim=-1).values
    got_scores, _ = got_scores.sort(dim=-1, descending=True)
    ref_scores, _ = ref_scores.sort(dim=-1, descending=True)
    finite = torch.isfinite(ref_scores)
    torch.testing.assert_close(
        got_scores[finite], ref_scores[finite], atol=2e-2, rtol=2e-2
    )


def _assert_timed_outputs(inputs: dict, timed) -> None:
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    inputs["out_buf"].fill_(-1)
    _assert_close(inputs, timed.rerun())


def run_compile() -> None:
    for case in (CASES[0], next(c for c in CASES if c["params"]["path"] == "prefill")):
        inputs = _make(_compile_smoke_case(case))
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
