#!/usr/bin/env python3
"""Harness for MiniMax-M3's block-sparse GQA prefill attention kernel
``_gqa_sparse_fwd_kernel`` (vllm/models/minimax_m3/amd/ops/sparse_attn.py).

The kernel is loaded from the editable workspace copy of the in-image source
tree, so agent edits to amd/ops/sparse_attn.py take effect. Everything the module
imports (common.ops.sparse_attn, vllm.platforms.rocm, vllm.triton_utils) is an
absolute import and keeps resolving against the installed vllm package.

Shapes, dtypes, sequence/prefix lengths and the launch geometry all come from the
20260815T100002Z session trace; see session_cases.json. Correctness and
performance run the SAME case list at the SAME sizes, and performance is measured
under a CUDA/HIP graph.
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
KERNEL_FILE = Path("amd") / "ops" / "sparse_attn.py"
EDIT_MODULE_NAME = "vllm.models.minimax_m3.amd.ops._ka_sparse_attn"

# Reference chunk size over the query dimension. The gathered KV for one chunk is
# chunk * topk * 128 * 256 elements; 128 keeps the fp32 working set around 400 MB.
_REF_Q_CHUNK = 128


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
    # Import the package first so the edited copy's absolute imports resolve.
    import vllm.models.minimax_m3.common.ops.sparse_attn  # noqa: F401

    path = WORKSPACE / REPO_SUBDIR / KERNEL_FILE
    spec = importlib.util.spec_from_file_location(EDIT_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[EDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _compile_smoke_case(case: dict) -> dict:
    """Tiny stand-in used ONLY by the compile smoke test.

    run_correctness and run_performance both iterate the full CASES list at the
    session sizes.
    """
    p = dict(case["params"])
    p["query_lens"] = [min(q, 256) for q in p["query_lens"]]
    p["seq_lens"] = [min(s, 512) for s in p["seq_lens"]]
    p["prefix_lens"] = [
        min(pf, s - q) for pf, s, q in zip(p["prefix_lens"], p["seq_lens"], p["query_lens"])
    ]
    p["num_blocks"] = 512
    return {**case, "params": p}


def _make_topk_idx(torch, params, device):
    """Deterministic, causally valid top-k block selection.

    For query i of sequence b (absolute position ``prefix_lens[b] + i``) the
    visible block count is ``ceil((pos+1)/128)``. We keep the local block (M3 sets
    sparse_local_block=1) and fill the rest with a seeded random subset of the
    remaining visible blocks, sorted ascending and right-padded with -1 -- the
    layout the kernel expects (``real_topk = sum(topk_idx >= 0)``, entries read
    sequentially).
    """
    topk = params["topk"]
    blk = params["sparse_block_size"]
    gen = torch.Generator(device=device).manual_seed(1234)

    rows = []
    for b, (q_len, prefix) in enumerate(zip(params["query_lens"], params["prefix_lens"])):
        pos = prefix + torch.arange(q_len, device=device)
        n_vis = (pos // blk) + 1  # [q_len]
        local = n_vis - 1  # block containing the query
        max_vis = int(n_vis.max().item())
        # Random priority per (query, block); invisible blocks get +inf so they
        # sort last, the local block gets the lowest priority so it always
        # survives the cut.
        pri = torch.rand((q_len, max_vis), generator=gen, device=device)
        blocks = torch.arange(max_vis, device=device).unsqueeze(0)
        pri = torch.where(blocks < n_vis.unsqueeze(1), pri, torch.inf)
        # -1.0, not -inf: the finite check below is what separates "selected"
        # from "padding", and -inf would classify the forced local block as
        # padding. Queries below position 128 have the local block as their only
        # visible block, so that would hand the kernel an empty selection.
        pri.scatter_(1, local.unsqueeze(1), -1.0)
        order = pri.argsort(dim=1)[:, :topk]  # [q_len, topk]
        chosen = order.to(torch.int32)
        keep = torch.gather(pri, 1, order).isfinite()
        chosen = torch.where(keep, chosen, torch.full_like(chosen, 2**30))
        chosen, _ = chosen.sort(dim=1)
        chosen = torch.where(
            chosen == 2**30, torch.full_like(chosen, -1), chosen
        )
        rows.append(chosen)
    return torch.cat(rows, dim=0).unsqueeze(0).contiguous()  # [1, total_q, topk]


def _make(case: dict) -> dict:
    torch = _torch()
    p = dict(case["params"])
    dev = "cuda"
    dtype = torch.bfloat16

    batch = p["batch"]
    query_lens = list(p["query_lens"])
    seq_lens = list(p["seq_lens"])
    prefix_lens = list(p["prefix_lens"])
    total_q = sum(query_lens)
    H, KVH, D = p["num_heads"], p["num_kv_heads"], p["head_dim"]
    blk = p["sparse_block_size"]
    max_blocks = (MODEL_CONFIG["max_model_len"] + blk - 1) // blk
    # The page pool must cover every sequence's full block_table row.
    num_blocks = max(p["num_blocks"], batch * max_blocks + 1)

    torch.manual_seed(23)

    q = torch.randn((total_q, H, D), device=dev, dtype=dtype)
    output = torch.empty_like(q)
    # Main KV cache: [num_blocks, num_kv_heads, 128, 2*head_dim], K then V.
    kv_cache = torch.randn(
        (num_blocks, KVH, blk, 2 * D), device=dev, dtype=dtype
    )

    # Non-contiguous page assignment, as a live server's allocator produces.
    gen = torch.Generator(device="cpu").manual_seed(77)
    perm = torch.randperm(num_blocks, generator=gen)[: batch * max_blocks]
    block_table = perm.view(batch, max_blocks).to(dev, torch.int32).contiguous()

    cu_seqlens_q = torch.zeros(batch + 1, device=dev, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(query_lens, device=dev, dtype=torch.int32).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, device=dev, dtype=torch.int32)
    prefix_lens_t = torch.tensor(prefix_lens, device=dev, dtype=torch.int32)

    topk_idx = _make_topk_idx(torch, p, dev)

    return {
        "cfg": case,
        "params": p,
        "module": _load_kernel_module(),
        "q": q,
        "output": output,
        "kv_cache": kv_cache,
        "topk_idx": topk_idx,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seq_lens": seq_lens_t,
        "prefix_lens": prefix_lens_t,
        "query_lens": query_lens,
        "seq_lens_list": seq_lens,
        "prefix_lens_list": prefix_lens,
        "max_query_len": max(query_lens),
        "num_kv_heads": KVH,
        "num_heads": H,
        "head_dim": D,
        "sm_scale": D**-0.5,
        "sparse_block_size": blk,
    }


def _perturb_inputs(inputs: dict) -> None:
    """Refresh data inputs in place so a replayed graph consumes fresh values."""
    torch = _torch()
    torch.manual_seed(37)
    inputs["q"].normal_()
    inputs["kv_cache"].normal_()


def _run(inputs: dict):
    inputs["module"].minimax_m3_sparse_attn(
        inputs["q"],
        inputs["kv_cache"],
        inputs["topk_idx"],
        inputs["block_table"],
        inputs["cu_seqlens_q"],
        inputs["seq_lens"],
        inputs["prefix_lens"],
        inputs["max_query_len"],
        inputs["num_kv_heads"],
        inputs["sm_scale"],
        inputs["output"],
    )
    return inputs["output"]


def _reference(inputs: dict):
    """Block-sparse GQA attention reference.

    Vectorized over (queries in a chunk) x (selected blocks) x (positions in a
    block) x (heads); the only Python loop is the chunking over the query
    dimension, which exists to bound the gathered-KV working set.

    Semantics taken from the kernel body (sparse_attn.py:119-235):
      * absolute query position = prefix_lens[b] + i
      * a selected block ``blk`` covers KV positions ``blk*128 + [0,128)``
      * a position is valid iff ``pos < seq_lens[b]`` (kernel: pos_mask_sub) AND
        ``pos <= prefix + i`` (kernel: off_q_sub >= c, i.e. causal)
      * K is kv_cache[page, kvh, :, :head_dim], V is [..., head_dim:]
    """
    torch = _torch()
    H = inputs["num_heads"]
    KVH = inputs["num_kv_heads"]
    D = inputs["head_dim"]
    blk = inputs["sparse_block_size"]
    scale = inputs["sm_scale"]
    group = H // KVH
    kv_cache = inputs["kv_cache"]
    topk = inputs["topk_idx"].shape[-1]

    out = torch.empty(
        (inputs["q"].shape[0], H, D), device=inputs["q"].device, dtype=torch.float32
    )
    pos_in_blk = torch.arange(blk, device=kv_cache.device)

    q_off = 0
    for b, q_len in enumerate(inputs["query_lens"]):
        prefix = inputs["prefix_lens_list"][b]
        seq_len = inputs["seq_lens_list"][b]
        bt = inputs["block_table"][b].long()
        for c0 in range(0, q_len, _REF_Q_CHUNK):
            c1 = min(c0 + _REF_Q_CHUNK, q_len)
            n = c1 - c0
            g0 = q_off + c0

            idx = inputs["topk_idx"][:, g0 : g0 + n, :]  # [KVH, n, topk]
            valid = idx >= 0
            safe = idx.clamp(min=0).long()
            pages = bt[safe]  # [KVH, n, topk]

            kvh_ids = torch.arange(KVH, device=kv_cache.device).view(KVH, 1, 1)
            kv = kv_cache[pages, kvh_ids.expand_as(pages)]  # [KVH,n,topk,blk,2D]
            k = kv[..., :D].float()
            v = kv[..., D:].float()

            # [KVH, n, topk, blk] absolute KV positions and their validity
            pos = safe.unsqueeze(-1) * blk + pos_in_blk
            qpos = (prefix + torch.arange(c0, c1, device=kv_cache.device)).view(
                1, n, 1, 1
            )
            ok = valid.unsqueeze(-1) & (pos < seq_len) & (pos <= qpos)

            qc = inputs["q"][g0 : g0 + n].float().view(n, KVH, group, D)
            qc = qc.permute(1, 0, 2, 3)  # [KVH, n, group, D]

            # [KVH, n, group, topk*blk]
            scores = torch.einsum("hngd,hntpd->hngtp", qc, k)
            scores = scores.reshape(KVH, n, group, topk * blk) * scale
            mask = ok.reshape(KVH, n, 1, topk * blk)
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

            vv = v.reshape(KVH, n, topk * blk, D)
            o = torch.einsum("hngs,hnsd->hngd", probs, vv)  # [KVH,n,group,D]
            out[g0 : g0 + n] = o.permute(1, 0, 2, 3).reshape(n, H, D)

            del kv, k, v, scores, probs, vv, o
        q_off += q_len

    return out.to(inputs["output"].dtype)


def _assert_close(inputs: dict, got) -> None:
    _torch().testing.assert_close(got, _reference(inputs), atol=0.08, rtol=0.08)


def _assert_timed_outputs(inputs: dict, timed) -> None:
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
            **{k: v for k, v in case["params"].items()},
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
