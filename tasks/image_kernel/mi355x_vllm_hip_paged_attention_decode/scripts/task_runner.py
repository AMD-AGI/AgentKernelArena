#!/usr/bin/env python3
"""Harness for the ROCm custom paged-attention decode kernels
``paged_attention_ll4mi_QKV_mfma16_kernel`` + ``paged_attention_ll4mi_reduce_kernel``
(AITER ``csrc/cpp_itfs/pa``), reached through ``aiter.paged_attention_rocm``.

The kernels are JIT specialized from the editable workspace copy of the in-image
AITER source tree: ``AITER_META_DIR`` points the importable ``csrc`` package and
``AITER_CORE_DIR`` at ``<workspace>/aiter_meta``, and ``AITER_REBUILD`` clears the
template-op build cache so an agent's edit to ``pa_kernels.cuh`` / ``pa.cuh`` /
``pa_common.cuh`` / ``pa.cpp.jinja`` is recompiled before it is measured.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]

REPO_SUBDIR = "aiter_meta"
PARTITION_SIZE = 256

# Profiling is a single-shape probe, pinned rather than derived from timings so
# the profiled kernel never drifts between runs. GQA 4:1 at head_size 128 /
# block_size 16 is the most common decode geometry in this suite (shared by
# Llama-3.1-8B-Instruct and Qwen3-8B). Correctness and performance still sweep
# every case in CASES.
PROFILE_CASE_ID = SPEC.get("profile_case") or CASES[0]["id"]


def _sources_edited() -> bool:
    """True unless every editable source still matches its in-image original.

    Fails safe: anything we cannot positively verify counts as edited. Serving a
    prebuilt .so for an edited kernel would silently benchmark the ORIGINAL, so
    a false "unedited" is far worse than a redundant rebuild.
    """
    try:
        import yaml

        cfg = yaml.safe_load((WORKSPACE / "config.yaml").read_text()) or {}
        image_root = Path(str(cfg["image_repo_path"]))
        sources = cfg["source_file_path"] or []
        if isinstance(sources, str):
            sources = [sources]
        if not sources or not image_root.is_dir():
            return True
        for rel in sources:
            ours = WORKSPACE / REPO_SUBDIR / str(rel)
            if ours.read_bytes() != (image_root / str(rel)).read_bytes():
                return True
        return False
    except Exception:  # noqa: BLE001 - unverifiable means "assume edited"
        return True


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")

    # compile_template_op caches purely by template arguments, so an edited
    # kernel would otherwise keep serving the previously built lib.so. Clearing
    # the cache is what makes a source edit take effect. AgentKernelArena also
    # injects AITER_REBUILD=1 per build subprocess (src/jit_rebuild.py); the
    # default here keeps standalone runs honest.
    #
    # Only force it once something has actually been edited. aiter treats any
    # non-zero AITER_REBUILD as "rebuild this module on first use", without
    # checking the source, and the profiler re-spawns this driver once per
    # counter pass, so an unedited baseline would re-enter the rebuild path
    # over and over for a kernel that never changed.
    if _sources_edited():
        os.environ.setdefault("AITER_REBUILD", "1")
    # Keep the template-op build cache inside the workspace instead of the
    # shared ~/.aiter, so parallel runs cannot serve each other's kernels.
    os.environ.setdefault("AITER_ROOT_DIR", str(WORKSPACE / "build" / "aiter_root"))

    repo = WORKSPACE / REPO_SUBDIR
    if repo.is_dir():
        # aiter/jit/core.py puts this on sys.path, which is what makes
        # `import csrc.cpp_itfs.pa.pa` resolve to the workspace copy; pa.py then
        # derives AITER_CORE_DIR from its own location, so the jinja template and
        # every include come from the same editable tree.
        os.environ["AITER_META_DIR"] = str(repo)
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


def _import_aiter():
    import aiter

    seeded = os.environ.get("AITER_META_DIR")
    if seeded:
        import csrc.cpp_itfs.utils as cpp_itfs_utils

        core = Path(cpp_itfs_utils.AITER_CORE_DIR).resolve()
        if core != Path(seeded).resolve():
            raise RuntimeError(
                "AITER template ops resolved to "
                f"{core}, not the editable workspace tree {seeded}; "
                "kernel edits would be silently ignored."
            )
    return aiter


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
    differ from the checked one. ``ctx_len`` also sets the split-K partition
    count, so a shortened context changes which reduction path the kernel takes.
    """
    torch = _torch()
    aiter = _import_aiter()
    params = dict(case["params"])
    num_seqs = params["num_seqs"]
    ctx_len = params["ctx_len"]
    num_query_heads = params["num_query_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_size = params["head_size"]
    block_size = params["block_size"]
    dtype = torch.bfloat16
    scale = head_size**-0.5

    torch.manual_seed(29)

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

    # Paged KV cache in the vLLM ROCm layout: (2, num_blocks, block_size,
    # num_kv_heads, head_size), split into the 5D key / 4D value views that
    # paged_attention_rocm expects.
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

    one = torch.ones(1, device="cuda", dtype=torch.float32)

    # Split-K scratch, sized exactly as vLLM sizes it in
    # chunked_prefill_paged_decode.py before calling paged_attention_rocm.
    max_num_partitions = (ctx_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    tmp_output = torch.empty(
        (num_seqs, num_query_heads, max_num_partitions, head_size),
        device="cuda",
        dtype=dtype,
    )
    exp_sums = torch.empty(
        (num_seqs, num_query_heads, max_num_partitions),
        device="cuda",
        dtype=torch.float32,
    )
    max_logits = torch.empty_like(exp_sums)

    inputs = {
        "cfg": case,
        "aiter": aiter,
        "query": query,
        "output": output,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "seq_lens": torch.full(
            (num_seqs,), ctx_len, device="cuda", dtype=torch.int32
        ),
        "exp_sums": exp_sums,
        "max_logits": max_logits,
        "tmp_output": tmp_output,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
        "max_seq_len": ctx_len,
        "scale": scale,
        "one": one,
        "slot_mapping": slot_mapping,
    }
    _fill_kv_cache(inputs)
    return inputs


def _fill_kv_cache(inputs: dict) -> None:
    """Page the contiguous key/value into the cache the kernel reads.

    The reference reads ``key``/``value`` while the kernel reads the paged cache,
    so the two have to be refreshed together or they stop describing the same
    workload.
    """
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
    """Refresh the data inputs in place with values no earlier launch has seen.

    A replayed CUDA graph reads the captured input addresses, so writing through
    them changes what the scored kernel consumes. Fresh values stop an output
    buffer that the kernel never wrote from matching the reference by accident.
    """
    torch = _torch()
    torch.manual_seed(71)
    inputs["query"].normal_()
    inputs["key"].normal_()
    inputs["value"].normal_()
    _fill_kv_cache(inputs)


def _run(inputs: dict):
    inputs["aiter"].paged_attention_rocm(
        inputs["output"],
        inputs["exp_sums"],
        inputs["max_logits"],
        inputs["tmp_output"],
        inputs["query"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["num_kv_heads"],
        inputs["scale"],
        inputs["block_table"],
        inputs["seq_lens"],
        inputs["block_size"],
        inputs["max_seq_len"],
        None,
        "auto",
        inputs["one"],
        inputs["one"],
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
        # implementation that leaves elements untouched cannot pass by reusing
        # whatever the allocator handed back.
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
