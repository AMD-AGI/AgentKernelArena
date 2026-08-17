#!/usr/bin/env python3
"""Image-kernel harness for the Qwen3.8-2.4T-A95B paged GQA decode attention.

Reproduces the full-attention decode path of Hyperloom session 100137
(Qwen3.8-2.4T-A95B-Quark-MXFP4, 20260814T175123Z) on MI355X/gfx950, sglang
0.5.17.dev20260812+gdc5f6c4883 / ROCm 7.2.0 / aiter d9e5ef7ce.

23 of the model's 92 layers are ``full_attention``; the other 69 are Gated
DeltaNet. Two kernels per call, both measured in the session's 8-rank trace:

  paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, unsigned char,
      (vllm::Fp8KVCacheDataType)1, 1, 256, 256, false, 8, 1,
      ck_tile::ComposedAttention<...>>       23x/step  1.7119 ms/step  3.146% E2E
  paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16,
      256, 256, 256, 1, false>               23x/step  0.1149 ms/step  0.211% E2E

Running this harness on the session image dispatches those two template
instantiations byte-for-byte, at ~65 us/call against the trace's ~79 us/call.

What makes this configuration worth its own task rather than another case on the
generic paged-attention task: **head_size = 256**. Almost every served model uses
128. Combined with fp8_e4m3 KV and a GQA group of 8, this instantiation sees far
less tuning attention than the 128 one.

Per-rank geometry (TP=8), from config.json:
    num_attention_heads 64            -> 8 q heads/rank
    num_key_value_heads 4             -> padded to tp_size 8 by QKVParallelLinear,
                                         so 1 k head and 1 v head per rank
    head_dim 256, page_size 1, partition_size 256 (_AITER_PARTITION_SIZE_ROCM)
    kv_cache_dtype fp8_e4m3, k_scale = v_scale = 1.0, logits_soft_cap = 0.0
    scale = head_dim ** -0.5

The kv-head padding is not a guess: it is what makes qkv_proj's per-rank output
4608 = (64*2 + 8 + 8) * 256 / 8, which is the width the capture trace records.
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
PA = SPEC["pa_config"]

_IMAGE_AITER_ROOT = Path(os.environ.get("QWEN38_AITER_ROOT", "/sgl-workspace/aiter"))
# Only csrc is seeded for this task -- the edit surface is the HIP/asm tree, not
# the Python package -- so these two are linked in beside it.
_AITER_SIBLINGS = ("3rdparty", "hsa")


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    # aiter's JIT reads its C++/HIP sources from AITER_META_DIR/csrc
    # (aiter/jit/core.py:409-416). Pointing it at the workspace is what makes the
    # agent's edits to csrc/cpp_itfs/pa/*.cuh the code that actually gets
    # compiled; without it the image copy is rebuilt and the patch is a no-op.
    # core.py:412 falls back to the install root if csrc is absent, so an
    # un-seeded workspace still runs (against unmodified sources).
    if (WORKSPACE / "csrc").is_dir():
        for name in _AITER_SIBLINGS:
            link, src = WORKSPACE / name, _IMAGE_AITER_ROOT / name
            if not link.exists() and not link.is_symlink() and src.is_dir():
                link.symlink_to(src, target_is_directory=True)
        os.environ.setdefault("AITER_META_DIR", str(WORKSPACE))
    else:
        os.environ.setdefault("AITER_META_DIR", str(_IMAGE_AITER_ROOT))
    _pin_template_op_build_dir()
    os.chdir(WORKSPACE)


def _pa_source_fingerprint() -> str:
    """sha256 over the PA sources this run will compile, truncated to 12 hex."""
    import hashlib

    root = WORKSPACE if (WORKSPACE / "csrc").is_dir() else _IMAGE_AITER_ROOT
    pa_dir = root / "csrc" / "cpp_itfs" / "pa"
    h = hashlib.sha256()
    for path in sorted(pa_dir.rglob("*")):
        if path.is_file() and path.suffix in (".cuh", ".h", ".jinja", ".cpp", ".py"):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    # utils.py owns the compile flags and the template-op machinery.
    utils = root / "csrc" / "cpp_itfs" / "utils.py"
    if utils.is_file():
        h.update(utils.read_bytes())
    return h.hexdigest()[:12]


def _pin_template_op_build_dir() -> None:
    """Make the template-op build cache workspace-local AND content-addressed.

    Without this the task silently scores stale binaries. aiter compiles the
    ll4mi kernels through ``compile_template_op``
    (csrc/cpp_itfs/utils.py:300), whose cache folder is
    ``$AITER_ROOT_DIR/build/<md_name>_<md5 of the TEMPLATE ARGUMENTS>`` and whose
    only freshness check is ``not_built()`` -- "does lib.so exist"
    (utils.py:297). Two consequences, both fatal to this task:

      * the md5 covers head_size / dtype / block_size etc. but NOT the contents
        of pa_kernels.cuh, so an agent edits the kernel, the folder name is
        unchanged, lib.so is already there, and the OLD binary runs;
      * AITER_ROOT_DIR defaults to ``$HOME/.aiter`` (utils.py:68), which is
        shared across every workspace on the machine, so one run's binary leaks
        into another's.

    Folding a hash of the PA sources into AITER_ROOT_DIR fixes both: the path is
    under the workspace, and it changes exactly when the sources change -- so an
    edit forces a real rebuild while an unchanged rerun still hits the cache.
    """
    root = WORKSPACE / "build" / f"aiter-{_pa_source_fingerprint()}"
    (root / "build").mkdir(parents=True, exist_ok=True)
    os.environ["AITER_ROOT_DIR"] = str(root)


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


class _TimedRun:
    """Handle on the exact invocation a benchmark measured.

    Timing and correctness are otherwise separate invocations, so a kernel can
    tell them apart and do less work in the one that is scored. Passing this
    collector to the benchmark makes the scored invocation itself observable:
    ``outputs`` aliases the buffers the timed unit last wrote, and ``rerun``
    executes that same unit again.

    Under CUDA-graph timing the buffers are captured once and every replay
    writes to those same addresses, so ``outputs`` keeps tracking replays. Under
    event-timing fallback the measured outputs cannot be observed reliably, so a
    benchmark that requests this collector fails closed instead of validating a
    separate post-timing invocation.
    """

    def __init__(self):
        self._rerun = None
        self.outputs = None

    def _bind(self, rerun, outputs=None):
        self._rerun = rerun
        self.outputs = outputs

    @property
    def bound(self):
        return self._rerun is not None

    def rerun(self):
        if self._rerun is None:
            raise RuntimeError(
                "timed run was never bound; the benchmark did not reach a "
                "measurement path"
            )
        self.outputs = self._rerun()
        return self.outputs


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
    timed_run=None,
):
    import torch

    def _reject_unobservable_fallback(reason):
        if timed_run is not None:
            raise RuntimeError(
                f"{reason}; timed_run requires an observable CUDA-graph replay "
                "and cannot validate a separate post-timing invocation"
            )

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
        _reject_unobservable_fallback("CUDA is unavailable")
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_unavailable",
        })
        return sum(times) / len(times), metadata

    if not use_cuda_graph:
        _reject_unobservable_fallback("CUDA-graph timing is disabled")
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
                captured_outputs = None
                for _ in range(n_repeat):
                    captured_outputs = fn()
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
            _reject_unobservable_fallback("CUDA graph captured no device work")
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
        if timed_run is not None:

            def _replay_once():
                # Callers stage work on their own stream before re-running (they
                # perturb inputs and poison outputs). The capture stream must be
                # ordered after that, or the replay races the staged writes and
                # they land on top of the kernel's results.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                torch.cuda.synchronize()
                return captured_outputs

            timed_run._bind(_replay_once, captured_outputs)
        return graph_mean, metadata
    except Exception as exc:
        # Isolate the aborted capture before re-measuring so the fallback timing is
        # not polluted by the failed attempt (a mid-capture failure can leave the
        # first few launches abnormally slow).
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if timed_run is not None:
            raise RuntimeError(
                "CUDA-graph capture failed; timed_run cannot validate the "
                "separate CUDA-event fallback invocation"
            ) from exc
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


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    torch.set_grad_enabled(False)
    return torch


def _free() -> None:
    _torch().cuda.empty_cache()


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _fp8_dtype():
    """The KV cache element type sglang uses for kv_cache_dtype=fp8_e4m3 on gfx950.

    gfx950 is OCP fp8, i.e. torch.float8_e4m3fn. Older CDNA parts use the fnuz
    variant, so resolve rather than hardcode -- but fail loudly instead of
    silently falling back to a type the kernel would reject.
    """
    torch = _torch()
    for name in PA["kv_cache_torch_dtype_candidates"]:
        if hasattr(torch, name):
            return getattr(torch, name)
    raise RuntimeError(
        f"torch has none of {PA['kv_cache_torch_dtype_candidates']}; cannot build "
        f"an fp8_e4m3 KV cache."
    )


def _dispatched_kernels(inputs: dict) -> list[str]:
    """Name the HIP kernels this call lands on, most expensive first.

    Recorded and asserted: unlike the GEMM task, the session's kernel here IS the
    thing being optimised in place, so a change that silently routes to a
    different backend is a change of subject, not an optimisation.
    """
    torch = _torch()
    from torch.profiler import ProfilerActivity, profile

    _run(inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _run(inputs)
        torch.cuda.synchronize()
    events = [
        (e.self_device_time_total, e.key)
        for e in prof.key_averages()
        if getattr(e, "self_device_time_total", 0) > 0
    ]
    events.sort(reverse=True)
    return [k for _, k in events]


def _assert_paged_attention_dispatch(label: str, kernels: list[str]) -> None:
    blob = "\n".join(kernels)
    if "paged_attention_ll4mi_QKV_mfma16_kernel" not in blob:
        raise AssertionError(
            f"{label}: paged_attention_ll4mi_QKV_mfma16_kernel did not run. "
            f"Kernels seen: {kernels[:4]}. This task reproduces that specific "
            f"aiter ll4mi instantiation; a different backend is a different op."
        )
    if "Fp8KVCacheDataType)1" not in blob:
        raise AssertionError(
            f"{label}: the dispatched instantiation is not the fp8 KV one. The "
            f"session runs kv_cache_dtype=fp8_e4m3 with in-kernel dequant; a bf16 "
            f"cache would be a different (and cheaper) kernel. Kernels: {kernels[:2]}"
        )


# --------------------------------------------------------------------------- #
# Inputs
#
# Argument order and every constant below are taken from the live call site,
# srt/layers/attention/aiter_backend.py:2638-2658:
#     paged_attention_ragged(
#         o.view(-1, tp_q_head_num, v_head_dim),
#         workspace_buffer,
#         q.view(-1, tp_q_head_num, qk_head_dim),
#         k_cache.view(-1, 1, tp_k_head_num, qk_head_dim),
#         v_cache.view(-1, 1, tp_v_head_num, v_head_dim),
#         scale, kv_indptr, kv_indices, kv_last_page_len,
#         1,                       # block_size (page_size=1)
#         max_num_partitions, None, aiter_kv_str, "NHD", logits_soft_cap,
#         k_scale, v_scale, None, _AITER_PARTITION_SIZE_ROCM)
# The workspace sizing is aiter_backend.py:277-282.
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    torch = _torch()

    p = dict(case["params"])
    batch, seq_len = int(p["batch"]), int(p["seq_len"])
    qh, kvh, hd = int(p["q_heads"]), int(p["kv_heads"]), int(p["head_dim"])
    part = int(p["partition_size"])
    fp8 = _fp8_dtype()

    torch.manual_seed(int(case.get("seed", 17)))
    dev = "cuda"
    q = torch.randn((batch, qh, hd), device=dev, dtype=torch.bfloat16) * 0.1
    # page_size = 1, so one "page" per token; the pool is exactly batch*seq_len.
    pages = batch * seq_len
    k_cache = (torch.randn((pages, 1, kvh, hd), device=dev, dtype=torch.bfloat16) * 0.1).to(fp8)
    v_cache = (torch.randn((pages, 1, kvh, hd), device=dev, dtype=torch.bfloat16) * 0.1).to(fp8)
    out = torch.empty((batch, qh, hd), device=dev, dtype=torch.bfloat16)

    max_num_partitions = (seq_len + part - 1) // part
    fp32_bytes = torch.finfo(torch.float32).bits // 8
    workspace = torch.empty(
        (batch * qh * max_num_partitions * hd) * fp32_bytes
        + 2 * (batch * qh * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=dev,
    )
    kv_indptr = torch.arange(0, (batch + 1) * seq_len, seq_len, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, pages, device=dev, dtype=torch.int32)
    kv_last_page_len = torch.ones((batch,), device=dev, dtype=torch.int32)
    k_scale = torch.tensor([float(p["k_scale"])], device=dev, dtype=torch.float32)
    v_scale = torch.tensor([float(p["v_scale"])], device=dev, dtype=torch.float32)

    return {
        "cfg": case, "params": p, "batch": batch, "seq_len": seq_len,
        "q_heads": qh, "kv_heads": kvh, "head_dim": hd,
        "q": q, "k_cache": k_cache, "v_cache": v_cache, "out": out,
        "workspace": workspace, "kv_indptr": kv_indptr, "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "k_scale": k_scale, "v_scale": v_scale,
        "scale": float(hd) ** -0.5,
        "max_num_partitions": max_num_partitions, "partition_size": part,
    }


def _run(inputs: dict):
    from aiter.ops.attention import paged_attention_ragged

    paged_attention_ragged(
        inputs["out"],
        inputs["workspace"],
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["scale"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        1,                                   # block_size / page_size
        inputs["max_num_partitions"],
        None,                                # alibi_slopes
        PA["aiter_kv_cache_dtype"],
        PA["kv_cache_layout"],
        float(PA["logits_soft_cap"]),
        inputs["k_scale"],
        inputs["v_scale"],
        None,                                # fp8_out_scale
        inputs["partition_size"],
    )
    return inputs["out"]


# --------------------------------------------------------------------------- #
# Reference
#
# Plain fp32 GQA attention over the dequantized cache. Fully vectorized: two
# einsums and a softmax, no loop over batch, head or partition -- so it is also
# an independent implementation of the partitioned reduction the ll4mi pair does
# in two kernels.
#
# Sizes stay small because only the scores are materialized: at the session's
# bs=64 / 8 heads / seq 8704 that is [64, 8, 8704] fp32 = 18 MiB. The
# dequantized cache is the larger term ([64, 8704, 256] fp32 = 570 MiB), so it
# is built per call and dropped rather than cached.
# --------------------------------------------------------------------------- #
def _reference(inputs: dict):
    torch = _torch()
    batch, seq_len = inputs["batch"], inputs["seq_len"]
    qh, kvh, hd = inputs["q_heads"], inputs["kv_heads"], inputs["head_dim"]

    # k_scale / v_scale are 1.0 in the session, but apply them anyway so the
    # reference stays correct if a case ever carries a real scale.
    k = inputs["k_cache"].view(batch, seq_len, kvh, hd).float() * inputs["k_scale"]
    v = inputs["v_cache"].view(batch, seq_len, kvh, hd).float() * inputs["v_scale"]
    q = inputs["q"].float().view(batch, kvh, qh // kvh, hd)   # GQA grouping

    scores = torch.einsum("bkgd,bskd->bkgs", q, k) * inputs["scale"]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bkgs,bskd->bkgd", probs, v)
    del k, v, scores, probs
    return out.reshape(batch, qh, hd)


def _deviation(got, expected):
    torch = _torch()
    g, e = got.float().flatten(), expected.float().flatten()
    cos = torch.nn.functional.cosine_similarity(g, e, dim=0).item()
    err = ((g - e).norm() / e.norm().clamp_min(1e-8)).item()
    return cos, err


def _assert_within_tolerance(case: dict, cos: float, err: float, label: str) -> None:
    tol = case["params"]
    assert cos > tol.get("min_cosine", 0.999), (
        case["id"], f"{label} cosine {cos:.6f} vs fp32 reference too low"
    )
    assert err < tol.get("max_rel_norm_err", 0.05), (
        case["id"], f"{label} relative norm error {err:.5f} too high"
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _prepare(CASES[0])
    out = _run(inputs)
    _torch().cuda.synchronize()
    kernels = _dispatched_kernels(inputs)
    _assert_paged_attention_dispatch("compile", kernels)
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} {out.dtype}")
    for k in kernels[:2]:
        print(f"  kernel: {k[:120]}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        # Identical case list and identical shapes to run_performance.
        inputs = _prepare(case)
        kernels = _dispatched_kernels(inputs)
        _assert_paged_attention_dispatch(case["id"], kernels)
        got = _run(inputs)
        torch.cuda.synchronize()
        expected = _reference(inputs)
        assert torch.isfinite(got).all(), (case["id"], "non-finite output")
        assert tuple(got.shape) == tuple(expected.shape), (
            case["id"], tuple(got.shape), tuple(expected.shape)
        )
        cos, err = _deviation(got, expected)
        _assert_within_tolerance(case, cos, err, "single call")
        print(f"correctness PASS {case['id']:34s} bs={inputs['batch']:<4d} "
              f"seq={inputs['seq_len']:<6d} cos={cos:.6f} rel_err={err:.5f}")
        del inputs, expected, got
        _free()


def _perturb(inputs: dict) -> None:
    """Redraw the query so the scored invocation sees unseen values.

    The KV cache stays put: it is the paged pool, which in serving is written by
    reshape_and_cache_flash and only appended to, and rebuilding the fp8 copy per
    check would dominate the validation for no added coverage.
    """
    torch = _torch()
    torch.manual_seed(83)
    inputs["q"].normal_(0.0, 0.1)


def _assert_timed_outputs(case: dict, inputs: dict, timed) -> None:
    torch = _torch()
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    got = timed.rerun()
    expected = _reference(inputs)
    assert torch.isfinite(got).all(), (case["id"], "timed run produced non-finite output")
    cos, err = _deviation(got, expected)
    _assert_within_tolerance(case, cos, err, "timed run")
    del expected


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _prepare(case)
        _run(inputs)
        _torch().cuda.synchronize()
        kernels = _dispatched_kernels(inputs)
        _assert_paged_attention_dispatch(case["id"], kernels)
        bench = case.get("benchmark", {})
        timed = _TimedRun()
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=bench.get("warmup", 10),
            repetition=bench.get("repetition", 30),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 100),
            timed_run=timed,
        )
        _assert_timed_outputs(case, inputs, timed)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "phase": case.get("phase"),
            "calls_per_forward": case.get("calls_per_forward"),
            "session_e2e_pct": case.get("session_e2e_pct"),
            "dispatched_kernels": kernels[:2],
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            **{k: v for k, v in meta.items() if k.startswith("benchmark_")},
            "metadata": metadata,
        })
        print(f"{case['id']:34s} {exec_ms:.6f} ms  {meta.get('benchmark_method')}  "
              f"{meta.get('benchmark_fallback_reason', '')}")
        del inputs
        _free()
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
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
