#!/usr/bin/env python3
"""Image-kernel harness for the Qwen3.8-2.4T-A95B MXFP4 routed-expert MoE 2-stage GEMM.

Reproduces the aiter FlyDSL MoE path behind Hyperloom session 100137
(Qwen3.8-2.4T-A95B-Quark-MXFP4, 20260814T175123Z) on MI355X/gfx950, sglang
0.5.17.dev20260812+gdc5f6c4883 / ROCm 7.2.0 / aiter d9e5ef7c.

Hot kernels reproduced (E2E numbers measured from the session's own 8-rank
torch trace, decode share of wall clock = 56.7%):

  mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32       11.62% E2E
  mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_
      fp4opt_v1_persist_cu256                                        6.25% E2E

Config, all read out of the session (see session_cases.json ``moe_config``):
  a4w4 -- fp4 activation x fp4 weight, MXFP4 group_size=32 -> QuantType.per_1x32,
  e8m0 scales, g1u1, ActivationType.Silu, per-rank (TP=8 / EP=8)
  model_dim=8192 / inter_dim=2048 / local experts=64 (of 512) / topk=10.

Two M buckets are covered because the session runs both and they dispatch
*different* FlyDSL kernels (aiter/fused_moe.py:2249-2255):

  decode  M=64      -> flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2
                       flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2
  prefill M=16384   -> flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0
                       flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic
  prefill M=8192    -> same t64 pair (26 of the session's 429 prefill chunks)

Three contract details that are easy to get wrong, each verified against the
in-image sources rather than assumed:

  * ``use_mxfp4_flydsl`` (aiter/fused_moe.py:2198) requires ``is_shuffled`` and
    ``not doweight_stage1``. Hand a MoE call unshuffled weights and it silently
    lands on the CK 2-stage path
    (module_moe_ck2stages_fp4x2_fp4x2_preshuffle_off_b16_silu_per_1x32_...)
    instead -- a completely different kernel from the one the session ran.
    ``_assert_flydsl_dispatch`` fails closed on that.
  * The weight/scale preparation must match what sglang's Quark MXFP4 MoE
    scheme does at load time
    (sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:577-597):
    ``e8m0_shuffle`` on the scales viewed 2D, then ``shuffle_weight(w, (16, 16))``
    on the packed nibbles, then ``is_shuffled = True``.
  * Unlike the Kimi-K3 sibling task, this session runs entirely on the
    **heuristic fallback** branch -- ``no tuned FlyDSL config`` appears 120 times
    in the session's server.log, covering all 15 M buckets. The fallback is the
    thing being optimised here, so the harness asserts the dispatch stays FlyDSL
    but deliberately does *not* require a tuned-CSV hit.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]
MOE = SPEC["moe_config"]

_IMAGE_AITER_ROOT = Path(os.environ.get("QWEN38_AITER_ROOT", "/sgl-workspace/aiter"))
# aiter resolves csrc/include/aiter_enum.h relative to the directory *containing*
# the package (aiter/utility/aiter_types.py:10 hardcodes parents[2]), so a
# workspace that only seeds `aiter/` cannot import at all without these siblings.
# They are not part of this task's edit surface, so they are linked, not copied.
_AITER_SIBLINGS = ("csrc", "3rdparty", "hsa")


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    if (WORKSPACE / "aiter").is_dir():
        for name in _AITER_SIBLINGS:
            link, src = WORKSPACE / name, _IMAGE_AITER_ROOT / name
            if not link.exists() and not link.is_symlink() and src.is_dir():
                link.symlink_to(src, target_is_directory=True)
        sys.path.insert(0, str(WORKSPACE))
    os.environ.setdefault("AITER_META_DIR", str(_IMAGE_AITER_ROOT))
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
# Dispatch capture
#
# aiter logs the FlyDSL pair it picked per M bucket at INFO:
#
#   [fused_moe] no tuned FlyDSL config for (...), using heuristic FlyDSL
#   fallback (kn1='flydsl_moe1_...', kn2='flydsl_moe2_...')
#
# Two failures are silent without this check and both are fatal to the task:
#   * weights not shuffled -> CK 2-stage path, i.e. the wrong kernel entirely;
#   * correctness and performance running at token counts in different M
#     buckets, so the pair being scored is never the pair being checked.
# --------------------------------------------------------------------------- #
_KN1_RE = re.compile(r"kn1='([^']*)'")
_KN2_RE = re.compile(r"kn2='([^']*)'")
_TUNED_KN1_RE = re.compile(r"kernelName1='([^']*)'")
_TUNED_KN2_RE = re.compile(r"kernelName2='([^']*)'")


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - never break a run on logging
            pass


@contextlib.contextmanager
def _capture_aiter_log():
    from aiter import logger as aiter_logger

    cap = _LogCapture()
    previous = aiter_logger.level
    aiter_logger.addHandler(cap)
    aiter_logger.setLevel(logging.INFO)
    try:
        yield cap
    finally:
        aiter_logger.removeHandler(cap)
        aiter_logger.setLevel(previous)


def _dispatch_names(inputs: dict) -> tuple[str, str, str]:
    """Run the op once and report which FlyDSL kernel pair aiter dispatched.

    ``get_2stage_cfgs`` is lru_cached, so the log line only appears on the first
    lookup of a key; clear it so the capture is not silently empty.
    """
    from aiter.fused_moe import get_2stage_cfgs

    get_2stage_cfgs.cache_clear()
    with _capture_aiter_log() as cap:
        _run(inputs)
        _torch().cuda.synchronize()
    blob = "\n".join(cap.messages)
    kn1 = _KN1_RE.search(blob) or _TUNED_KN1_RE.search(blob)
    kn2 = _KN2_RE.search(blob) or _TUNED_KN2_RE.search(blob)
    return (kn1.group(1) if kn1 else "", kn2.group(1) if kn2 else "", blob)


def _assert_flydsl_dispatch(label: str, kn1: str, kn2: str, blob: str) -> None:
    if not (kn1 and kn2):
        raise AssertionError(
            f"{label}: aiter logged no FlyDSL kernel pair (kn1={kn1!r}, kn2={kn2!r}). "
            f"The session's MoE runs on the FlyDSL a4w4 path; a missing pair means "
            f"the call fell through to another backend. Captured log:\n{blob[:2000]}"
        )
    if not (kn1.startswith("flydsl_moe1_afp4_wfp4") and kn2.startswith("flydsl_moe2_afp4_wfp4")):
        raise AssertionError(
            f"{label}: dispatched pair is not the FlyDSL a4w4 pair this task "
            f"reproduces (kn1={kn1!r}, kn2={kn2!r}). The usual cause is weights "
            f"reaching fused_moe without .is_shuffled=True, which sends the call "
            f"to the CK 2-stage path (aiter/fused_moe.py:2198 use_mxfp4_flydsl "
            f"requires is_shuffled and not doweight_stage1)."
        )
    if "ck2stages" in blob:
        raise AssertionError(f"{label}: a CK 2-stage module was loaded; not the session path.")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _aiter():
    import aiter

    return aiter


def _free() -> None:
    _torch().cuda.empty_cache()


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _assert_sglang_shuffle_contract(nlane: tuple) -> None:
    if tuple(nlane) != (16, 16):
        raise AssertionError(
            f"shuffle_weight layout {tuple(nlane)}, but sglang hardcodes (16, 16) at "
            f"quark_w4a4_mxfp4_moe.py:590,593. A kernel that needs a different "
            f"layout cannot be reached from sglang."
        )


# --------------------------------------------------------------------------- #
# Inputs
#
# Shapes come straight out of the session's CUDA-graph capture trace
# (record_shapes=True), aiter::fused_moe_ cpu_op at bs=64:
#     hidden [M, 8192] bf16
#     w1     [64, 4096, 4096] fp4x2   (= [E_local, 2*inter, hidden] MXFP4)
#     w2     [64, 8192, 1024] fp4x2   (= [E_local, hidden, inter]   MXFP4)
#     w1_scale [64, 4096, 256] e8m0 / w2_scale [64, 8192, 64] e8m0
#     topk_weight [M, 10] f32 / topk_ids [M, 10] i32 / expert_mask [512] i32
# Only M changes between cases; every weight shape is fixed by the model config.
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    torch = _torch()
    aiter = _aiter()
    from aiter import dtypes
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    p = dict(case["params"])
    if (p["quant_type"], p["a_dtype"], p["w_dtype"], p["use_g1u1"]) != (
        "per_1x32", "fp4", "fp4", True
    ):
        raise RuntimeError(f"case {case['id']} is not the Qwen3.8 a4w4 g1u1 contract: {p}")

    token = int(p["token"])
    experts = int(p["local_experts"])
    total_experts = int(p["total_experts"])
    model_dim = int(p["model_dim"])
    inter_dim = int(p["inter_dim"])
    topk = int(p["topk"])

    torch.manual_seed(int(case.get("seed", 7)))
    hidden = torch.randn((token, model_dim), device="cuda", dtype=dtypes.bf16) * 0.1
    w1_bf16 = torch.randn(
        (experts, inter_dim * 2, model_dim), device="cuda", dtype=dtypes.bf16
    ) * 0.03
    w2_bf16 = torch.randn(
        (experts, model_dim, inter_dim), device="cuda", dtype=dtypes.bf16
    ) * 0.03

    # aiter's own MXFP4 quantizer; rows are laid out GGUU (all gate rows then all
    # up rows), matching torch_moe_stage1's out.split([inter, inter], -1).
    torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_s = torch_quant(w1_bf16, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = torch_quant(w2_bf16, quant_dtype=dtypes.fp4x2)
    del w1_bf16, w2_bf16
    w1_s = w1_s.view(experts, inter_dim * 2, -1)
    w2_s = w2_s.view(experts, model_dim, -1)

    # Exactly the sglang load-time preparation, in the same order.
    nlane = tuple(MOE.get("shuffle_layout", (16, 16)))
    _assert_sglang_shuffle_contract(nlane)
    w1_rt = shuffle_weight(w1_q.view(torch.uint8).contiguous(), nlane).view(dtypes.fp4x2)
    w2_rt = shuffle_weight(w2_q.view(torch.uint8).contiguous(), nlane).view(dtypes.fp4x2)
    w1_rt.is_shuffled = True
    w2_rt.is_shuffled = True
    w1_s_rt = e8m0_shuffle(w1_s.reshape(experts * inter_dim * 2, -1)).view(
        experts, inter_dim * 2, -1
    )
    w2_s_rt = e8m0_shuffle(w2_s.reshape(experts * model_dim, -1)).view(
        experts, model_dim, -1
    )

    score = torch.randn((token, total_experts), device="cuda", dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
    # EP=8: this rank owns experts [0, 64). Routing is global (512 experts), so
    # ~7/8 of the (token, expert) pairs are dropped by the mask -- same as serving.
    expert_mask = torch.zeros(total_experts, dtype=torch.int32, device="cuda")
    expert_mask[:experts] = 1

    return {
        "cfg": case, "params": p, "token": token, "topk": topk,
        "experts": experts, "model_dim": model_dim, "inter_dim": inter_dim,
        "hidden": hidden,
        "w1": w1_rt, "w2": w2_rt, "w1_scale": w1_s_rt, "w2_scale": w2_s_rt,
        # Unshuffled copies drive the torch reference.
        "w1_ref": w1_q, "w2_ref": w2_q, "w1_scale_ref": w1_s, "w2_scale_ref": w2_s,
        "topk_weights": topk_weights, "topk_ids": topk_ids, "expert_mask": expert_mask,
    }


def _run(inputs: dict):
    torch = _torch()
    aiter = _aiter()
    from aiter.fused_moe import fused_moe

    return fused_moe(
        inputs["hidden"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        expert_mask=inputs["expert_mask"],
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        activation=aiter.ActivationType.Silu,
        dtype=torch.bfloat16,
    )


# --------------------------------------------------------------------------- #
# Reference
#
# aiter's own torch_moe_stage1/stage2 cannot be used here: they dequantize the
# whole expert tensor to fp32 and materialize [token, topk, 2*inter], which at
# the session's prefill M=16384 is 2.7 TB. This reference keeps memory bounded by
# dequantizing one expert at a time, and stays vectorized -- the only Python loop
# is over the 64 local experts, and each iteration is two dense GEMMs.
#
# It is an independent implementation, not a wrapper: it unpacks the MXFP4
# nibbles, applies the per-1x32 e8m0 group scales, and fake-quantizes the
# activation and the stage-1 output the same way the kernel does (the a4w4 path
# quantizes both), so the only remaining difference is accumulation order.
# --------------------------------------------------------------------------- #
def _dequant_mxfp4(packed, scale_e8m0):
    """[R, K/2] fp4x2 + [R, K/32] e8m0  ->  [R, K] bf16."""
    torch = _torch()
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    values = mxfp4_to_f32(packed)
    scale = e8m0_to_f32(scale_e8m0)
    rows = values.shape[0]
    out = (values.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, -1)
    return out.to(torch.bfloat16)


def _fake_quant_mxfp4(x):
    """Round x through MXFP4 exactly as the kernel's input quantizer does."""
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    packed, scale = dynamic_mxfp4_quant(x)
    return _dequant_mxfp4(packed, scale)


def _reference(inputs: dict):
    torch = _torch()
    with torch.no_grad():
        token = inputs["token"]
        model_dim, inter_dim = inputs["model_dim"], inputs["inter_dim"]
        topk = inputs["topk"]

        x = _fake_quant_mxfp4(inputs["hidden"])                     # [M, H] bf16
        flat_expert = inputs["topk_ids"].reshape(-1)
        flat_token = torch.arange(token, device="cuda").repeat_interleave(topk)
        flat_weight = inputs["topk_weights"].reshape(-1).float()

        out = torch.zeros((token, model_dim), device="cuda", dtype=torch.float32)
        for e in range(inputs["experts"]):
            rows = (flat_expert == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            tok = flat_token[rows]
            w1 = _dequant_mxfp4(inputs["w1_ref"][e], inputs["w1_scale_ref"][e])
            gate_up = x[tok] @ w1.transpose(0, 1)                   # [n, 2*inter]
            del w1
            gate, up = gate_up.split([inter_dim, inter_dim], dim=-1)
            act = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
            del gate_up, gate, up
            act = _fake_quant_mxfp4(act)                            # stage-2 input quant
            w2 = _dequant_mxfp4(inputs["w2_ref"][e], inputs["w2_scale_ref"][e])
            y = (act @ w2.transpose(0, 1)).float()                  # [n, H]
            del w2, act
            out.index_add_(0, tok, y * flat_weight[rows].unsqueeze(-1))
            del y
        return out.to(torch.bfloat16)


def _deviation(got, expected):
    torch = _torch()
    g, e = got.float().flatten(), expected.float().flatten()
    cos = torch.nn.functional.cosine_similarity(g, e, dim=0).item()
    err = ((g - e).norm() / e.norm().clamp_min(1e-8)).item()
    return cos, err


def _assert_within_tolerance(case: dict, cos: float, err: float, label: str) -> None:
    tol = case["params"]
    assert cos > tol.get("min_cosine", 0.97), (
        case["id"], f"{label} cosine {cos:.6f} vs torch reference too low"
    )
    assert err < tol.get("max_rel_norm_err", 0.25), (
        case["id"], f"{label} relative norm error {err:.4f} too high"
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _prepare(CASES[0])
    out = _run(inputs)
    _torch().cuda.synchronize()
    kn1, kn2, blob = _dispatch_names(inputs)
    _assert_flydsl_dispatch("compile", kn1, kn2, blob)
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} {out.dtype}")
    print(f"  FlyDSL dispatch OK  stage1={kn1}\n                      stage2={kn2}")


# stage2 reduces with atomics, so the result is not bit-reproducible. Gate on the
# worst of N rather than getting lucky once.
_CORRECTNESS_REPEATS = 3


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        # Correctness runs at the case's own token -- identical to the shape the
        # performance mode scores. Shrinking it would move the call into another
        # M bucket and check a different FlyDSL kernel than the one measured.
        inputs = _prepare(case)
        kn1, kn2, blob = _dispatch_names(inputs)
        _assert_flydsl_dispatch(case["id"], kn1, kn2, blob)
        _assert_expected_pair(case, kn1, kn2)

        expected = _reference(inputs)
        worst_cos, worst_err = 1.0, 0.0
        for _ in range(_CORRECTNESS_REPEATS):
            got = _run(inputs)
            torch.cuda.synchronize()
            assert torch.isfinite(got).all(), (case["id"], "non-finite output")
            assert tuple(got.shape) == tuple(expected.shape), (
                case["id"], tuple(got.shape), tuple(expected.shape)
            )
            cos, err = _deviation(got, expected)
            worst_cos, worst_err = min(worst_cos, cos), max(worst_err, err)

        _assert_within_tolerance(case, worst_cos, worst_err, f"worst-of-{_CORRECTNESS_REPEATS}")
        print(f"correctness PASS {case['id']:38s} M={inputs['token']:<6d} "
              f"cos={worst_cos:.6f} rel_err={worst_err:.4f}")
        del inputs, expected, got
        _free()


def _assert_expected_pair(case: dict, kn1: str, kn2: str) -> None:
    """Pin the kernel pair the session actually ran for this M bucket."""
    want = case.get("expected_dispatch")
    if not want:
        return
    if (kn1, kn2) != (want["stage1"], want["stage2"]):
        raise AssertionError(
            f"{case['id']}: dispatched ({kn1}, {kn2}) but the session ran "
            f"({want['stage1']}, {want['stage2']}) at this M. Either the tile "
            f"heuristic moved or the case's token no longer lands in the session's "
            f"M bucket; both make the scored kernel unrepresentative."
        )


def _perturb(inputs: dict) -> None:
    """Refresh the activation in place with values no earlier launch has seen.

    Only ``hidden`` is redrawn. Routing tensors are kernel inputs, not something
    the kernel derives, so leaving them fixed keeps kernel and reference on the
    same workload; the quantized weights have shuffled runtime copies that would
    have to be rebuilt in lockstep, which buys nothing here.
    """
    torch = _torch()
    torch.manual_seed(43)
    inputs["hidden"].normal_(0.0, 0.1)


def _assert_timed_outputs(case: dict, inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed.

    run_correctness checks a separate call, which a kernel could tell apart from
    the scored one. This re-runs the timed unit against a freshly perturbed
    activation and checks the buffer it wrote.
    """
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
        # Same case list, same shapes as run_correctness -- no perf-only case and
        # no correctness-only case, so every scored shape is a checked shape.
        inputs = _prepare(case)
        _run(inputs)
        _torch().cuda.synchronize()
        kn1, kn2, blob = _dispatch_names(inputs)
        _assert_flydsl_dispatch(case["id"], kn1, kn2, blob)
        _assert_expected_pair(case, kn1, kn2)

        bench = case.get("benchmark", {})
        timed = _TimedRun()
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 20),
            target_ms=bench.get("target_ms", 1.0),
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
            "dispatched_stage1_kernel": kn1,
            "dispatched_stage2_kernel": kn2,
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            **{k: v for k, v in meta.items() if k.startswith("benchmark_")},
            "metadata": metadata,
        })
        print(case["id"], f"{exec_ms:.6f} ms", meta.get("benchmark_method"),
              meta.get("benchmark_fallback_reason", ""))
        print(f"  timed stage1={kn1}\n        stage2={kn2}")
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
