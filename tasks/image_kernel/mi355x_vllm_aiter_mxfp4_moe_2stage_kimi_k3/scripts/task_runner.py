#!/usr/bin/env python3
"""Image-kernel harness for the Kimi-K3 routed-expert MoE 2-stage GEMM (aiter mxfp4).

Reproduces the MoE path behind Hyperloom session 20260728T091437Z hot kernels
k001/k002 (decode-graph moe_gemm1_0 / moe_gemm2_0) and k003/k006 (eager FlyDSL
moe_flydsl_stage1 / stage2) on MI355X/gfx950.

Config, all taken from the session (see session_cases.json ``moe_config``):
  bf16 activation x mxfp4 weight (group_size=32 -> QuantType.per_1x32), g1u1,
  SiTUv2 gated activation with beta=4.0 / linear_beta=25.0, per-rank (TP=8)
  model_dim=3584 / inter_dim=384 / experts=896 / topk=16.

Three contract details that are easy to get wrong, each verified against the
in-image sources rather than assumed:

  * The activation enum is ``ActivationType.Situv2`` (aiter/fused_moe.py:619,
    1202, 2198). There is no ``ActivationType.Situ`` in this build.
  * K3's SiTU a16w4 path runs the **SEPARATED** (GGUU) gate/up layout, so the
    weights and scales must be shuffled with ``gate_up=False``.
    ``gate_up=True`` produces the GUGU/INTERLEAVE layout used by the gpt-oss
    ``use_mxfp4_w4a16`` path instead, and silently yields a correct-magnitude but
    numerically unrelated result. See vllm .../experts/rocm_aiter_moe.py:369-386.
  * ``torch_moe_stage1`` defaults to ``situ_beta=2.0 / situ_linear_beta=1.5``
    while ``fused_moe`` defaults to ``1.0 / 1.0`` (aiter/fused_moe.py:676-677).
    Neither is K3's value, so both sides are driven from the session config here.
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
MOE_CONFIG = SPEC["moe_config"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Keep aiter's JIT output inside the run workspace so one run can never load a
    # module another run built.
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
    # The agent edits the workspace-seeded copy of aiter, so it must shadow the
    # in-image install; otherwise `import aiter` resolves to
    # /usr/local/lib/python3.12/dist-packages/aiter and the edits are ignored.
    if (WORKSPACE / "aiter").is_dir():
        _link_aiter_meta()
        sys.path.insert(0, str(WORKSPACE))
    os.environ.setdefault("AITER_META_DIR", str(_IMAGE_AITER_META))
    os.chdir(WORKSPACE)


_IMAGE_AITER_META = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta")


def _link_aiter_meta() -> None:
    """Give the workspace copy of aiter its sibling C++ tree.

    aiter locates aiter_enum.h relative to the directory that *contains* the
    package -- aiter/utility/aiter_types.py:_find_aiter_enum_h hardcodes
    ``parents[2]`` and ignores AITER_META_DIR. The workspace only seeds `aiter`,
    so without this the copied package cannot import at all. aiter_meta holds the
    C++/HIP sources, which are not part of this task's edit surface, so it is
    linked rather than duplicated per run.
    """
    link = WORKSPACE / "aiter_meta"
    if link.exists() or link.is_symlink():
        return
    if _IMAGE_AITER_META.is_dir():
        link.symlink_to(_IMAGE_AITER_META, target_is_directory=True)


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


# --------------------------------------------------------------------------- #
# Dispatch capture
#
# aiter picks the FlyDSL kernel pair per M bucket by looking the padded token
# count up in the tuned CSV, and logs the choice at INFO:
#
#   [fused_moe] using 2stage (kernelName1='flydsl_moe1_...', kernelName2='...') for (...)
#
# Two things can go wrong silently and both are caught here:
#   * the tuned CSV (/tmp/aiter_configs/tuned_fmoe.csv, a runtime merge artifact)
#     is missing, so aiter falls back to the heuristic branch
#     (aiter/fused_moe.py:2272 "no tuned FlyDSL config ... heuristic FlyDSL
#     fallback"). The session ran entirely on the tuned path, so optimising the
#     fallback branch would be wasted work.
#   * correctness and performance run at token counts that land in different M
#     buckets, so the kernel pair being scored is never the kernel pair being
#     checked. Each of the 14 buckets dispatches a *different* tuned pair.
# --------------------------------------------------------------------------- #
_KN1_RE = re.compile(r"kernelName1='([^']*)'")
_KN2_RE = re.compile(r"kernelName2='([^']*)'")


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - never break the run on logging
            pass


@contextlib.contextmanager
def _capture_aiter_log():
    """Capture aiter's logger output for the duration of the block."""
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
    kn1 = _KN1_RE.search(blob)
    kn2 = _KN2_RE.search(blob)
    return (kn1.group(1) if kn1 else "", kn2.group(1) if kn2 else "", blob)


def _assert_tuned_dispatch(label: str, kn1: str, kn2: str, blob: str) -> None:
    if "heuristic FlyDSL fallback" in blob:
        raise AssertionError(
            f"{label}: aiter fell back to the heuristic FlyDSL branch. The session "
            f"ran entirely on the tuned path, so this run would optimise code that "
            f"never executes in production. Check that the merged tuned config "
            f"(normally /tmp/aiter_configs/tuned_fmoe.csv) is present."
        )
    if "2stage default" in blob or not (kn1 and kn2):
        raise AssertionError(
            f"{label}: no tuned 2-stage config was found (kernelName1={kn1!r}, "
            f"kernelName2={kn2!r}). Expected a 'using 2stage (kernelName1=...)' "
            f"line from aiter. Captured log:\n{blob[:2000]}"
        )
    if not (kn1.startswith("flydsl_moe1_") and kn2.startswith("flydsl_moe2_")):
        raise AssertionError(
            f"{label}: dispatched pair is not the FlyDSL a16w4 pair this task "
            f"reproduces (kernelName1={kn1!r}, kernelName2={kn2!r})."
        )


def _free() -> None:
    """Release the ~37 GiB of dequantized fp32 weights a case leaves behind.

    Callers must ``del`` their own references first: deleting a parameter inside
    this function would only drop the local name.
    """
    _torch().cuda.empty_cache()


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _aiter():
    import aiter

    return aiter


def _activation(aiter):
    """Resolve the session's activation enum, failing loudly on an unknown build."""
    for name in MOE_CONFIG["activation_enum_candidates"]:
        if hasattr(aiter.ActivationType, name):
            return getattr(aiter.ActivationType, name)
    available = [x for x in dir(aiter.ActivationType) if not x.startswith("_")]
    raise RuntimeError(
        f"aiter.ActivationType has none of "
        f"{MOE_CONFIG['activation_enum_candidates']} (session activation="
        f"{MOE_CONFIG['activation']!r}). Run in the session build "
        f"(vLLM 0.1.dev19253+g5f76ae224.d20260727 + matching aiter). "
        f"Available: {available}"
    )


# --------------------------------------------------------------------------- #
# vLLM boundary contract
#
# The patch this task produces is applied to `aiter`; vLLM is never modified.
# vLLM reaches the same shuffle code, but through a *different entry point* and
# at a different time (weight load, not per call):
#
#   vllm .../quantization/mxfp4.py:788-799  _setup_kernel_k3_situ
#       w13       = rocm_aiter_ops.shuffle_weight_a16w4(w13, 16, guinterleave)
#       w2        = rocm_aiter_ops.shuffle_weight_a16w4(w2,  16, False)
#       w13_scale = rocm_aiter_ops.shuffle_scale_a16w4(..., num_experts, guinterleave)
#       w2_scale  = e8m0_shuffle(...)          <-- NOT shuffle_scale_a16w4
#
# rocm_aiter_ops.* are pure forwarders into aiter.ops.shuffle
# (vllm/_aiter_ops.py:2727,2748), so a patched aiter is used by vLLM's loader
# too and layouts stay consistent -- with two exceptions that the harness cannot
# notice on its own, because it drives both sides itself:
#
#   * nLane is hardcoded to 16 on the vLLM side.
#   * w2_scale goes through e8m0_shuffle -> shuffle_scale(is_guinterleave=False),
#     a different branch of aiter/ops/shuffle.py:338 than the
#     shuffle_scale_a16w4 -> shuffle_scale(is_guinterleave=True) path used here.
#     They are byte-identical for K3's (3211264, 16) w2_scale today; a patch that
#     touches one branch and not the other would pass here and corrupt stage2 in
#     vLLM.
# --------------------------------------------------------------------------- #
def _assert_vllm_shuffle_contract(torch, n_lane: int, w2_scale, w2_scale_runtime) -> None:
    if n_lane != 16:
        raise AssertionError(
            f"n_lane={n_lane}, but vLLM hardcodes 16 at "
            f"vllm/model_executor/layers/quantization/mxfp4.py:789,792. A kernel "
            f"that needs a different nLane cannot be reached from vLLM."
        )
    from aiter.utility.fp4_utils import e8m0_shuffle

    vllm_layout = e8m0_shuffle(w2_scale)
    if vllm_layout.shape != w2_scale_runtime.shape or not torch.equal(
        vllm_layout.reshape(-1), w2_scale_runtime.reshape(-1)
    ):
        raise AssertionError(
            "w2_scale layout divergence: shuffle_scale_a16w4(w2_scale, experts, "
            "False) no longer equals e8m0_shuffle(w2_scale). vLLM's loader uses "
            "e8m0_shuffle (mxfp4.py:799), so this patch would pass correctness "
            "here and produce a wrong w2_scale layout in vLLM. Keep the "
            "is_guinterleave=False and is_guinterleave=True branches of "
            "aiter/ops/shuffle.py:338 in agreement for this shape."
        )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    aiter = _aiter()
    from aiter import dtypes
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    p = dict(case["params"])
    if (p["quant_type"], p["a_dtype"], p["w_dtype"], p["use_g1u1"]) != (
        "per_1x32", "bf16", "fp4", True
    ):
        raise RuntimeError(f"case {case['id']} is not the K3 a16w4 g1u1 contract: {p}")

    # Correctness runs at the case's own token by default, because the FlyDSL
    # kernel pair is selected per M bucket: shrinking the token count for
    # correctness would check a different kernel than performance measures. The
    # torch reference is dominated by dequantizing all 896 expert weights, which
    # is token-independent (~0.4 s / ~37 GiB at token=7211 on MI355X), so there is
    # nothing to save by shrinking it. ``correctness_max_token`` stays as an
    # escape hatch; when it lowers the token, run_correctness proves the bucket
    # is unchanged rather than trusting it.
    token = p["token"]
    if correctness:
        token = int(case.get("correctness_token", token))
        cap = MOE_CONFIG.get("correctness_max_token")
        if cap:
            token = min(token, int(cap))
    experts, model_dim, inter_dim, topk = (
        p["experts"], p["model_dim"], p["inter_dim"], p["topk"]
    )
    quant_type = aiter.QuantType.per_1x32
    activation = _activation(aiter)

    torch.manual_seed(int(case.get("seed", 19)))
    hidden = torch.randn((token, model_dim), device="cuda", dtype=dtypes.bf16) * 0.1
    w1 = torch.randn((experts, inter_dim * 2, model_dim), device="cuda", dtype=dtypes.bf16) * 0.03
    w2 = torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=dtypes.bf16) * 0.03
    score = torch.randn((token, experts), device="cuda", dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)

    torch_quant = aiter.get_torch_quant(quant_type)
    # Weight rows are laid out GGUU (all gate rows, then all up rows), matching the
    # SEPARATED gate mode and torch_moe_stage1's out.split([inter, inter], -1).
    w1_quant, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_quant, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    # gate_up=False keeps the GGUU (SEPARATED) layout the SiTU a16w4 kernel wants.
    n_lane = int(MOE_CONFIG.get("n_lane", 16))
    w1_runtime = shuffle_weight_a16w4(w1_quant, n_lane, False)
    w2_runtime = shuffle_weight_a16w4(w2_quant, n_lane, False)
    w1_scale_runtime = shuffle_scale_a16w4(w1_scale, experts, False)
    w2_scale_runtime = shuffle_scale_a16w4(w2_scale, experts, False)
    _assert_vllm_shuffle_contract(torch, n_lane, w2_scale, w2_scale_runtime)

    return {
        "cfg": case, "params": p, "token": token, "topk": topk,
        "hidden": hidden,
        "w1": w1_runtime, "w2": w2_runtime,
        "w1_scale": w1_scale_runtime, "w2_scale": w2_scale_runtime,
        # Unshuffled copies drive the torch reference.
        "w1_reference": w1_quant, "w2_reference": w2_quant,
        "w1_scale_reference": w1_scale, "w2_scale_reference": w2_scale,
        "topk_weights": topk_weights, "topk_ids": topk_ids,
        "quant_type": quant_type, "activation": activation,
    }


def _run(inputs: dict):
    torch = _torch()
    from aiter.fused_moe import fused_moe

    return fused_moe(
        inputs["hidden"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        quant_type=inputs["quant_type"],
        activation=inputs["activation"],
        dtype=torch.bfloat16,
        gate_mode=MOE_CONFIG["gate_mode"],
        beta=MOE_CONFIG["situ_beta"],
        linear_beta=MOE_CONFIG["situ_linear_beta"],
    )


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _reference(inputs: dict):
    """Dequantized torch reference for the same two-stage MoE.

    Runs aiter's own ``torch_moe_stage1``/``torch_moe_stage2``, which unpack the
    mxfp4 nibbles, apply the per-1x32 e8m0 group scales and accumulate in fp32 --
    i.e. a real independent implementation of the op, not a wrapper around the
    kernel under test. Activation params are pinned to the session config so the
    reference cannot silently drift onto the library defaults.
    """
    torch = _torch()
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    token, topk = inputs["token"], inputs["topk"]
    # a16w4: the activation stays bf16 and carries no scale.
    stage1 = torch_moe_stage1(
        inputs["hidden"],
        inputs["w1_reference"],
        inputs["w2_reference"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        activation=inputs["activation"],
        quant_type=inputs["quant_type"],
        a1_scale=None,
        w1_scale=inputs["w1_scale_reference"],
        situ_beta=MOE_CONFIG["situ_beta"],
        situ_linear_beta=MOE_CONFIG["situ_linear_beta"],
    )
    return torch_moe_stage2(
        stage1.view(token, topk, -1),
        inputs["w1_reference"],
        inputs["w2_reference"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        quant_type=inputs["quant_type"],
        w2_scale=inputs["w2_scale_reference"],
        a2_scale=None,
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _prepare(CASES[0], correctness=True)
    out = _run(inputs)
    _torch().cuda.synchronize()
    kn1, kn2, blob = _dispatch_names(inputs)
    _assert_tuned_dispatch("compile", kn1, kn2, blob)
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)}")
    print(f"  tuned dispatch OK  stage1={kn1}\n                     stage2={kn2}")


# Correctness is repeated because stage2 reduces with atomics, so the result is
# not bit-reproducible (observed cosine spread ~1.5e-5). A single pass can be
# lucky; gate on the worst of N.
_CORRECTNESS_REPEATS = 3


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _prepare(case, correctness=True)
        tol = case["params"]
        token_used = inputs["token"]

        kn1, kn2, blob = _dispatch_names(inputs)
        _assert_tuned_dispatch(case["id"], kn1, kn2, blob)

        # If someone shrinks the correctness token, prove the M bucket -- and so
        # the kernel pair -- is unchanged instead of assuming it.
        perf_token = case["params"]["token"]
        if not case.get("correctness_only") and token_used != perf_token:
            perf_inputs = _prepare(case, correctness=False)
            p_kn1, p_kn2, p_blob = _dispatch_names(perf_inputs)
            _assert_tuned_dispatch(f"{case['id']} (perf token)", p_kn1, p_kn2, p_blob)
            del perf_inputs
            _free()
            if (kn1, kn2) != (p_kn1, p_kn2):
                raise AssertionError(
                    f"{case['id']}: correctness runs at token={token_used} which "
                    f"dispatches ({kn1}, {kn2}), but performance is measured at "
                    f"token={perf_token} which dispatches ({p_kn1}, {p_kn2}). The "
                    f"scored kernel would never be correctness-checked. Set "
                    f"correctness_token to the performance token, or add an "
                    f"mbucket-* case for that bucket."
                )

        expected = _reference(inputs)
        worst_cos, worst_err = 1.0, 0.0
        for _ in range(_CORRECTNESS_REPEATS):
            got = _run(inputs)
            torch.cuda.synchronize()
            assert torch.isfinite(got).all(), (case["id"], "non-finite output")
            assert tuple(got.shape) == tuple(expected.shape), (
                case["id"], tuple(got.shape), tuple(expected.shape)
            )
            g, e = got.float().flatten(), expected.float().flatten()
            worst_cos = min(
                worst_cos, torch.nn.functional.cosine_similarity(g, e, dim=0).item()
            )
            worst_err = max(worst_err, ((g - e).norm() / e.norm().clamp_min(1e-8)).item())

        assert worst_cos > tol.get("min_cosine", 0.97), (
            case["id"], f"worst-of-{_CORRECTNESS_REPEATS} cosine {worst_cos:.6f} "
            f"vs torch reference too low"
        )
        assert worst_err < tol.get("max_rel_norm_err", 0.25), (
            case["id"], f"worst-of-{_CORRECTNESS_REPEATS} relative norm error "
            f"{worst_err:.4f} too high"
        )
        print(f"correctness PASS {case['id']:34s} token={token_used:<6d} "
              f"cos={worst_cos:.6f} rel_err={worst_err:.4f}")
        del inputs, expected, got, g, e
        _free()


def run_performance() -> None:
    rows = []
    for case in CASES:
        if case.get("correctness_only"):
            continue
        inputs = _prepare(case, correctness=False)
        _run(inputs)
        _torch().cuda.synchronize()
        # The scored path must be the tuned FlyDSL path the session ran, not the
        # heuristic fallback; record which pair was actually timed.
        kn1, kn2, blob = _dispatch_names(inputs)
        _assert_tuned_dispatch(case["id"], kn1, kn2, blob)
        bench = case.get("benchmark", {})
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 20),
            target_ms=bench.get("target_ms", 1.0),
            max_graph_repeats=bench.get("max_graph_repeats", 100),
        )
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "session_breakdown_id": case.get("session_breakdown_id"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
            "dispatched_stage1_kernel": kn1,
            "dispatched_stage2_kernel": kn2,
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            # Flat, not nested: src/testcases.py reads benchmark_method from the top
            # level of each row when it builds TestCaseResult.metadata.
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
