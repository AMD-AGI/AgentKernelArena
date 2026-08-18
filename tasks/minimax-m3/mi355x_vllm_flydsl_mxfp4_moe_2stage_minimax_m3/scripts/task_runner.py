#!/usr/bin/env python3
"""Harness for MiniMax-M3's routed-expert MoE 2-stage GEMM (aiter, MXFP4 x MXFP4).

One `aiter.fused_moe.fused_moe()` entry, two kernel families selected by token
count -- exactly as the session runs it:

  token = 8192 (chunked-prefill step) -> aiter FlyDSL MFMA pair
        mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_v32
        mfma_moe2_afp4_wfp4_bf16_cshuffle_t128x128x256_..._persist_cu256
  token = 64   (pure-decode step)     -> CK-Tile flat GEMM pair
        ck_tile::MoeFlatmmKernel<MoeFlatmmKind=3>  (FFN gemm1, split-k)
        ck_tile::MoeFlatmmKernel<MoeFlatmmKind=2>  (FFN gemm2)

Both token counts are scored in BOTH MoE tasks so the report always shows the
full MoE picture; each task's edit surface only reaches one of the two families.

The weight preparation reproduces vLLM's AITER_MXFP4_MXFP4 loader
(vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:973-1014) rather than
guessing: e8m0_shuffle on both scale tensors, view as float4_e2m1fn_x2, then
rocm_aiter_ops.shuffle_weights on the packed weights.

Correctness and performance run the SAME case list at the SAME sizes, and
performance is measured under a CUDA/HIP graph.
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
MOE_CONFIG = SPEC["moe_config"]

_IMAGE_AITER_META = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta")


def _link_aiter_meta() -> None:
    """Give a workspace copy of `aiter` its sibling C++ tree.

    aiter locates aiter_enum.h relative to the directory that *contains* the
    package (aiter/utility/aiter_types.py:_find_aiter_enum_h uses parents[2] and
    ignores AITER_META_DIR), so a workspace that only seeds `aiter` cannot import
    it at all without this link.
    """
    link = WORKSPACE / "aiter_meta"
    if link.exists() or link.is_symlink():
        return
    if _IMAGE_AITER_META.is_dir():
        link.symlink_to(_IMAGE_AITER_META, target_is_directory=True)


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    if (WORKSPACE / "aiter").is_dir():
        # FlyDSL task: the agent edits the seeded python package, which must
        # shadow the in-image install.
        #
        # Deliberately NOT redirecting AITER_JIT_DIR. Left unset, aiter defaults
        # to <workspace>/aiter/jit -- the copy that already carries all 108
        # prebuilt .so modules, and which is per-run anyway, so isolation is
        # preserved. Redirecting it to an empty build dir instead forces a
        # from-source rebuild of whatever C++ module a case happens to need: the
        # token=64 case reaches module_moe_cktile2stages, a ~6 minute CK-Tile
        # build that is not even this task's edit surface. FlyDSL kernels are
        # generated from the (editable) Python DSL at call time and do not go
        # through this .so mechanism, so agent edits still take effect.
        _link_aiter_meta()
        sys.path.insert(0, str(WORKSPACE))
        os.environ.setdefault("AITER_META_DIR", str(_IMAGE_AITER_META))
    elif (WORKSPACE / "aiter_meta").is_dir():
        # CK-Tile task: the agent edits C++/HIP sources, so point aiter's JIT at
        # the workspace tree and force a rebuild. Here the rebuild IS the point,
        # and the JIT output must not land in the image's aiter package.
        os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "jit"))
        os.environ["AITER_META_DIR"] = str(WORKSPACE / "aiter_meta")
        os.environ.setdefault("AITER_REBUILD", "1")
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


def _aiter():
    import aiter

    return aiter


def _assert_vllm_contract(aiter, p: dict) -> None:
    """Pin the two enum choices vLLM makes, so a patch that needs different ones
    cannot pass here and then silently mismatch in production."""
    if p["quant_type"] != "per_1x32":
        raise AssertionError(f"session quant_type is per_1x32, got {p['quant_type']}")
    if not hasattr(aiter.ActivationType, "Swiglu"):
        raise AssertionError("this aiter build has no ActivationType.Swiglu")
    if p["inter_dim"] * 2 != MOE_CONFIG["w1_n"]:
        raise AssertionError(
            f"w1 N ({MOE_CONFIG['w1_n']}) must be 2*inter_dim ({p['inter_dim']})"
        )


def _make(case: dict) -> dict:
    """Build one case's inputs.

    Named ``_make`` (not ``_prepare``) on purpose: forge-loop's shipped
    ``scripts/forge_driver.py`` calls ``tr._make(case)`` for its --profile-run
    contract. A task that names it anything else fails the profiling preflight
    and makes forge-loop spend an LLM prep agent re-authoring a driver on every
    single run.
    """
    torch = _torch()
    aiter = _aiter()
    from aiter import dtypes
    from aiter.utility.fp4_utils import e8m0_shuffle
    from vllm._aiter_ops import rocm_aiter_ops

    p = dict(case["params"])
    _assert_vllm_contract(aiter, p)

    token = p["token"]
    experts, model_dim, inter_dim, topk = (
        p["experts"], p["model_dim"], p["inter_dim"], p["topk"]
    )
    quant_type = aiter.QuantType.per_1x32
    activation = aiter.ActivationType.Swiglu

    torch.manual_seed(int(case.get("seed", 19)))
    hidden = torch.randn((token, model_dim), device="cuda", dtype=dtypes.bf16) * 0.1
    w1 = (
        torch.randn(
            (experts, inter_dim * 2, model_dim), device="cuda", dtype=dtypes.bf16
        )
        * 0.03
    )
    w2 = (
        torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=dtypes.bf16)
        * 0.03
    )

    # Router: M3 uses sigmoid scoring with a routing bias, but the MoE kernel only
    # consumes (topk_weights, topk_ids); the router itself is a separate GEMM and
    # is not part of this task. Draw a realistic, seeded assignment.
    score = torch.randn((token, experts), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.sigmoid(score), topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(
        torch.float32
    ) * MOE_CONFIG["routed_scaling_factor"]
    topk_ids = topk_ids.to(torch.int32)

    torch_quant = aiter.get_torch_quant(quant_type)
    w1_quant, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_quant, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    # ---- vLLM AITER_MXFP4_MXFP4 loader contract (oracle/mxfp4.py:973-1014) ----
    # vLLM holds the scales as [E, N, K/32] from the checkpoint and reshapes to
    # [E*N, K/32] before e8m0_shuffle; aiter's get_torch_quant already hands back
    # that flat layout, so shuffle it directly. Same buffer, same bytes.
    w1_scale_rt = e8m0_shuffle(w1_scale)
    w2_scale_rt = e8m0_shuffle(w2_scale)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    w1_view = w1_quant.view(fp4_dtype) if fp4_dtype is not None else w1_quant
    w2_view = w2_quant.view(fp4_dtype) if fp4_dtype is not None else w2_quant
    w1_rt, w2_rt = rocm_aiter_ops.shuffle_weights(w1_view, w2_view)
    # -------------------------------------------------------------------------

    return {
        "cfg": case,
        "params": p,
        "token": token,
        "topk": topk,
        "hidden": hidden,
        "w1": w1_rt,
        "w2": w2_rt,
        "w1_scale": w1_scale_rt,
        "w2_scale": w2_scale_rt,
        # Unshuffled copies drive the torch reference.
        "w1_reference": w1_quant,
        "w2_reference": w2_quant,
        "w1_scale_reference": w1_scale,
        "w2_scale_reference": w2_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "quant_type": quant_type,
        "activation": activation,
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
    )


def _reference(inputs: dict):
    """Dequantized torch reference for the same two-stage MoE.

    Uses aiter's own torch_moe_stage1/torch_moe_stage2, which unpack the mxfp4
    nibbles, apply the per-1x32 e8m0 group scales and accumulate in fp32 -- an
    independent implementation of the op, not a wrapper around the kernel under
    test. Both stages are single batched GEMMs over the expert dimension, so the
    reference is vectorized, not a per-expert Python loop.
    """
    torch = _torch()
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    token, topk = inputs["token"], inputs["topk"]
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


def _moe_deviation(got, expected):
    torch = _torch()
    g, e = got.float().flatten(), expected.float().flatten()
    cos = torch.nn.functional.cosine_similarity(g, e, dim=0).item()
    err = ((g - e).norm() / e.norm().clamp_min(1e-8)).item()
    return cos, err


def _assert_moe_within_tolerance(case: dict, cos: float, err: float, label: str) -> None:
    tol = case["params"]
    assert cos > tol.get("min_cosine", 0.97), (
        case["id"],
        f"{label} cosine {cos:.6f} vs torch reference too low",
    )
    assert err < tol.get("max_rel_norm_err", 0.25), (
        case["id"],
        f"{label} relative norm error {err:.4f} too high",
    )


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the activation in place so a replayed graph sees fresh values.

    Only ``hidden`` is redrawn: the routing tensors are kernel inputs rather than
    something the kernel derives, and the quantized weights have shuffled runtime
    copies that would have to be rebuilt in lockstep.
    """
    torch = _torch()
    torch.manual_seed(43)
    inputs["hidden"].normal_(0.0, 0.1)


def _assert_timed_outputs(case: dict, inputs: dict, timed) -> None:
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb_inputs(inputs)
    got = timed.rerun()
    cos, err = _moe_deviation(got, _reference(inputs))
    _assert_moe_within_tolerance(case, cos, err, "timed")


def run_compile() -> None:
    inputs = _make(CASES[0])
    out = _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _make(case)
        got = _run(inputs)
        torch.cuda.synchronize()
        cos, err = _moe_deviation(got, _reference(inputs))
        _assert_moe_within_tolerance(case, cos, err, "correctness")
        print(f"correctness PASS {case['id']}  cos={cos:.6f} rel_err={err:.4f}")
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
        _assert_timed_outputs(case, inputs, timed)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "session_id": case.get("session_id"),
            "phase": case.get("phase"),
            "gpu_pct": case.get("gpu_pct"),
            "kernel_family": case.get("kernel_family"),
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
