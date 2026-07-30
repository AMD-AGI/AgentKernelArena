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
import json
import os
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

    # The float64-free torch reference dequantizes every expert weight, so trim the
    # token count for correctness; expert/dim geometry stays at the session values
    # so the same FlyDSL kernel pair is dispatched.
    token = min(p["token"], MOE_CONFIG["correctness_max_token"]) if correctness else p["token"]
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
    w1_runtime = shuffle_weight_a16w4(w1_quant, 16, False)
    w2_runtime = shuffle_weight_a16w4(w2_quant, 16, False)
    w1_scale_runtime = shuffle_scale_a16w4(w1_scale, experts, False)
    w2_scale_runtime = shuffle_scale_a16w4(w2_scale, experts, False)

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
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inputs = _prepare(case, correctness=True)
        got = _run(inputs)
        torch.cuda.synchronize()
        expected = _reference(inputs)

        assert torch.isfinite(got).all(), (case["id"], "non-finite output")
        assert tuple(got.shape) == tuple(expected.shape), (
            case["id"], tuple(got.shape), tuple(expected.shape)
        )
        g, e = got.float().flatten(), expected.float().flatten()
        cos = torch.nn.functional.cosine_similarity(g, e, dim=0).item()
        rel_norm = ((g - e).norm() / e.norm().clamp_min(1e-8)).item()
        tol = case["params"]
        assert cos > tol.get("min_cosine", 0.97), (
            case["id"], f"cosine {cos:.6f} vs torch reference too low"
        )
        assert rel_norm < tol.get("max_rel_norm_err", 0.25), (
            case["id"], f"relative norm error {rel_norm:.4f} too high"
        )
        print("correctness PASS", case["id"], f"cos={cos:.6f} rel_err={rel_norm:.4f}")


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _prepare(case, correctness=False)
        _run(inputs)
        _torch().cuda.synchronize()
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
