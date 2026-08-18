#!/usr/bin/env python3
"""Image-kernel harness for the GLM-5.2 routed-expert MoE (aiter MXFP4 2-stage).

Reproduces the MoE path of Hyperloom session GLM-5.2-MXFP4_20260814T163244Z on
MI355X/gfx950. The timed callable is the whole ``aiter.fused_moe`` for one MoE
layer, which is deliberate: the session's MoE cost is not just the two FlyDSL
grouped GEMMs (decode 15.08% + 7.86%) but also the HIP family fused_moe launches
around them -- ``opus_moe_sorting_entry<P0_v2>`` and ``<P23>``, the two
``fused_mx_quant_moe_sort_kernel`` passes, ``grouped_topk_kernel``,
``moe_reduction_kernel_plain_bf16_topk9_md6144`` and
``_fused_append_shared_experts_kernel`` (another 12.5% of decode GPU time
combined). Timing the whole op keeps all of that on the scoreboard.

Quantization contract, verified against the in-image sources rather than assumed:

  * GLM-5.2 is **afp4_wfp4** -- BOTH operands are MXFP4. The weights come
    pre-quantized from the quark checkpoint; the activation is dynamically
    quantized inside ``fused_moe``. This is a different path from the Kimi-K3
    task's a16w4 (``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``), and the
    two are not interchangeable.
  * Weight prep is ``e8m0_shuffle`` on the scales (viewed 2D and restored) plus
    ``shuffle_weight(w, (16, 16))`` on the packed weights, copied from
    quark_w4a4_mxfp4_moe.py:562-597.
  * The activation is ``ActivationType.Silu``. sglang's aiter runner only sets
    ``gate_mode`` / ``beta`` / ``swiglu_limit`` for situ or swiglu_limit>0
    models, and GLM-5.2 is neither, so the defaults are the session behaviour.
  * ``doweight_stage1=False``: the top-k weights are applied in the stage-2
    reduction, which is what ``moe_reduction_kernel_plain_bf16_topk9_md6144``
    is doing in the trace.

Routing structure is faithful too: topk=9 is 8 routed experts plus the shared
expert, which sglang fuses in as expert id 256 (hence expert count 257 and the
``_fused_append_shared_experts_kernel`` at 75 calls/step). Column 8 of
``topk_ids`` is always the shared expert.
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
CFG = SPEC["moe_config"]

HIDDEN = CFG["model_dim"]
INTER = CFG["inter_dim"]
EXPERTS = CFG["experts"]
ROUTED = CFG["n_routed_experts"]
TOPK = CFG["topk"]
SHARED_ID = CFG["shared_expert_id"]
ROUTED_SCALE = CFG["routed_scaling_factor"]

def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.environ.setdefault("SGLANG_USE_AITER", "1")

    # The agent edits the workspace-seeded copy of aiter, so it must shadow the
    # in-image install. Three details, each of which silently benchmarks the
    # WRONG code if you get it wrong:
    #
    #  * ``image_repo_path`` is the aiter REPO ROOT, so the package lives at
    #    <ws>/aiter/aiter/__init__.py and the path entry has to be <ws>/aiter,
    #    not <ws>. Pointing at <ws> makes `import aiter` see <ws>/aiter as a
    #    namespace portion (no __init__.py); Python then keeps scanning and the
    #    in-image regular package at /sgl-workspace/aiter wins.
    #  * aiter is an editable install whose .pth appends /sgl-workspace/aiter to
    #    sys.path, so inserting at position 0 is what puts the copy in front.
    #  * Do NOT set AITER_JIT_DIR. get_user_jit_dir() (jit/core.py:438) returns
    #    it verbatim, and an empty dir means every aiter module rebuilds from
    #    scratch. Left unset it falls through to `this_dir` -- the workspace's own
    #    <ws>/aiter/aiter/jit, which is writable AND already holds the copied
    #    prebuilt .so and flydsl_cache. Builds stay inside the workspace either
    #    way; this way they start warm.
    seeded_root = WORKSPACE / "aiter"
    if (seeded_root / "aiter" / "__init__.py").is_file():
        sys.path.insert(0, str(seeded_root))
    os.chdir(WORKSPACE)


def _assert_aiter_is_workspace_copy() -> None:
    """Refuse to run against the in-image aiter when a workspace copy was seeded.

    Failing closed here is the whole point: if the import resolves past the
    workspace, an agent's edits are invisible and every number this harness
    prints describes the original kernel.
    """
    import aiter

    seeded_root = WORKSPACE / "aiter"
    if not (seeded_root / "aiter" / "__init__.py").is_file():
        return  # standalone run against the image tree; nothing to shadow
    resolved = Path(aiter.__file__).resolve()
    if seeded_root.resolve() not in resolved.parents:
        raise RuntimeError(
            f"aiter resolved to {resolved}, not the workspace copy under "
            f"{seeded_root}. Source edits would be ignored. Check sys.path[0] "
            f"and the __editable__ .pth for amd-aiter."
        )


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


def _assert_weight_prep_contract() -> None:
    """Fail loudly if the sglang weight-prep this harness copies has moved.

    quark_w4a4_mxfp4_moe.py:562-597 is the source of truth for how GLM-5.2's
    MoE weights reach aiter. If a patch (or an image bump) changes the shuffle,
    the harness would silently benchmark a layout the server never produces.
    """
    from aiter.ops.shuffle import shuffle_weight  # noqa: F401
    from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: F401


def _aiter_torch_quant():
    """aiter's own torch-side MXFP4 quantizer for the per-1x32 scheme.

    This is the quantizer aiter validates ``fused_moe`` against
    (op_tests/test_moe_2stage.py:106 ``torch_quant = aiter.get_torch_quant(qType)``).
    It is NOT interchangeable with ``fp4_utils.dynamic_mxfp4_quant``: the triton
    quantizer rounds the e8m0 block scale differently, and substituting it leaves
    a ~15% mean relative error against the kernel that has nothing to do with the
    kernel being wrong.
    """
    import aiter

    return aiter.get_torch_quant(aiter.QuantType.per_1x32)


def _make(case: dict) -> dict:
    """Build one case at its scored shape.

    No correctness/performance switch: the shape that is timed is the shape that
    is validated. Only ``compile`` shrinks anything.
    """
    torch = _torch()
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

    _assert_weight_prep_contract()
    _assert_aiter_is_workspace_copy()

    tokens = int(case["params"]["tokens"])
    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(case["params"]["seed"]))

    x = torch.randn(
        (tokens, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=gen
    ) * 0.25

    w1_bf16 = (
        torch.randn(
            (EXPERTS, INTER * 2, HIDDEN), device="cuda", dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.125
    )
    w2_bf16 = (
        torch.randn(
            (EXPERTS, HIDDEN, INTER), device="cuda", dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.125
    )

    # Quantize to the checkpoint layout: MXFP4, group_size=32, e8m0 scales.
    q1, s1 = dynamic_mxfp4_quant(w1_bf16.reshape(-1, HIDDEN))
    q1 = q1.view(EXPERTS, INTER * 2, -1)
    s1 = s1.view(EXPERTS, INTER * 2, -1)
    q2, s2 = dynamic_mxfp4_quant(w2_bf16.reshape(-1, INTER))
    q2 = q2.view(EXPERTS, HIDDEN, -1)
    s2 = s2.view(EXPERTS, HIDDEN, -1)

    # sglang quark_w4a4_mxfp4_moe.py:576-596 -- scales e8m0_shuffle'd through a
    # 2D view, weights shuffle_weight'd with (16, 16).
    a, b, _ = s1.shape
    s1_rt = e8m0_shuffle(s1.reshape(a * b, -1)).view(a, b, -1)
    a, b, _ = s2.shape
    s2_rt = e8m0_shuffle(s2.reshape(a * b, -1)).view(a, b, -1)
    w1_rt = shuffle_weight(q1.contiguous(), (16, 16))
    w2_rt = shuffle_weight(q2.contiguous(), (16, 16))

    # Routing: 8 distinct routed experts + the fused shared expert in column 8.
    scores = torch.rand((tokens, ROUTED), device="cuda", generator=gen)
    routed_ids = scores.topk(TOPK - 1, dim=-1).indices.to(torch.int32)
    routed_w = torch.softmax(
        scores.gather(1, routed_ids.long()).float(), dim=-1
    ) * ROUTED_SCALE
    shared_ids = torch.full(
        (tokens, 1), SHARED_ID, device="cuda", dtype=torch.int32
    )
    shared_w = torch.ones((tokens, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.cat([routed_ids, shared_ids], dim=1).contiguous()
    topk_weight = torch.cat([routed_w, shared_w], dim=1).contiguous()

    return {
        "cfg": case,
        "aiter": aiter,
        "fused_moe": __import__(
            "aiter.fused_moe", fromlist=["fused_moe"]
        ).fused_moe,
        "quant_type": QuantType.per_1x32,
        "activation": ActivationType.Silu,
        "x": x,
        "w1": w1_rt,
        "w2": w2_rt,
        "w1_scale": s1_rt,
        "w2_scale": s2_rt,
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        # Unshuffled copies drive the torch reference.
        "w1_q": q1,
        "w1_s": s1,
        "w2_q": q2,
        "w2_s": s2,
    }


def _run(inputs: dict):
    return inputs["fused_moe"](
        hidden_states=inputs["x"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_weight=inputs["topk_weight"],
        topk_ids=inputs["topk_ids"],
        quant_type=inputs["quant_type"],
        activation=inputs["activation"],
        doweight_stage1=False,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
    )


def _reference(inputs: dict):
    """aiter's own two-stage torch reference, driven at the session's config.

    Uses ``torch_moe_stage1`` / ``torch_moe_stage2`` rather than a hand-rolled
    dequant-and-matmul. They unpack the MXFP4 nibbles, apply the per-1x32 e8m0
    group scales and accumulate in fp32, and they are the references aiter itself
    validates ``fused_moe`` against, so the harness cannot drift from the kernel
    over a detail of the quantization contract.

    They are also fully batched -- one grouped GEMM per stage over all
    (token, topk) rows, no Python loop over tokens or experts -- so the full
    16384-token case stays affordable.

    Both activations are quantized with aiter's torch quantizer because this is
    the **a4w4** path: the activation is MXFP4 too, once going into stage 1 and
    again on the stage-1 output going into stage 2 (the trace's two
    ``fused_mx_quant_moe_sort_kernel`` launches).
    """
    torch = _torch()
    import aiter
    from aiter import dtypes
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    torch_quant = _aiter_torch_quant()
    x = inputs["x"]
    tokens = x.shape[0]

    a1_qt, a1_scale = torch_quant(x, quant_dtype=dtypes.fp4x2)
    out1 = torch_moe_stage1(
        a1_qt,
        inputs["w1_q"],
        inputs["w2_q"],
        inputs["topk_weight"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        activation=inputs["activation"],
        quant_type=aiter.QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=inputs["w1_s"],
        doweight=False,
    )
    a2_qt, a2_scale = torch_quant(out1, quant_dtype=dtypes.fp4x2)
    a2_qt = a2_qt.view(tokens, TOPK, -1)
    if a2_scale is not None:
        a2_scale = a2_scale.view(tokens, TOPK, -1)
    return torch_moe_stage2(
        a2_qt,
        inputs["w1_q"],
        inputs["w2_q"],
        inputs["topk_weight"],
        inputs["topk_ids"],
        dtype=torch.bfloat16,
        quant_type=aiter.QuantType.per_1x32,
        w2_scale=inputs["w2_s"],
        a2_scale=a2_scale,
        doweight=True,
    )


def _assert_close(inputs: dict, got) -> None:
    torch = _torch()
    ref = _reference(inputs)
    got = got.reshape(ref.shape).float()
    ref = ref.float()
    # MXFP4 on both operands plus a 257-way fp32 accumulation ordering
    # difference: compare on a relative-error basis against the reference's own
    # scale rather than elementwise absolute tolerance.
    denom = ref.abs().mean().clamp_min(1e-6)
    rel = (got - ref).abs().mean() / denom
    max_rel = float(inputs["cfg"]["params"].get("max_relerr", 0.01))
    if not torch.isfinite(got).all():
        raise AssertionError("kernel output contains non-finite values")
    if rel.item() > max_rel:
        raise AssertionError(
            f"mean relative error {rel.item():.4f} exceeds {max_rel} "
            f"for case {inputs['cfg']['id']}"
        )


def _perturb_inputs(inputs: dict) -> None:
    """Refresh the activation through its captured address.

    Only ``x`` is data. The weights and the routing are workload structure: a
    replayed graph captured the expert assignment, and re-drawing it would change
    the sorting/padding work the benchmark is measuring.
    """
    torch = _torch()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(47)
    inputs["x"].normal_(generator=gen)
    inputs["x"].mul_(0.25)


def _compile_smoke_case(case: dict) -> dict:
    """Shrink a case so the compile smoke test stays cheap.

    Only ``compile`` may use this. Expert count, hidden and inter dims are left
    alone because they select the FlyDSL kernel variant.
    """
    smoke = {**case, "params": dict(case["params"])}
    smoke["params"]["tokens"] = min(int(case["params"]["tokens"]), 64)
    return smoke


def _assert_timed_outputs(inputs: dict, timed) -> None:
    """Validate the invocation the benchmark actually timed."""
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
            warmup=10,
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
            "m_routed": int(case["params"]["tokens"]) * TOPK,
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
