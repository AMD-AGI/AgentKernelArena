#!/usr/bin/env python3
"""Image-kernel harness for the Kimi-K3 MoE routing / sorting / MX-quant stages.

Scope
-----
The MoE pipeline stages that sit AROUND the 2-stage expert GEMM, i.e. everything
`mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3` does NOT cover:

  * ``aiter::grouped_topk_kernel``                        2.75% E2E  (csrc/kernels/topk_softmax_kernels_group.cu:320)
  * ``aiter::opus_moe_sorting_entry<...P23...>``          1.20% E2E  (csrc/include/moe_sorting_opus.h:109)
  * ``aiter::opus_moe_sorting_entry<...P0_v2...>``        0.90% E2E  (same file, multi-phase)
  * ``aiter::fused_mx_quant_moe_sort_kernel<bf16,fp8,..>``1.50% E2E  (csrc/kernels/quant_kernels.cu:1731)

6.35% of end-to-end GPU time together -- more than any single MoE GEMM kernel in
the session (largest is 2.92%). Each fires 92 times per forward pass, matching the
92 MoE layers (``first_k_dense_replace=1`` of 93).

Chain under test
----------------
The three stages are timed as one chain because that is how the model runs them::

    biased_grouped_topk_hip(gating[T,896], bias[896]) -> topk_weights[T,16] f32,
                                                          topk_ids[T,16] i32
    moe_sorting_opus_fwd(topk_ids, topk_weights)      -> sorted_ids, sorted_weights,
                                                          sorted_expert_ids, num_valid_ids
    fused_dynamic_mx_quant_moe_sort_hip(latent[T,3584] bf16, sorted_ids, ...)
                                                      -> fp8_e4m3 out + e8m0 scales

``quant`` is in the chain only for the small-M / decode cases: at big M the session's
MoE1 kernel is ``mfma_moe1_..._fp8q_sort_async_...``, which folds the quant into the
GEMM prologue, so the standalone quant kernel measures 0.02% in prefill.

Goldens
-------
All references are vectorized torch (no per-token Python loops):

  * **topk** -- full torch reference of the biased grouped top-k, compared on the
    selected expert set and on the renormalized weights.
  * **sorting** -- ordering-agnostic *semantic* check. ``sorted_ids`` encodes
    ``(slot_id << 24) | token_id`` (confirmed by
    ``op_tests/flydsl_tests/test_silu_and_mul_fq.py:42``); the harness decodes every
    valid slot, asserts its expert matches the owning block's ``sorted_expert_ids``,
    asserts the (token, slot) multiset is exactly the T*topk routing pairs, and
    asserts ``sorted_weights`` carries the matching ``topk_weights``. Intra-expert
    order and padding fill are deliberately NOT constrained -- the agent may change
    them.
  * **quant** -- dequantize ``out * 2^(e8m0 scale)`` per 32-element group and compare
    against the bf16 input.
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

_TOKEN_MASK = (1 << 24) - 1


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Prefer the workspace-seeded editable copy so the agent's csrc edits are the
    # ones aiter's JIT compiles; fall back to the in-image install for dev runs.
    seeded = WORKSPACE / "aiter"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("AITER_ROOT", "/sgl-workspace/aiter"))
    # aiter compiles csrc through its own JIT; point the build dir inside the
    # workspace so one workspace can never serve another workspace's .so.
    os.environ.setdefault("AITER_JIT_DIR", str(WORKSPACE / "build" / "aiter_jit"))
    # CRITICAL: aiter's get_module() only rebuilds on an ARCH mismatch -- it does
    # NOT hash the csrc sources (jit/core.py:633-659). Without this the agent's
    # edits to csrc/** are silently ignored and the prebuilt .so is served
    # (measured: edit + no AITER_REBUILD = 0.1 s and an unchanged .so mtime).
    # AITER_REBUILD=2 does rm_module() only (not clear_build()), so the ninja build
    # dir persists inside the workspace: measured ~53 s for all three target modules
    # cold (moe_asm 33 s + quant 13 s + moe_sorting_opus 7 s) and ~17 s per module
    # on a rebuild. Do NOT use AITER_REBUILD=1 -- that also wipes the build dir.
    # module_aiter_core is exempt (jit/core.py seeds it into rebuilded_list), so this
    # never triggers a full aiter build.
    os.environ.setdefault("AITER_REBUILD", "2")
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


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    """Build one case at its scored shape (same shape for correctness and perf)."""
    torch = _torch()
    p = case["params"]
    T = int(p["num_tokens"])
    E = int(p["num_experts"])
    topk = int(p["topk"])
    bs = int(p["block_size"])
    md = int(p["latent_dim"])
    stages = list(p["stages"])

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 9141)))

    gating = (torch.randn(T, E, device="cuda", dtype=torch.float32, generator=gen)
              * 0.5).to(torch.bfloat16).contiguous()
    bias = (torch.randn(E, device="cuda", dtype=torch.float32, generator=gen)
            * 0.1).to(torch.bfloat16).contiguous()
    topk_weights = torch.empty(T, topk, device="cuda", dtype=torch.float32)
    topk_ids = torch.empty(T, topk, device="cuda", dtype=torch.int32)

    # Sorting buffers, sized exactly the way aiter/fused_moe.py:203-210 does.
    max_padded = int(T * topk + E * bs - topk)
    max_blocks = int((max_padded + bs - 1) // bs)
    sorted_ids = torch.empty(max_padded, device="cuda", dtype=torch.int32)
    sorted_weights = torch.empty(max_padded, device="cuda", dtype=torch.float32)
    sorted_expert_ids = torch.empty(max_blocks, device="cuda", dtype=torch.int32)
    num_valid_ids = torch.empty(2, device="cuda", dtype=torch.int32)
    if p.get("moe_buf_mode") == "real":
        moe_buf = torch.empty(T, md, device="cuda", dtype=torch.bfloat16)
    else:
        moe_buf = torch.empty((0, 0), device="cuda", dtype=torch.bfloat16)

    inp = {
        "cfg": case, "T": T, "E": E, "topk": topk, "block_size": bs, "md": md,
        "stages": stages, "gating": gating, "bias": bias,
        "topk_weights": topk_weights, "topk_ids": topk_ids,
        "sorted_ids": sorted_ids, "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids, "num_valid_ids": num_valid_ids,
        "moe_buf": moe_buf, "max_padded": max_padded, "max_blocks": max_blocks,
        "num_expert_group": int(p["num_expert_group"]),
        "topk_group": int(p["topk_group"]),
        "need_renorm": bool(p["need_renorm"]),
        "routed_scaling_factor": float(p["routed_scaling_factor"]),
    }

    if "sorting" in stages:
        aiter = _aiter()
        ws_size = aiter.moe_sorting_opus_get_workspace_size(T, E, topk, 0)
        inp["workspace"] = (torch.empty(ws_size, device="cuda", dtype=torch.uint8)
                            if ws_size > 0 else None)

    if "quant" in stages:
        g = int(p.get("mx_group_size", MOE_CONFIG["mx_group_size"]))
        assert md % g == 0, (case["id"], "latent_dim must be a multiple of the MX group")
        inp["mx_group_size"] = g
        inp["quant_in"] = (torch.randn(T, md, device="cuda", dtype=torch.float32,
                                       generator=gen) * 0.5).to(torch.bfloat16).contiguous()
        inp["quant_out"] = torch.empty(T, md, device="cuda",
                                       dtype=torch.float8_e4m3fn)
        # Scale rows are the padded sorted-slot count rounded up to a block,
        # columns are md / group_size. Trace: (29696, 112) for T=64, md=3584, bs=32.
        inp["quant_scales"] = torch.empty(inp["max_blocks"] * bs, md // g,
                                          device="cuda", dtype=torch.float8_e8m0fnu)
    return inp


def _run(inp: dict):
    """Execute the session's routing chain in order. Returns nothing; the harness
    reads the output buffers, which is also what the model does."""
    aiter = _aiter()
    stages = inp["stages"]

    if "topk" in stages:
        aiter.biased_grouped_topk_hip(
            inp["gating"], inp["bias"], inp["topk_weights"], inp["topk_ids"],
            inp["num_expert_group"], inp["topk_group"], inp["need_renorm"],
            inp["routed_scaling_factor"],
        )
    if "sorting" in stages:
        aiter.moe_sorting_opus_fwd(
            inp["topk_ids"], inp["topk_weights"],
            inp["sorted_ids"], inp["sorted_weights"], inp["sorted_expert_ids"],
            inp["num_valid_ids"], inp["moe_buf"],
            inp["E"], int(inp["block_size"]),
            None, None, inp.get("workspace"), 0, None, None, None,
        )
    if "quant" in stages:
        aiter.fused_dynamic_mx_quant_moe_sort_hip(
            inp["quant_out"], inp["quant_scales"], inp["quant_in"],
            inp["sorted_ids"], inp["num_valid_ids"],
            inp["T"], int(inp["block_size"]), inp["mx_group_size"],
            inp["sorted_weights"],
        )
    return inp


# --------------------------------------------------------------------------- #
# References (vectorized torch)
# --------------------------------------------------------------------------- #
def _golden_topk(inp: dict):
    """Biased grouped top-k, exactly as the kernel defines it, fully vectorized.

    K3 has num_expert_group=1 / topk_group=1, so the group stage is a no-op and the
    reference reduces to: score = sigmoid(gating) (moe_router_activation_func), add
    the correction bias, take top-k on the biased score, but return the UNBIASED
    score as the weight, then renormalize and scale.
    """
    torch = _torch()
    g = inp["gating"].float()
    score = torch.sigmoid(g)
    biased = score + inp["bias"].float().unsqueeze(0)
    _, ids = torch.topk(biased, inp["topk"], dim=-1)
    w = torch.gather(score, 1, ids)
    if inp["need_renorm"]:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    w = w * inp["routed_scaling_factor"]
    return w, ids.to(torch.int32)


def _check_topk(inp: dict, case: dict) -> str:
    """Tie-safe validation of the biased grouped top-k.

    Comparing against a torch top-k *element by element* is not a valid test here:
    with 896 experts the 16th and 17th biased scores are routinely within fp32 noise
    of each other, so the kernel and torch legitimately pick different experts at the
    boundary, and any id-aligned weight diff then compares two unrelated experts.
    (Measured: that formulation reports 8.7e-3 max weight error on pure ties.)

    So the contract is checked in the two ways that are actually well defined:

      A. SELECTION -- the chosen set must be a valid top-k of the biased score:
         min(biased[chosen]) >= max(biased[not chosen]) - eps. Exact, tie-tolerant.
      B. WEIGHTS -- the returned weights must be the renormalized *unbiased* scores of
         the kernel's OWN chosen experts. Immune to which side of a tie was taken.
    """
    torch = _torch()
    got_ids = inp["topk_ids"].long()
    got_w = inp["topk_weights"]
    T, E, topk = inp["T"], inp["E"], inp["topk"]

    assert torch.isfinite(got_w).all(), (case["id"], "non-finite topk weights")
    assert int(got_ids.min()) >= 0 and int(got_ids.max()) < E, (
        case["id"], "topk id out of range"
    )
    assert (got_ids.sort(dim=-1).values.diff(dim=-1) > 0).all(), (
        case["id"], "duplicate expert inside one token's top-k list"
    )

    score = torch.sigmoid(inp["gating"].float())
    biased = score + inp["bias"].float().unsqueeze(0)

    # A. selection validity
    chosen_min = torch.gather(biased, 1, got_ids).min(dim=-1).values
    rest = biased.scatter(1, got_ids, float("-inf"))
    rest_max = rest.max(dim=-1).values
    sel_gap = (rest_max - chosen_min).max().item()          # <= 0 when strictly correct
    eps = case["params"].get("max_select_gap", 2e-3)
    assert sel_gap < eps, (
        case["id"],
        f"a non-selected expert outscores a selected one by {sel_gap:.2e} (> {eps})",
    )

    # B. weight consistency against the kernel's own ids
    w_ref = torch.gather(score, 1, got_ids)
    if inp["need_renorm"]:
        w_ref = w_ref / w_ref.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    w_ref = w_ref * inp["routed_scaling_factor"]
    w_err = (got_w - w_ref).abs().max().item()
    tol = case["params"].get("max_weight_err", 1e-3)
    assert w_err < tol, (case["id"], f"topk weight max err {w_err:.2e} > {tol}")

    # Informational: how far the selection drifted from a plain torch top-k.
    _, ref_ids = _golden_topk(inp)
    same = (got_ids.sort(dim=-1).values == ref_ids.long().sort(dim=-1).values)
    return (f"topk(w_err={w_err:.2e} sel_gap={sel_gap:.1e} "
            f"tie_drift={1.0 - same.float().mean().item():.2e})")


def _check_sorting(inp: dict, case: dict) -> str:
    """Ordering-agnostic semantic validation of the opus sort.

    sorted_ids packs (slot_id << 24) | token_id. Padding slots decode to a token id
    >= T. For every valid slot we require the owning block's expert to be the expert
    that (token, slot) actually routed to, and we require the set of valid
    (token, slot) pairs to be exactly the T*topk routing pairs -- nothing dropped,
    nothing duplicated. Intra-expert order and the padding fill are NOT constrained.
    """
    torch = _torch()
    T, topk, bs = inp["T"], inp["topk"], inp["block_size"]
    nvalid = int(inp["num_valid_ids"][0].item())
    assert 0 < nvalid <= inp["max_padded"], (case["id"], f"num_valid_ids={nvalid}")
    assert nvalid % bs == 0, (case["id"], f"num_valid_ids {nvalid} not block-aligned")

    ids = inp["sorted_ids"][:nvalid].long()
    tok = ids & _TOKEN_MASK
    slot = ids >> 24
    valid = tok < T
    assert int(valid.sum()) == T * topk, (
        case["id"], f"valid slot count {int(valid.sum())} != T*topk {T * topk}"
    )
    assert int(slot[valid].max()) < topk, (case["id"], "slot id out of range")

    # Each valid slot must sit in a block whose expert is the routed expert.
    blk_expert = inp["sorted_expert_ids"][: nvalid // bs].long()
    per_slot_expert = blk_expert.repeat_interleave(bs)
    routed = inp["topk_ids"].long()[tok[valid], slot[valid]]
    assert torch.equal(routed, per_slot_expert[valid]), (
        case["id"], "a sorted slot landed in a block for the wrong expert"
    )

    # Every (token, slot) pair appears exactly once.
    flat = (tok[valid] * topk + slot[valid])
    uniq = torch.unique(flat)
    assert uniq.numel() == T * topk, (
        case["id"], f"routing pairs not a permutation ({uniq.numel()} unique)"
    )

    # sorted_weights must carry the matching topk weight.
    w_err = (inp["sorted_weights"][:nvalid][valid]
             - inp["topk_weights"][tok[valid], slot[valid]]).abs().max().item()
    tol = case["params"].get("max_weight_err", 1e-3)
    assert w_err < tol, (case["id"], f"sorted_weights err {w_err:.2e} > {tol}")
    return f"sort(nvalid={nvalid} w_err={w_err:.2e})"


def _check_quant(inp: dict, case: dict) -> str:
    """Validate the MX quantization WITHOUT assuming the scale tensor's layout.

    ``out`` is token-row aligned ``[T, md]`` fp8_e4m3, but ``scales`` is written in
    sorted-slot order AND hardware-swizzled -- an inspected row reads
    ``[119, 0, 119, 0, ...]``, so a plain ``scales[t]`` lookup is meaningless (it
    dequantizes to zero). That layout is an implementation detail the agent is
    allowed to change, so pinning it here would be both wrong and over-constraining.

    Instead the quantization MATH is validated from the payload alone:

      * per 32-element group, the implied scale is ``amax(x) / amax(q)``; for a
        correct MX quantization that ratio must be a power of two, so
        ``e = round(log2(ratio))`` and ``|log2(ratio) - e|`` must be small;
      * ``q * 2^e`` must reproduce the bf16 input to within e4m3 precision
        (3 mantissa bits -> ~6% of the group max).

    The scale tensor itself is checked structurally: the number of rows it actually
    wrote must equal ``num_valid_ids[0]`` -- layout-independent, but still catches a
    sort/scale-placement regression.
    """
    torch = _torch()
    g = inp["mx_group_size"]
    T, md = inp["T"], inp["md"]
    ng = md // g
    x = inp["quant_in"].float().view(T, ng, g)
    q = inp["quant_out"].to(torch.float32).view(T, ng, g)
    assert torch.isfinite(q).all(), (case["id"], "non-finite fp8 payload")

    xa = x.abs().amax(dim=-1, keepdim=True)
    qa = q.abs().amax(dim=-1, keepdim=True)
    live = (xa > 0) & (qa > 0)                       # skip all-zero groups
    ratio = torch.where(live, xa / qa.clamp_min(1e-30), torch.ones_like(xa))
    log2r = torch.log2(ratio)
    e = torch.round(log2r)
    p2_err = (log2r - e).abs()[live].max().item() if live.any() else 0.0
    assert p2_err < 0.35, (
        case["id"],
        f"implied per-group scale is not a power of two (max |log2 dev| {p2_err:.3f}); "
        "the payload is not an MX quantization of the input",
    )

    deq = (q * torch.pow(2.0, e)).view(T, md)
    ref = inp["quant_in"].float()
    denom = ref.abs().amax().clamp_min(1e-8)
    rel = ((deq - ref).abs().amax() / denom).item()
    tol = case["params"].get("max_quant_rel_err", 0.08)
    assert torch.isfinite(deq).all(), (case["id"], "non-finite dequant")
    assert rel < tol, (case["id"], f"MX quant normalized max err {rel:.4f} > {tol}")

    nvalid = int(inp["num_valid_ids"][0].item())
    written = int((inp["quant_scales"].view(torch.uint8).sum(dim=-1) > 0).sum().item())
    assert written == nvalid, (
        case["id"],
        f"scale rows written = {written} but num_valid_ids = {nvalid}",
    )
    return f"quant(rel_max_err={rel:.4f} p2_dev={p2_err:.3f} scale_rows={written})"


def _golden(inp: dict):
    """Kept for the forge driver's SNR path: returns the reference topk weights."""
    ref_w, _ = _golden_topk(inp)
    return ref_w


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0])
    _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  stages={inp['stages']} "
          f"nvalid={int(inp['num_valid_ids'][0].item())}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case)
        _run(inp)
        torch.cuda.synchronize()
        notes = []
        if "topk" in inp["stages"]:
            notes.append(_check_topk(inp, case))
        if "sorting" in inp["stages"]:
            notes.append(_check_sorting(inp, case))
        if "quant" in inp["stages"]:
            notes.append(_check_quant(inp, case))
        print("correctness PASS", case["id"], f"[{case['phase']}]", " ".join(notes))
        del inp
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inp = _prepare(case)
        _run(inp)                       # settle aiter's JIT / kernel selection
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda i=inp: _run(i),
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 30),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        metadata = {
            **case["params"],
            "phase": case.get("phase"),
            "model": case.get("model"),
            "kernel_ids": case.get("kernel_ids"),
            "exact_shape_source": case.get("exact_shape_source"),
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
        del inp
        torch.cuda.empty_cache()
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    {"compile": run_compile, "correctness": run_correctness,
     "performance": run_performance}[mode]()


if __name__ == "__main__":
    main()
