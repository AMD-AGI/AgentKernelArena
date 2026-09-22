#!/usr/bin/env python3
"""EXTRACTION-TIME provenance script (run once; NOT part of the immutable UT).

Builds the stage-1 (`moe_gemm1_0`) oracle WITHOUT a second live server run, by reusing the REAL
served routing that was captured at the stage-2 seam of the 0817 session
(`../ut_stage2/reference_io.pt`, same model / TP=8 / ISL 8192 / OSL 1024 / conc 64):

  1. invert the captured sorted arrays back to dense (topk_ids, topk_weights) -- the same inversion
     `../ut_stage2/_extract_gen.py:invert_routing()` uses for its own decode cases;
  2. re-sort the first M of those served tokens with the PRODUCTION sorter
     `aiter.fused_moe.moe_sorting` at STAGE-1's own block size (= tile_m of the live variant), because
     stage-1 derives its grid from `sorted_token_ids` / `sorted_expert_ids` at that block;
  3. resolve the LIVE 0828 launch variant from the kernel name recorded in the run
     (`flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2`) through the frozen registry
     `moe_kernels.get_flydsl_kernel_params()`, i.e. exactly what `_flydsl_stage1_wrapper` does;
  4. take the golden output from the FROZEN production kernel on deterministically re-synthesized
     value-independent operands (a / w1 / w1_scale).

Routing is never synthesized (it is the MoE perf signal). The heavy value-INDEPENDENT operands are.

Writes: meta.json, reference_io.pt.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

# The live 0828 stage-1 kernel name (architect report + tuning/work capture of the cycle1 run).
LIVE_KNAME = "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2"

# Model/serving geometry (Kimi-K3 config.json + untuned_fmoe_live.csv, TP=8 shard).
E, TOPK, MODEL_DIM, INTER_DIM = 896, 16, 3584, 384


def invert_routing(routing, tok, topk):
    """sorted arrays -> dense (topk_ids, topk_weights). Verbatim port of ut_stage2/_extract_gen.py."""
    import torch
    sti, sei, nvi = routing["sorted_token_ids"], routing["sorted_expert_ids"], routing["num_valid_ids"]
    sw = routing["sorted_weights"]
    n_valid = int(nvi[0])
    block = min([16, 32, 64, 128, 256],
                key=lambda b: abs(-(-sti.numel() // b) - sei.numel()))
    ids = sti[:n_valid].to(torch.int64)
    slot = (ids >> 24)
    token = ids & 0xFFFFFF
    keep = (slot < topk) & (token < tok)
    idx = torch.nonzero(keep).flatten()
    expert = sei.to(torch.int64)[(idx // block).clamp(max=sei.numel() - 1)]
    topk_ids = torch.full((tok, topk), -1, dtype=torch.int32)
    topk_w = torch.zeros((tok, topk), dtype=torch.float32)
    topk_ids[token[idx], slot[idx]] = expert.to(torch.int32)
    topk_w[token[idx], slot[idx]] = sw[:n_valid][idx].to(torch.float32)
    return topk_ids, topk_w, block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-oracle", default=os.path.join(HERE, "..", "ut_stage2", "reference_io.pt"))
    ap.add_argument("--prefill-m", type=int, default=8192)
    ap.add_argument("--decode-m", type=int, nargs="*", default=[1, 64])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl import moe_kernels as MK

    # ---- launch variant -------------------------------------------------------------------
    params = MK.get_flydsl_kernel_params(LIVE_KNAME)
    if params is None:
        print(f"[gen] FATAL: {LIVE_KNAME} not resolvable by the frozen registry", file=sys.stderr)
        return 1
    in_registry = LIVE_KNAME in MK.get_flydsl_stage1_kernels(
        params["a_dtype"], params["b_dtype"], params["out_dtype"])
    tile_m = int(params["tile_m"])
    tile_n_effective = int(MK.resolve_flydsl_stage1_tile_n(INTER_DIM, int(params["tile_n"])))
    print(f"[gen] variant {LIVE_KNAME}: {params} in_registry={in_registry} "
          f"tile_n {params['tile_n']} -> effective {tile_n_effective}")

    # These are the keywords `aiter.fused_moe:_flydsl_stage1_wrapper` (fused_moe.py:1169) passes
    # through verbatim; a candidate MUST accept the same set.
    kwargs = {
        "tile_m": tile_m, "tile_n": int(params["tile_n"]), "tile_k": int(params["tile_k"]),
        "a_dtype": params["a_dtype"], "b_dtype": params["b_dtype"], "out_dtype": params["out_dtype"],
        # activation = ActivationType.Situv2 (untuned_fmoe_live.csv); beta/linear_beta default to 1.0
        # in fused_moe_2stages when the model config leaves them unset.
        "act": "situv2", "situ_beta": 1.0, "situ_linear_beta": 1.0,
        "use_async_copy": True,
        "k_batch": int(params.get("k_batch", 1)),
        "waves_per_eu": int(params.get("waves_per_eu", 3)),
        "b_nt": int(params.get("b_nt", 2)),
        "gate_mode": params.get("gate_mode", "separated"),
        "inter_dim_pad": 0, "model_dim_pad": 0,
        "a_scale_one": bool(params.get("a_scale_one", False)),
        "xcd_swizzle": int(params.get("xcd_swizzle", 0)),
        "swiglu_limit": None,
        "k_wave": int(params.get("k_wave", 1)),
    }

    # ---- real routing ---------------------------------------------------------------------
    blob = torch.load(os.path.abspath(args.stage2_oracle), map_location="cpu", weights_only=False)
    src = max((c for c in blob["cases"] if c["routing"].get("sorted_weights") is not None),
              key=lambda c: int(c["spec"]["token_num"]))
    src_tok = int(src["spec"]["token_num"])
    topk_ids_all, topk_w_all, src_block = invert_routing(src["routing"], src_tok, TOPK)
    hit = int(torch.unique(topk_ids_all[topk_ids_all >= 0]).numel())
    print(f"[gen] served routing inverted from stage-2 case {src['spec']['sig']} "
          f"(M={src_tok}, sort block={src_block}); distinct experts hit={hit}/{E}; "
          f"weights min/max={float(topk_w_all.min()):.4f}/{float(topk_w_all.max()):.4f}")
    if args.dry_run:
        return 0
    if src_tok < args.prefill_m:
        print(f"[gen] FATAL: only {src_tok} served tokens, need {args.prefill_m}", file=sys.stderr)
        return 1

    def build_routing(m):
        tki = topk_ids_all[:m].contiguous().cuda()
        tkw = topk_w_all[:m].contiguous().cuda()
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _buf = moe_sorting(
            tki, tkw, E, MODEL_DIM, torch.bfloat16, tile_m, None, None)
        return {"sorted_token_ids": sorted_ids.cpu(), "sorted_weights": sorted_weights.cpu(),
                "sorted_expert_ids": sorted_expert_ids.cpu(), "num_valid_ids": num_valid_ids.cpu()}

    # ---- case specs -----------------------------------------------------------------------
    # Shapes are the 0828 profile's own operand list for this op:
    #   [[M, 3584], [896, 768, 1792], [896, 3584, 192], [M, 16], [M, 16], [688128, 112], [3211264, 16]]
    # -> a [M, model_dim] bf16 ; w1 [E, 2*inter_dim, model_dim//2] fp4x2 ; w1_scale [E*2*inter_dim,
    #    model_dim//32] e8m0 (the [688128, 112] entry). w2/w2_scale belong to stage-2, not this seam.
    def spec_for(sig, m, regime, seed):
        return {"sig": sig, "token_num": m, "seed": seed, "regime": regime,
                "a_shape": [m, MODEL_DIM], "a_dtype_t": "torch.bfloat16",
                "w1_shape": [E, 2 * INTER_DIM, MODEL_DIM // 2], "w1_dtype_t": "torch.float4_e2m1fn_x2",
                "w1_scale_shape": [E * 2 * INTER_DIM, MODEL_DIM // 32],
                "scale_dtype_t": "torch.float8_e8m0fnu",
                # e8m0 exponent byte range for the synthetic w1_scale. 117 == 2^-10, NOT the ~2^0
                # that ut_stage2 uses: a real mxfp4 w1 entry has std ~1/sqrt(model_dim), and with
                # K=3584 anything near 2^0 pins the SiTUv2 epilogue at its +-1 clamp for ~half the
                # output (sat 0.50 at byte 127, 0.15 at 120, 0 at 117). See
                # meta.scale_calibration.
                "scale_byte_lo": 117, "scale_byte_hi": 121,
                # doweight_stage1=0 in untuned_fmoe_live.csv -> fused_moe_2stages passes
                # sorted_weights=None to stage-1 (the topk weights are applied in stage-2).
                "pass_sorted_weights": False,
                # a16w4: q_dtype_a is bf16, so no activation quantization and a1_scale is None.
                "pass_a1_scale": False,
                "kwargs": dict(kwargs), "caller_allocates_out": True}

    cases = []
    sp = spec_for(f"prefill_M{args.prefill_m}", args.prefill_m, "prefill", 1000)
    cases.append({"spec": sp, "routing": build_routing(args.prefill_m)})
    for j, m in enumerate(sorted(args.decode_m)):
        sp = spec_for(f"decode_M{m}", m, "decode", 2000 + j)
        cases.append({"spec": sp, "routing": build_routing(m)})
    for c in cases:
        r = c["routing"]
        print(f"[gen] {c['spec']['sig']}: sorted_token_ids={tuple(r['sorted_token_ids'].shape)} "
              f"sorted_expert_ids={tuple(r['sorted_expert_ids'].shape)} "
              f"num_valid={int(r['num_valid_ids'][0])} block={tile_m}")

    # ---- meta.json ------------------------------------------------------------------------
    with open(os.path.join(HERE, "meta_seed.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(HERE, "workload.json")) as fh:
        meta["workload"] = json.load(fh)
    meta["workload_note"] = (
        "workload.json verbatim. The `weight` fields are informational; the primary metric is "
        "recomputed by harness_lib.serving_weighted_speedup from baseline_ms x analytic calls "
        "(prefill = CONC*ceil(ISL/chunk) = 64, decode = OSL = 1024).")
    meta["geometry"] = {"num_experts": E, "topk": TOPK, "model_dim": MODEL_DIM,
                        "inter_dim": INTER_DIM, "sort_block_m": tile_m,
                        "experts_hit_in_served_routing": hit}
    meta["live_kernel_name"] = LIVE_KNAME
    meta["live_kernel_params"] = params
    meta["live_kernel_in_registry"] = in_registry
    meta["live_call_kwargs"] = kwargs
    meta["tile_n_effective"] = tile_n_effective
    meta["tile_n_note"] = (
        f"resolve_flydsl_stage1_tile_n(inter_dim={INTER_DIM}, tile_n={params['tile_n']}) "
        f"-> {tile_n_effective}. The live variant's tile_n=64 divides 384 and is kept. The architect "
        "report's field warning is about tile_n=256, which does NOT divide 384 and is silently "
        "downgraded to 128 -- so a tuned config naming 256 here actually runs 128. That is a "
        "PERFORMANCE lever, not a correctness one: measured, tile_n=256 gives cos 0.99970 at M=8192 "
        "and a bitwise-identical result at M=1, i.e. tiling is mathematically neutral and it PASSES "
        "this UT (correctly). Judge that lever in the timing legs. The correctness negative controls "
        "that do bite are act='silu' and situ_beta/situ_linear_beta=0.5 -- see "
        "meta.nondeterminism_calibration.discriminating_power_at_tol_0.02_via_median.")
    meta["case_specs"] = [c["spec"] for c in cases]
    meta["replay_specs"] = [c["spec"] for c in cases if c["spec"]["regime"] == "decode"]
    meta["call_variants"] = {c["spec"]["sig"]: c["spec"]["kwargs"] for c in cases}
    meta["routing_provenance"] = {
        "source": os.path.relpath(os.path.abspath(args.stage2_oracle), HERE),
        "source_case": src["spec"]["sig"], "source_token_num": src_tok,
        "source_sort_block_m": src_block, "restored_sort_block_m": tile_m,
        "distinct_experts_hit": hit,
        "note": ("REAL served topk distribution, captured at the stage-2 seam of the 0817 session and "
                 "re-sorted by the production aiter moe_sorting at stage-1's block size. NOT captured "
                 "at the stage-1 seam -- see _capture_overlay1/ for the first-hand hook.")}
    with open(os.path.join(HERE, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- routing-only oracle so the UT module can import, then fill in the golden refs -----
    torch.save({"schema": "geak-moe-oracle-v1",
                "cases": [{"spec": c["spec"], "routing": c["routing"], "fingerprint": None,
                           "ref": torch.zeros(1)} for c in cases]},
               os.path.join(HERE, "reference_io.pt"))

    spec = importlib.util.spec_from_file_location("ut_mod", os.path.join(HERE, "unittest.py"))
    ut = importlib.util.module_from_spec(spec)
    sys.modules["ut_mod"] = ut
    spec.loader.exec_module(ut)

    out_cases = []
    for c in cases:
        s = c["spec"]
        a = ut.build_inputs(s)
        fp = ut.fingerprint(a)
        a2 = ut.build_inputs(s)
        assert fp == ut.fingerprint(a2), f"non-deterministic rebuild for {s['sig']}"
        del a2
        # golden = FROZEN production kernel, medianed over MEDIAN_LAUNCHES independent launches
        # (a single launch is not reproducible -- see the note atop unittest.py)
        o = ut._invoke_median(ut.BASELINE_FN, a)
        torch.cuda.synchronize()
        # self-agreement of the MEDIAN: this is the property the oracle actually relies on, so it
        # is verified here rather than assumed. Anything but True means MEDIAN_LAUNCHES is too low.
        o2 = ut._invoke_median(ut.BASELINE_FN, ut.build_inputs(s))
        torch.cuda.synchronize()
        bitwise = bool(torch.equal(o, o2))
        single = ut._invoke(ut.BASELINE_FN, a, None)
        torch.cuda.synchronize()
        single_rl2 = float((single.float() - o.float()).norm() / o.float().norm().clamp_min(1e-12))
        del o2, single
        print(f"[gen] {s['sig']}: out={tuple(o.shape)} {o.dtype} "
              f"|o|mean={float(o.float().abs().mean()):.5f} absmax={float(o.float().abs().max()):.4f} "
              f"nonzero_pairs={int((o.float().abs().sum(-1) > 0).sum())} "
              f"median_self_bitwise={bitwise} single_launch_relL2={single_rl2:.3e} fp={fp}")
        if not bitwise:
            print(f"[gen] FATAL: median-of-{ut.MEDIAN_LAUNCHES} is NOT reproducible for {s['sig']}; "
                  "raise median_launches in meta_seed.json and regenerate", file=sys.stderr)
            return 1
        out_cases.append({"spec": s, "routing": c["routing"], "fingerprint": fp,
                          "ref": o.detach().to("cpu").clone(), "median_self_bitwise": bitwise,
                          "single_launch_relL2": single_rl2})
        del a, o
        torch.cuda.empty_cache()

    torch.save({"schema": "geak-moe-oracle-v1", "cases": out_cases,
                "note": ("routing is REAL (served topk captured at the stage-2 seam of the 0817 session, "
                         "re-sorted by the production aiter moe_sorting at stage-1's block size). Heavy "
                         "value-independent operands are rebuilt deterministically by "
                         "unittest.build_inputs(spec); 'ref' is the frozen production kernel's output on "
                         "them; 'fingerprint' pins the rebuild.")},
               os.path.join(HERE, "reference_io.pt"))

    # pin the oracle blob in meta (same fields ut_stage2/meta.json carries)
    import hashlib
    p = os.path.join(HERE, "reference_io.pt")
    hsh = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hsh.update(chunk)
    meta["reference_io_bytes"] = os.path.getsize(p)
    meta["reference_io_sha256"] = hsh.hexdigest()
    meta["median_self_bitwise"] = {c["spec"]["sig"]: c["median_self_bitwise"] for c in out_cases}
    meta["single_launch_relL2"] = {c["spec"]["sig"]: round(c["single_launch_relL2"], 6)
                                   for c in out_cases}
    with open(os.path.join(HERE, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[gen] wrote meta.json + reference_io.pt "
          f"({meta['reference_io_bytes'] / 1e6:.1f} MB, sha256 {meta['reference_io_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
