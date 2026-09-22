#!/usr/bin/env python3
"""L2 INDEPENDENT-REFERENCE probe for `flydsl_moe_stage1` (= profile's `moe_gemm1_0`).

WHY THIS EXISTS
---------------
`unittest.py` ships an **L1 self-consistency** oracle: golden = the frozen production kernel's own
(21-launch median) output. That is enough to catch a candidate regressing against what production
does today, but it structurally CANNOT answer:

    is what production does today actually CORRECT?

That question became load-bearing once the run-to-run spread was characterised
(meta.nondeterminism_calibration): the disagreeing elements are NOT rounding -- only 1e-4 of them
sit within one bf16 ulp, ~69% differ by >10%, and the outlier value has the SAME magnitude as the
majority value and is never zero. That is the signature of reading someone else's data, not of FP
reassociation (and with k_batch=1 there are no split-K atomics, so there is no legitimate
reassociation source here at all). The median restores a deterministic value; it does not prove
that value is the mathematically right one.

WHAT IT DOES
------------
Builds a SECOND, independent path to the same number using aiter's OWN torch reference
(`aiter.fused_moe.torch_moe_stage1`) -- the same one `op_tests/flydsl_tests/` gates this family
with -- and compares three things against it:

    single-launch kernel output   vs  torch
    21-launch median              vs  torch
    (and, for the contested elements only, which of the two competing kernel values torch picks)

That last one is the discriminator:
  * if torch consistently matches the MEDIAN and not the outlier  -> the majority value is correct,
    the kernel intermittently corrupts ~1% of elements, the median is a valid repair.
  * if torch matches NEITHER                                      -> the kernel is systematically
    wrong and the shipped L1 oracle is pinned to a wrong function.
  * if torch is ~equidistant from both within mxfp4 dequant noise -> inconclusive at this precision;
    the reference itself is too loose to arbitrate (report as such, do not over-claim).

`torch_moe_stage1` is used VERBATIM, never re-implemented: a hand-rolled reference silently becomes
a different function (packed-fp4 unpack order, e8m0 bias, gate/up split order, SiTUv2 clamp).
It natively supports `activation=ActivationType.Situv2` with `situ_beta` / `situ_linear_beta`, so
this is an EXACT-contract reference for the live 0828 variant, not a Silu stand-in.

PROVENANCE / WHAT IS SHARED WITH THE SHIPPED ORACLE
---------------------------------------------------
  shared : the REAL served routing (sorted_token_ids / sorted_expert_ids / num_valid_ids) read
           straight out of reference_io.pt, plus the live launch kwargs from meta.json.
  NOT shared : the heavy operands. `unittest.build_inputs` writes random bytes DIRECTLY into the
           kernel's pre-shuffled layout, which cannot be un-shuffled to feed a torch reference. So
           this script generates an unshuffled (w1_qt, w1_scale) pair with the SAME byte
           distribution (fp4 uniform 0..255, e8m0 byte 117..121 == ~2^-10, see
           meta.scale_calibration) and applies the production `shuffle_weight((16,16))` /
           `e8m0_shuffle` on the way into the kernel. Conclusions are therefore about the OPERATOR,
           not about the shipped reference_io.pt bytes.

Read-only: writes nothing into the bundle. Prints a report; `--json PATH` to save it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

E, TOPK, MODEL_DIM, INTER_DIM = 896, 16, 3584, 384
BF16_ULP = 2.0 ** -8  # 8 mantissa bits -> 1 ulp is ~0.39% relative. The rounding yardstick.


def invert_routing(routing, tok, topk, block):
    """sorted arrays -> dense (topk_ids, topk_weights). Port of ut_stage2/_extract_gen.py."""
    import torch
    sti, sei, nvi = routing["sorted_token_ids"], routing["sorted_expert_ids"], routing["num_valid_ids"]
    sw = routing["sorted_weights"]
    n_valid = int(nvi[0])
    ids = sti[:n_valid].to(torch.int64)
    slot = ids >> 24
    token = ids & 0xFFFFFF
    keep = (slot < topk) & (token < tok)
    idx = torch.nonzero(keep).flatten()
    expert = sei.to(torch.int64)[(idx // block).clamp(max=sei.numel() - 1)]
    topk_ids = torch.full((tok, topk), -1, dtype=torch.int32)
    topk_w = torch.zeros((tok, topk), dtype=torch.float32)
    topk_ids[token[idx], slot[idx]] = expert.to(torch.int32)
    topk_w[token[idx], slot[idx]] = sw[:n_valid][idx].to(torch.float32)
    return topk_ids, topk_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["decode_M1", "decode_M64", "prefill_M8192"])
    ap.add_argument("--launches", type=int, default=21)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch
    import aiter
    from aiter import dtypes, QuantType, ActivationType
    from aiter.fused_moe import torch_moe_stage1
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    dev = "cuda"
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    kwargs = dict(meta["live_call_kwargs"])
    block = int(meta["geometry"]["sort_block_m"])
    blob = torch.load(os.path.join(HERE, "reference_io.pt"), map_location="cpu", weights_only=False)
    by_sig = {c["spec"]["sig"]: c for c in blob["cases"]}

    # ---- operands: unshuffled pair (torch ref) + shuffled pair (kernel) --------------------
    # Same distributions as unittest.build_inputs; see the module docstring on why they cannot be
    # the same BYTES.
    g = torch.Generator(device=dev).manual_seed(args.seed)
    w1_qt = torch.randint(0, 256, (E, 2 * INTER_DIM, MODEL_DIM // 2), dtype=torch.uint8,
                          device=dev, generator=g).view(torch.float4_e2m1fn_x2)
    w1_scale = torch.randint(117, 121, (E * 2 * INTER_DIM, MODEL_DIM // 32), dtype=torch.uint8,
                             device=dev, generator=g).view(torch.float8_e8m0fnu)
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    # w2 is not consumed at this seam; torch_moe_stage1 only reads its SHAPE (get_inter_dim).
    # It MUST be the PACKED shape [E, model_dim, inter_dim//2] (== the 0828 profile's own operand
    # entry [896, 3584, 192]), not the logical one: get_inter_dim multiplies by
    # int4_war = model_dim // w1.shape[-1] == 2 to undo fp4x2 packing, so handing it the logical
    # 384 yields inter_dim=768, use_g1u1 goes False, and the epilogue silently falls through to
    # `torch_act(out)` where get_torch_act returns the NotImplementedError *class* -- which is
    # callable, so it constructs an exception object and the failure surfaces 20 lines later as
    # "'NotImplementedError' object has no attribute 'to'". Silent wrong-function, exactly the
    # hazard that makes a hand-rolled reference dangerous.
    w2_shape_only = torch.empty((E, MODEL_DIM, INTER_DIM // 2), dtype=torch.uint8, device="meta")
    print(f"[l2] w1 {tuple(w1_qt.shape)} {w1_qt.dtype} -> shuf {tuple(w1_qt_shuf.shape)} | "
          f"w1_scale {tuple(w1_scale.shape)} -> shuf {tuple(w1_scale_shuf.shape)}", flush=True)

    report = {"launches": args.launches, "seed": args.seed, "cases": {}}

    for sig in args.cases:
        if sig not in by_sig:
            print(f"[l2] SKIP {sig}: not in reference_io.pt")
            continue
        spec = by_sig[sig]["spec"]
        m = int(spec["token_num"])
        routing_cpu = by_sig[sig]["routing"]
        routing = {k: v.to(dev) for k, v in routing_cpu.items()}
        topk_ids, topk_w = invert_routing(routing_cpu, m, TOPK, block)
        topk_ids, topk_w = topk_ids.to(dev), topk_w.to(dev)
        if int((topk_ids < 0).sum()) != 0:
            print(f"[l2] FATAL {sig}: routing inversion left {(topk_ids<0).sum()} unset slots")
            return 1

        ga = torch.Generator(device=dev).manual_seed(int(spec["seed"]))
        a = torch.randn((m, MODEL_DIM), dtype=torch.bfloat16, device=dev, generator=ga)

        def launch():
            out = torch.empty((m, TOPK, INTER_DIM), dtype=torch.bfloat16, device=dev)
            r = flydsl_moe_stage1(
                a, w1_qt_shuf, routing["sorted_token_ids"], routing["sorted_expert_ids"],
                routing["num_valid_ids"], out, TOPK,
                w1_scale=w1_scale_shuf, a1_scale=None, sorted_weights=None, **kwargs)
            torch.cuda.synchronize()
            return (r[0] if isinstance(r, (tuple, list)) else r) if r is not None else out

        stack = torch.stack([launch().float() for _ in range(args.launches)])
        med = stack.median(0).values
        single = stack[0]
        # the "other" competing value: the launch that disagrees most with the median
        dev_from_med = (stack - med).abs()
        outlier = torch.gather(stack, 0, dev_from_med.argmax(0, keepdim=True)).squeeze(0)
        del stack, dev_from_med

        ref = torch_moe_stage1(
            a, w1_qt, w2_shape_only, topk_w, topk_ids, dtype=torch.bfloat16,
            activation=ActivationType.Situv2, quant_type=QuantType.per_1x32,
            a1_scale=None, w1_scale=w1_scale, doweight=False,
            situ_beta=float(kwargs["situ_beta"]),
            situ_linear_beta=float(kwargs["situ_linear_beta"])).float()
        torch.cuda.synchronize()

        def stats(x, name):
            d = (x - ref).abs()
            rel = d / ref.abs().clamp_min(1e-6)
            cos = float(torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), ref.reshape(1, -1)).item())
            return {"name": name,
                    "relL2": float((x - ref).norm() / ref.norm().clamp_min(1e-12)),
                    "cosine": cos,
                    "max_abs_delta": float(d.max()),
                    "frac_within_1ulp": float((rel <= BF16_ULP).float().mean()),
                    "pct_isclose_aiter_gate": float(torch.isclose(
                        x, ref, atol=1.0, rtol=0.05).float().mean() * 100)}

        s_med, s_sgl, s_out = stats(med, "median"), stats(single, "single"), stats(outlier, "outlier")

        # ---- the discriminator: on CONTESTED elements only, who does torch side with? -------
        contested = (single != outlier)
        n_c = int(contested.sum())
        disc = {"n_contested": n_c, "frac_contested": n_c / ref.numel()}
        if n_c:
            r_c, m_c, o_c = ref[contested], med[contested], outlier[contested]
            dm, do = (m_c - r_c).abs(), (o_c - r_c).abs()
            disc.update({
                "torch_closer_to_median": float((dm < do).float().mean()),
                "torch_closer_to_outlier": float((do < dm).float().mean()),
                "median_relerr_on_contested_mean": float((dm / r_c.abs().clamp_min(1e-6)).mean()),
                "outlier_relerr_on_contested_mean": float((do / r_c.abs().clamp_min(1e-6)).mean()),
                # baseline for scale: how far is torch from the kernel on UNCONTESTED elements?
                # that is the irreducible mxfp4-dequant/accum-order gap of the reference itself.
                "median_relerr_on_uncontested_mean": float(
                    ((med[~contested] - ref[~contested]).abs()
                     / ref[~contested].abs().clamp_min(1e-6)).mean()),
            })

        report["cases"][sig] = {"M": m, "numel": int(ref.numel()),
                                "median": s_med, "single": s_sgl, "outlier": s_out,
                                "discriminator": disc}
        print(f"\n[l2] === {sig} (M={m}, {ref.numel()} elements) ===", flush=True)
        for s in (s_med, s_sgl, s_out):
            print(f"[l2]  {s['name']:>7} vs torch: relL2 {s['relL2']:.4e}  cos {s['cosine']:.6f}  "
                  f"maxdelta {s['max_abs_delta']:.4f}  within1ulp {s['frac_within_1ulp']:.4f}  "
                  f"aiter-gate {s['pct_isclose_aiter_gate']:.2f}%", flush=True)
        print(f"[l2]  contested {n_c} ({disc['frac_contested']:.4%})", flush=True)
        if n_c:
            print(f"[l2]  on contested: torch closer to MEDIAN {disc['torch_closer_to_median']:.4f} "
                  f"/ to OUTLIER {disc['torch_closer_to_outlier']:.4f}", flush=True)
            print(f"[l2]  mean relerr vs torch -- median {disc['median_relerr_on_contested_mean']:.4f}"
                  f"  outlier {disc['outlier_relerr_on_contested_mean']:.4f}"
                  f"  (uncontested floor {disc['median_relerr_on_uncontested_mean']:.4f})", flush=True)
        del ref, med, single, outlier, contested, a
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[l2] wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
