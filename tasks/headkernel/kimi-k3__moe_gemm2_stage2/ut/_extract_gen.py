#!/usr/bin/env python3
"""EXTRACTION-TIME provenance script (run once by the Kernel Extractor; NOT part of the immutable UT).

Turns the LIVE capture (<task>/_capture/routing_capture_<pid>.pt, recorded by a PYTHONPATH overlay
hook on aiter.ops.flydsl.moe_kernels:flydsl_moe_stage2 while a TP=8 sglang server served the real
ISL=8192/OSL=1024/CONC=64 workload) into:

  meta.json        -- geometry, per-case launch variants (prefill: mode=reduce t64x256x128;
                      decode: mode=atomic t32x128x128), case_specs, replay_specs
  reference_io.pt  -- the REAL routing tensors + golden outputs from the FROZEN baseline

Routing provenance (never synthesized):
  * prefill cases keep the captured sorted_token_ids / sorted_expert_ids / num_valid_ids /
    sorted_weights verbatim;
  * decode cases are rebuilt from the SAME served routing: the captured sorted arrays are inverted
    back to (token, slot) -> expert / weight (id = slot<<24 | token_id), the first M served tokens
    are taken, and the REAL production sorter aiter.fused_moe.moe_sorting re-derives the decode
    metadata at the live decode block size. Decode steps run inside a HIP graph, so no eager decode
    call exists to hook; this keeps the served routing distribution instead of fabricating one.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

KW_KEYS = ["tile_m", "tile_n", "tile_k", "a_dtype", "b_dtype", "out_dtype", "mode",
           "sort_block_m", "persist", "waves_per_eu", "use_async_copy", "cu_num_mul",
           "b_nt", "model_dim_pad", "inter_dim_pad", "xcd_swizzle"]


def spec_of(rec, name):
    for s in rec["specs"]:
        if s["name"] == name:
            return s
    return None


def uniform_weights(rec):
    sw = rec["routing"].get("sorted_weights")
    nvi = rec["routing"].get("num_valid_ids")
    if sw is None or nvi is None:
        return True
    v = sw[: int(nvi[0])]
    return bool(float(v.max() - v.min()) < 1e-6)


def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "_capture", "routing_capture_*.pt"))):
        import torch
        blob = torch.load(f, map_location="cpu", weights_only=False)
        for r in blob["records"]:
            r["_file"] = os.path.basename(f)
            recs.append(r)
    return recs


def invert_routing(rec, topk, num_experts):
    """sorted arrays -> dense (topk_ids, topk_weights) for the tokens of THIS captured call."""
    import torch
    r = rec["routing"]
    sti, sei, nvi = r["sorted_token_ids"], r["sorted_expert_ids"], r["num_valid_ids"]
    sw = r["sorted_weights"]
    tok = int(rec["token_num"])
    n_valid = int(nvi[0])
    # sorted_expert_ids has ceil(padded_len / block) entries -> recover the live sort block size
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
    ap.add_argument("--decode-m", type=int, nargs="*", default=[1, 64])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    import aiter
    from aiter.fused_moe import moe_sorting

    recs = load_records()
    served = [r for r in recs if r.get("routing") and not uniform_weights(r)]
    warm = [r for r in recs if r.get("routing") and uniform_weights(r)]
    graph_only = [r for r in recs if not r.get("routing")]
    print(f"[gen] records: {len(recs)} total, {len(served)} served-routing, "
          f"{len(warm)} warmup-routing, {len(graph_only)} graph specs-only")
    for r in served:
        print(f"   served  M={r['token_num']:6d} mode={r['scalars'].get('mode')} "
              f"tiles={r['scalars'].get('tile_m')}x{r['scalars'].get('tile_n')}x{r['scalars'].get('tile_k')} "
              f"t={r.get('t')}")
    sig_seen = {}
    for r in warm:
        k = (r["token_num"], r["scalars"].get("mode"))
        sig_seen[k] = sig_seen.get(k, 0) + 1
    print(f"   warmup shapes: {sorted(sig_seen)[:20]}")
    if args.dry_run:
        return 0
    if not served:
        print("[gen] FATAL: no served-phase routing captured", file=sys.stderr)
        return 1

    # ---- prefill cases: the largest served token_num values, verbatim routing -------------
    served.sort(key=lambda r: (-int(r["token_num"]), r.get("t", 0)))
    prefill, seen_tok = [], set()
    for r in served:
        t = int(r["token_num"])
        if t in seen_tok or t < 1024:
            continue
        seen_tok.add(t)
        prefill.append(r)
        if len(prefill) >= 2:
            break
    base = prefill[0]
    topk = int(base["topk"])
    w2s = spec_of(base, "w2")
    num_experts = int(w2s["shape"][0])
    model_dim = int(w2s["shape"][1])
    inter_dim = int(spec_of(base, "inter_states")["shape"][2])

    # ---- decode cases: REAL served routing, re-sorted by the production sorter ------------
    # The launch VARIANT is per-batch-size (aiter picks the registered kernel name per shape:
    # bs=64 -> mode=atomic t32x256x128, bs=1 -> mode=reduce t32x256x256), so each decode bucket
    # takes the scalars/tensor specs recorded at exactly that token_num.
    by_tok = {}
    for r in recs:
        t = int(r["token_num"])
        cur = by_tok.get(t)
        if cur is None or (not cur.get("routing") and r.get("routing")):
            by_tok[t] = r

    def variant_for(m):
        if m in by_tok:
            return by_tok[m]
        near = min(by_tok, key=lambda t: (abs(t - m), t))
        print(f"[gen] WARNING: no captured launch variant at M={m}; using M={near}")
        return by_tok[near]

    pre_kwargs = {k: base["scalars"].get(k) for k in KW_KEYS}
    print(f"[gen] prefill variant kwargs={pre_kwargs}")

    topk_ids_all, topk_w_all, pre_block = invert_routing(base, topk, num_experts)
    print(f"[gen] served routing inverted from M={base['token_num']} (prefill block={pre_block}); "
          f"weights min/max={float(topk_w_all.min()):.4f}/{float(topk_w_all.max()):.4f}")

    def build_decode_routing(m, block):
        tki = topk_ids_all[:m].contiguous().cuda()
        tkw = topk_w_all[:m].contiguous().cuda()
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _buf = moe_sorting(
            tki, tkw, num_experts, model_dim, torch.bfloat16, block, None, None)
        return {"sorted_token_ids": sorted_ids.cpu(), "sorted_weights": sorted_weights.cpu(),
                "sorted_expert_ids": sorted_expert_ids.cpu(), "num_valid_ids": num_valid_ids.cpu()}

    cases = []

    def add_case(sig, tok, routing, kwargs, regime, rec, seed):
        a = spec_of(rec, "inter_states")
        w2 = spec_of(rec, "w2")
        w2sc = spec_of(rec, "w2_scale")
        a2sc = spec_of(rec, "a2_scale")
        spec = {"sig": sig, "token_num": tok, "seed": seed, "regime": regime,
                "a_shape": [tok, topk, inter_dim], "a_dtype_t": a["dtype"],
                "w2_shape": w2["shape"], "w2_bytes_shape": w2["shape"], "w2_dtype_t": w2["dtype"],
                "w2_scale_shape": w2sc["shape"],
                "a2_scale_shape": (a2sc["shape"] if a2sc and "shape" in a2sc else None),
                "scale_dtype_t": (w2sc["dtype"]),
                "kwargs": {k: v for k, v in kwargs.items() if v is not None or k == "persist"},
                "caller_allocates_out": True}
        cases.append({"spec": spec, "routing": routing})

    for i, r in enumerate(prefill):
        tok = int(r["token_num"])
        add_case(f"prefill_M{tok}", tok, r["routing"], pre_kwargs, "prefill", r, 1000 + i)

    dec_blocks = {}
    for j, m in enumerate(args.decode_m):
        rec = variant_for(m)
        dec_kwargs = {k: rec["scalars"].get(k) for k in KW_KEYS}
        block = int(dec_kwargs.get("sort_block_m") or 0) or int(dec_kwargs.get("tile_m") or 32)
        if rec.get("routing"):
            block = invert_routing(rec, topk, num_experts)[2]
        dec_blocks[m] = block
        print(f"[gen] decode M={m} variant kwargs={dec_kwargs} sort_block={block}")
        rt = build_decode_routing(m, block)
        # a2_scale is a live static workspace; size it to this case's pairs when it is per-pair
        # a2_scale keeps the LIVE static workspace shape (the decode graph's padded pair buffer),
        # so padded lanes cannot read out of bounds -- exactly as in the server.
        add_case(f"decode_M{m}", m, rt, dec_kwargs, "decode", rec, 2000 + j)

    # ---- meta.json ------------------------------------------------------------------------
    with open(os.path.join(HERE, "meta_seed.json")) as fh:
        meta = json.load(fh)
    meta["geometry"] = {"num_experts": num_experts, "topk": topk, "model_dim": model_dim,
                        "inter_dim": inter_dim, "decode_sort_block_m": dec_blocks,
                        "prefill_sort_block_m": pre_block}
    meta["case_specs"] = [c["spec"] for c in cases]
    meta["replay_specs"] = [c["spec"] for c in cases if c["spec"]["regime"] == "decode"]
    meta["call_variants"] = {c["spec"]["sig"]: c["spec"]["kwargs"] for c in cases}
    # `cases` is what attribute_weights.py consumes (op_kind=moe): the enclosing fused_moe operand
    # shapes, tagged with the serving regime the extractor observed them in.
    w1_shape = [num_experts, 768, 1792]
    meta["cases"] = [{"name": c["spec"]["sig"],
                      "dims": [[c["spec"]["token_num"], model_dim], w1_shape,
                               list(c["spec"]["w2_shape"])],
                      "dtypes": ["c10::BFloat16", "c10::Float4_e2m1fn_x2", "c10::Float4_e2m1fn_x2"],
                      "regime": c["spec"]["regime"]}
                     for c in cases]
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
        o = ut._invoke(ut.BASELINE_FN, a, None)        # golden = FROZEN production kernel
        torch.cuda.synchronize()
        print(f"[gen] {s['sig']}: out={tuple(o.shape)} {o.dtype} "
              f"|o|mean={float(o.float().abs().mean()):.5f} nonzero_rows="
              f"{int((o.float().abs().sum(-1) > 0).sum())} fp={fp}")
        out_cases.append({"spec": s, "routing": c["routing"], "fingerprint": fp,
                          "ref": o.detach().to("cpu").clone()})
        del a, o
        torch.cuda.empty_cache()

    torch.save({"schema": "geak-moe-oracle-v1", "cases": out_cases,
                "note": ("routing is REAL (captured live at the seam; decode re-sorted from the same "
                         "served topk by aiter moe_sorting). Heavy value-independent operands are "
                         "rebuilt deterministically by unittest.build_inputs(spec); 'ref' is the "
                         "frozen production kernel's output on them; 'fingerprint' pins the rebuild.")},
               os.path.join(HERE, "reference_io.pt"))
    print("[gen] wrote reference_io.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
