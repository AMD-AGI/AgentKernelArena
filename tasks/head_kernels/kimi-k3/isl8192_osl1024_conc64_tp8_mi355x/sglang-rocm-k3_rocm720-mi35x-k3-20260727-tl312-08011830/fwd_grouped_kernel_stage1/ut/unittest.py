#!/usr/bin/env python3
"""IMMUTABLE op unittest — sglang MLA absorbed-decode split-KV attention stage 1.

Op:      sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd
         (the deepest SAFE python launcher of the profiled GPU symbol `_fwd_grouped_kernel_stage1`)
Kind:    attn (decode only — extend/prefill goes through extend_attention.py:_fwd_kernel)
Judge:   frozen live-captured oracle (reference_io.pt) + random-value parity vs the LIVE baseline
         callable, timed with harness_lib device-event timing in the DEPLOYMENT graph context.

DO NOT EDIT. The task's reference_io.pt + this file are checksummed by the validator.
"""
import importlib
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Load the vendored harness BEFORE dropping HERE from sys.path / importing torch: this file is named
# `unittest.py`, so its directory must not shadow the stdlib `unittest` package once torch imports.
_spec = importlib.util.spec_from_file_location("harness_lib", os.path.join(HERE, "harness_lib.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
sys.path[:] = [p for p in sys.path if p not in ("", ".") and os.path.abspath(p) != HERE]

import torch  # noqa: E402

META = json.load(open(os.path.join(HERE, "meta.json")))
REGIME = META["regime"]
GEO = META["geometry"]
TOL = float(META.get("tol", 2e-2))
DRAWS = int(META.get("random_draws", 3))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH = h.deployment_graph_mode(REGIME)

# positional slots of _decode_grouped_att_m_fwd(q, k_buffer, v_buffer, att_out, att_lse, kv_indptr,
#                                               kv_indices, num_kv_splits, max_kv_splits, ...)
I_Q, I_K, I_V, I_AO, I_AL, I_IPTR, I_IDX, I_NKS = 0, 1, 2, 3, 4, 5, 6, 7



def _load_frozen_capture(torch, path, **kwargs):
    import importlib.util
    from pathlib import Path
    helper = Path(__file__).resolve().with_name("task_contract.py")
    spec = importlib.util.spec_from_file_location("_frozen_capture_loader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verified_torch_load(torch, path, **kwargs)


def _resolve(dotted):
    mod, attr = dotted.split(":")
    return getattr(importlib.import_module(mod), attr)


# --------------------------------------------------------------------------- the seam both legs use
def _make_call(fn):

    def call(args):
        pos = list(args["pos"])
        outs = []
        for i, (shape, dt) in sorted(args["out_slots"].items()):
            # FRESH output every call (never a persistent/static buffer) and ZERO-filled, so the
            # split slots the kernel legitimately leaves untouched (split_id >= num_kv_splits[b])
            # compare equal to the oracle, which was zeroed the same way.
            t = torch.zeros(shape, dtype=dt, device=DEV)
            pos[i] = t
            outs.append(t)
        fn(*pos, **args["kw"])
        return tuple(outs)

    return call


_bindings_spec = importlib.util.spec_from_file_location(
    "_kimi_task_bindings", os.path.join(HERE, "bindings.py"))
_bindings = importlib.util.module_from_spec(_bindings_spec)
_bindings_spec.loader.exec_module(_bindings)
BASELINE_FN, CANDIDATE_FN = _bindings.resolve_pair(HERE)
current_call = _make_call(CANDIDATE_FN)
baseline_call = _make_call(BASELINE_FN)
# `assert_independent_outputs` snapshots the return value as a bare tensor; this seam returns
# (att_out, att_lse), so the probe gets a variant that returns the primary output only. Both
# tensors are allocated by the same `call`, so proving att_out is fresh proves the allocation.
current_call_primary = (lambda c: (lambda args: c(args)[0]))(current_call)


# --------------------------------------------------------------------------- frozen oracle cases
def _hydrate(blob):
    k = blob["k"].to(DEV)
    v = k[:, :, : blob["v_head_dim"]] if blob["v_is_slice"] else blob["v"].to(DEV)
    slots = {"k": k, "v": v, "kv_indices": blob["kv_indices"].to(DEV)}
    pos, out_slots = [], {}
    for i, p in enumerate(blob["pos"]):
        if isinstance(p, dict) and "__slot__" in p:
            nm = p["__slot__"]
            if nm in ("att_out", "att_lse"):
                ref = blob["ref"][nm]
                out_slots[i] = (tuple(ref.shape), ref.dtype)
                pos.append(None)
            else:
                pos.append(slots[nm])
        elif isinstance(p, dict) and p.get("__tensor__"):
            pos.append(p["data"].to(DEV))
        else:
            pos.append(p)
    kw = {kk: (vv["data"].to(DEV) if isinstance(vv, dict) and vv.get("__tensor__") else vv)
          for kk, vv in blob["kw"].items()}
    args = {"pos": pos, "kw": kw, "out_slots": out_slots}
    ref = (blob["ref"]["att_out"].to(DEV), blob["ref"]["att_lse"].to(DEV))
    return args, ref


def eager_cases():
    io = _load_frozen_capture(torch, os.path.join(HERE, "reference_io.pt"), map_location="cpu", weights_only=False)
    out = []
    for blob in io["records"]:
        args, ref = _hydrate(blob)
        out.append({"args": args, "ref": ref, "sig": f"oracle:{blob['sig'][:60]}",
                    "regime": blob.get("regime", "decode")})
    return out


# --------------------------------------------------------------------------- in-regime synthesis
def _num_kv_splits(bs, ctx):
    """Per-sequence split count computed by the LIVE deployment metadata kernel, not a constant.

    `attn_logits` is sized [bs, H, max_kv_splits, Lv] but the stage-1 kernel only runs (and only
    writes) split_id < num_kv_splits[b], which sglang's scheduler shrinks as the batch grows
    (bs=1 -> 256 splits, bs=64 -> ~34 on this device). Hardcoding max_kv_splits would inflate the
    benched work ~8x at the served batch, so we call the same kernel the server calls.
    """
    from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
    from sglang.srt.utils import get_device_core_count
    import triton
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=DEV)
    nks = torch.empty((bs,), dtype=torch.int32, device=DEV)
    get_num_kv_splits_triton[(1,)](
        nks, seq_lens, bs, 1, GEO["num_q_heads"], GEO["splits_num_kv_head"],
        GEO["max_kv_splits"], get_device_core_count(torch.cuda.current_device()),
        MAX_NUM_SEQ=(256 if bs < 256 else triton.next_power_of_2(bs)))
    return nks


def _synth(bs, ctx, rng):
    """FRESH in-regime operands at FIXED online dims (dims never randomize; only values do).

    Regime-driven: the compute/KV dtype comes from h.regime_spec(meta.regime) (kv_cache_dtype=auto ->
    the model compute dtype), never a hardcoded bf16 default. MLA stores ONE latent 'KV' row of
    kv_lora_rank+qk_rope per token (K == V, v = K[..., :kv_lora_rank]) at page_size=1, so the paged
    layout here is the flat slot pool the live server hands the kernel — not the vLLM x-packed
    key_cache layout that h.synth_kv_cache builds.
    """
    spec = h.regime_spec(REGIME)
    dt = h.regime_dtype(spec["kv_dtype"], torch)
    H, Lk, Lv = GEO["num_q_heads"], GEO["head_dim_k"], GEO["v_head_dim"]
    S = GEO["max_kv_splits"]
    need = bs * ctx
    pool = need + 64                                    # padded pool: real servers never pack tightly
    q = (torch.randn(bs, H, Lk, generator=rng, dtype=torch.float32, device=DEV) * 0.1).to(dt)
    kbuf = (torch.randn(pool, 1, Lk, generator=rng, dtype=torch.float32, device=DEV) * 0.1).to(dt)
    vbuf = kbuf[:, :, :Lv]
    # NON-contiguous slot mapping (page_size=1): the live pool is scattered, and a contiguous arange
    # would hide indexing/coalescing bugs a real gather exposes.
    kv_indices = torch.randperm(pool, generator=rng, device=DEV)[:need].to(torch.int64)
    kv_indptr = (torch.arange(bs + 1, device=DEV, dtype=torch.int32) * ctx)
    num_kv_splits = _num_kv_splits(bs, ctx)
    pos = [q, kbuf, vbuf, None, None, kv_indptr, kv_indices, num_kv_splits, S,
           float(GEO["sm_scale"]), float(GEO["logit_cap"]), int(GEO["xai_temperature_len"])]
    out_slots = {I_AO: ((bs, H, S, Lv), torch.float32), I_AL: ((bs, H, S), torch.float32)}
    kw = {"has_mla": bool(GEO["has_mla"]), "use_pdl": False, "page_size": int(GEO["page_size"]),
          "score_mod": None, "aux_tensors": None}
    return {"pos": pos, "kw": kw, "out_slots": out_slots}


def _online_buckets():
    """(sig, bs, ctx, regime) per ONLINE M-bucket. Dims always come from meta.cases (the shape
    contract); meta.workload only decides WHICH cases are timed + tags their regime."""
    by_name = {}
    for c in META.get("cases", []):
        by_name[c.get("sig") or c.get("name")] = c
    wl = (META.get("workload") or {}).get("cases") or []
    out, seen = [], set()
    for w in wl:
        c = by_name.get(w.get("name"))
        if c is None or not c.get("timing", True):
            continue
        nm = c.get("sig") or c.get("name")
        if nm in seen:
            continue
        seen.add(nm)
        out.append((nm, int(c["bs"]), int(c["ctx_per_seq"]), w.get("regime") or c.get("regime") or "decode"))
    if not out:
        for c in META.get("cases", []):
            if not c.get("timing", True):
                continue
            out.append((c.get("sig") or c.get("name"), int(c["bs"]), int(c["ctx_per_seq"]),
                        c.get("regime") or "decode"))
    return out


def random_shapes():
    return [{"sig": sig, "make_inputs": (lambda rng, b=bs, c=ctx: _synth(b, c, rng))}
            for sig, bs, ctx, _ in _online_buckets()]


# --------------------------------------------------------------------------- main
def main():
    print(f"== {META['short_name']} | op_kind={META['op_kind']} | dtype={META['dtype']} "
          f"| served_regimes={META.get('served_regimes')} | graph_deploy={GRAPH}")
    print(f"== target   : {META['target_callable']}")
    print(f"== baseline : {META['baseline_callable']}")

    # Calibration gate: the synthetic buckets are only faithful if our split schedule reproduces the
    # one the LIVE server used at the captured (bs, seq_len). A mismatch means the synthetic timing
    # shape does not match deployment, which is a HARNESS defect, not a kernel failure.
    for cal in META.get("splits_calibration", []):
        got = int(_num_kv_splits(int(cal["bs"]), int(cal["seq_len"]))[0])
        print(f"[splits] bs={cal['bs']} seq={cal['seq_len']} live={cal['num_kv_splits']} ut={got}")
        if got != int(cal["num_kv_splits"]):
            print("UT_HARNESS_INCOMPLETE: split schedule does not reproduce the captured "
                  f"num_kv_splits (bs={cal['bs']} seq={cal['seq_len']} "
                  f"live={cal['num_kv_splits']} ut={got})")
            raise h.HarnessIncompleteError("split schedule mismatch")

    cases = eager_cases()
    # Replay in RECORDED ORDER with every output held live: catches both a shared/persistent return
    # buffer and shape-dependent stale state across the bs=1 -> bs=CONC transition the server makes.
    ok_eager, per_eager = h.check_correct_sequence(current_call, cases, TOL)
    for r in per_eager:
        print(f"[oracle] {r.get('case')}: correct={r.get('correct')} "
              f"max_rel_err={r.get('max_rel_err')} {r.get('note','')}")
    if len(cases) >= 2:
        ok_ind, why = h.assert_independent_outputs(
            current_call_primary, cases[0]["args"], cases[1]["args"])
        ok_eager = ok_eager and ok_ind
        print(f"[oracle] output_independence: correct={ok_ind} {why}")

    ok_rand, per_rand = h.check_random_vs_baseline(
        baseline_call, current_call, random_shapes(), TOL,
        draws=DRAWS, warmup=5, repeats=10, graph=GRAPH)
    for r in per_rand:
        print(f"[random] {r['case']}: correct={r['correct']} max_rel_err={r.get('max_rel_err')} "
              f"speedup={r.get('speedup')}")

    # ---- timing: one bucket per meta.workload case, both legs device-event timed in the deployment
    # graph context (so a candidate cannot win by collapsing host dispatch the live graph already ate)
    per_case = []
    rng0 = torch.Generator(device=DEV).manual_seed(1234)
    for sig, bs, ctx, regime in _online_buckets():
        args = _synth(bs, ctx, rng0)
        b_ms = h.time_op(lambda: baseline_call(args), warmup=10, repeats=30, inner=1, graph=GRAPH)
        c_ms = h.time_op(lambda: current_call(args), warmup=10, repeats=30, inner=1, graph=GRAPH)
        sp = (b_ms / c_ms) if (b_ms and c_ms) else None
        per_case.append({"sig": sig, "regime": regime, "m": bs,
                         "baseline_ms": b_ms, "optimized_ms": c_ms, "speedup": sp})
        print(f"[time] {sig} regime={regime} m={bs} ctx={ctx} baseline_ms={b_ms} "
              f"optimized_ms={c_ms} speedup={None if sp is None else round(sp, 4)}")
        del args
        torch.cuda.empty_cache()

    w = h.serving_weighted_speedup(per_case, META)
    print(f"GEAK_PER_CASE {json.dumps(w['per_case'], default=str)}")
    if w.get("dropped_unserved"):
        print(f"GEAK_DROPPED_UNSERVED {w['dropped_unserved']}")
    if w.get("suspect_identity"):
        print(f"GEAK_SUSPECT_IDENTITY {w['suspect_identity']}")
    if w["weighted"] is None:
        print(f"GEAK_WEIGHTED_SPEEDUP untrusted ({w['reason']})")
    else:
        print(f"GEAK_GEOMEAN_SPEEDUP {w['geomean']:.4f}")
        print(f"GEAK_WEIGHTED_SPEEDUP {w['weighted']:.4f}")
        ceil_pct = h.amdahl_ceiling(META.get("pct_gpu_time", 0.0), w["weighted"])
        print(f"GEAK_AMDAHL_CEILING_PCT {ceil_pct:.3f}")

    ok = bool(ok_eager and ok_rand)
    print(f"RESULT {'PASS' if ok else 'FAIL'} (oracle={ok_eager} random_parity={ok_rand})")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except h.HarnessIncompleteError:
        sys.exit(3)
