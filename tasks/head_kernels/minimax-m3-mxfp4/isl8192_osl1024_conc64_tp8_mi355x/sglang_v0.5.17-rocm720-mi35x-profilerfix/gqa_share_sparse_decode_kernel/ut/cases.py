"""Case builders for the MiniMax-M3 sparse-attention DECODE MAIN-attention seam
(`_gqa_share_sparse_decode_kernel` via `flash_decode_with_gqa_share_sparse`).

Loaded by BOTH legs (leg_runner.py) and by unittest.py. `call` must resolve the target through the
NORMAL import path so the overlay on PYTHONPATH decides which implementation runs — never import
kernel_src/ directly.

Regime note: this seam is reached ONLY from `minimax_sparse_decode`; prefill has its OWN kernel
(`flash_prefill_with_gqa_share_sparse`, extracted as a separate task). Every case here is therefore
tagged regime="decode" and served_regimes is ["decode"].

Return-shape note: the seam returns ONE tensor (`o_partial[0].contiguous()`), so `call` hands the
launcher's return value back unchanged.
"""
import importlib
import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEO = None
_META = None



def _load_frozen_capture(torch, path, **kwargs):
    import importlib.util
    from pathlib import Path
    helper = Path(__file__).resolve().with_name("task_contract.py")
    spec = importlib.util.spec_from_file_location("_frozen_capture_loader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verified_torch_load(torch, path, **kwargs)


def _meta():
    global _META
    if _META is None:
        with open(os.path.join(_HERE, "meta.json")) as fh:
            _META = json.load(fh)
    return _META


def _geo():
    """Compact per-call online geometry from build_task.py: integers, the real paging rows for the
    RECORDED calls, and the served decode M buckets (DECODE_M_BUCKETS) at the served context."""
    global _GEO
    if _GEO is None:
        _GEO = _load_frozen_capture(torch, os.path.join(_HERE, "timing_geometry.pt"), map_location="cpu",
                          weights_only=False)
    return _GEO


def _resolve():
    tgt = _meta()["target_callable"]
    mod_name, _, attr = tgt.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def call(args):
    """Invoke the CURRENT implementation of the seam on a kwargs dict (tensors already on device)."""
    return _resolve()(**args)


# ---------------------------------------------------------------- synthetic (timing / random) inputs
def _dt(name):
    return {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
            "torch.float32": torch.float32, "torch.int32": torch.int32,
            "torch.int64": torch.int64}.get(name, torch.bfloat16)


def _paged_rows(g, device):
    """A REAL-shaped paged token map for a synthesized served bucket.

    page_size is 1 on this path, so req_to_token maps token position -> KV slot directly. The server
    allocates in contiguous runs that are scattered across the pool, so the map is built as
    `block_size`-long contiguous runs placed at a SEEDED random permutation of the pool's runs: a
    plain arange would hide exactly the stride/indirection costs this memory-bound kernel is being
    optimized for. Every column of the row is filled (not just up to seq_len), so any boundary
    variant that lengthens a sequence still reads in-bounds slots.
    """
    bs, cols, blk = g["batch_size"], g["max_kv_len"], g["block_size"]
    runs_per_seq = (cols + blk - 1) // blk
    pool_runs = max(g["max_slots"] // blk, bs * runs_per_seq + 16)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    perm = torch.randperm(pool_runs, generator=gen)[: bs * runs_per_seq].to(torch.int32)
    base = (perm.reshape(bs, runs_per_seq) * blk).repeat_interleave(blk, dim=1)[:, :cols]
    off = torch.arange(cols, dtype=torch.int32).remainder(blk).unsqueeze(0)
    return (base + off).clamp_(0, g["max_slots"] - 1).contiguous().to(device)


def _topk_idx(g, seq_lens, device, rng=None):
    """A realistic sparse block selection: [num_kv_heads, batch, topk] int32, front-packed, -1 pad.

    The live index pass (`_decode_score_kernel`, a separate task) always force-selects the first
    (init) and last (local) block of each row and fills the rest with the highest-scoring blocks; the
    resulting index list is sorted-by-score, front-packed, and padded with -1 when a row has fewer
    than topk blocks. Reproduced here EXACTLY in structure — which is all the main kernel branches on
    (`real_topk`, then one gather per selected block) — with the block CHOICE randomized, so the
    scatter pattern over the paged pool is representative instead of a contiguous prefix.
    """
    nkv, bs, topk, blk = g["num_kv_heads"], g["batch_size"], g["topk"], g["block_size"]
    nblocks = torch.div(seq_lens.to(torch.int64) + blk - 1, blk, rounding_mode="floor")
    max_nb = int(max(1, nblocks.max().item()))
    score = torch.rand((nkv, bs, max_nb), device=device, generator=rng)
    ar = torch.arange(max_nb, device=device)
    valid = ar[None, None, :] < nblocks.to(device)[None, :, None]
    score = torch.where(valid, score, torch.full_like(score, float("-inf")))
    # force the init block (0) and the local block (last valid) exactly as the index pass does
    score[:, :, 0] = torch.where(valid[:, :, 0], torch.full_like(score[:, :, 0], 3.0),
                                 score[:, :, 0])
    last = (nblocks - 1).clamp_(min=0).to(device)
    score.scatter_(2, last.view(1, bs, 1).expand(nkv, bs, 1).contiguous(), 2.0)
    k = min(topk, max_nb)
    sel = score.topk(k, dim=-1)
    idx = sel.indices.to(torch.int32)
    idx = torch.where(torch.isfinite(sel.values), idx, torch.full_like(idx, -1))
    if k < topk:                                   # pad short rows to the online topk width
        idx = torch.cat([idx, torch.full((nkv, bs, topk - k), -1, dtype=torch.int32,
                                         device=device)], dim=2)
    return idx.contiguous()


def _build_args(g, rng=None, device="cuda"):
    """Materialize one call's kwargs at EXACTLY the recorded/served online geometry.

    q / k_cache / v_cache carry random VALUES (the kernel's cost is value-independent; correctness
    against the recorded golden is checked separately on the recorded values). Everything that steers
    control flow or memory addressing — batch, per-row context, req_to_token paging, slot_ids,
    block_size, the topk block list — is the live geometry.
    """
    qd, kd = _dt(g["q_dtype"]), _dt(g["kv_dtype"])

    def randn(shape, dt):
        t = torch.empty(shape, device=device, dtype=torch.float32)
        t.normal_(generator=rng) if rng is not None else t.normal_()
        out = t.to(dt)
        del t
        return out

    bs = g["batch_size"]
    if g.get("req_rows") is not None:
        rows = g["req_rows"].to(device)                # REAL recorded paging rows
    else:
        rows = _paged_rows(g, device)
    seq_lens = torch.tensor(g["seq_lens"], device=device, dtype=_dt(g["seq_lens_dtype"]))
    slot_ids = torch.arange(bs, device=device, dtype=_dt(g["slot_ids_dtype"]))
    return {
        "q": randn((bs, g["num_q_heads"], g["head_dim"]), qd),
        "sink": (randn((g["num_q_heads"], g["head_dim"]), qd) if g["has_sink"] else None),
        "k_cache": randn((g["max_slots"], g["num_kv_heads"], g["head_dim"]), kd),
        "v_cache": randn((g["max_slots"], g["num_kv_heads"], g["head_dim"]), kd),
        "req_to_token": rows,
        "seq_lens": seq_lens,
        "slot_ids": slot_ids,
        "block_size": g["block_size"],
        "topk_idx": _topk_idx(g, seq_lens, device, rng=rng),
        "sm_scale": g["sm_scale"],
        "q_scale": g["q_scale"],
        "k_scale": g["k_scale"],
        "v_scale": g["v_scale"],
    }


def timing_buckets(h, meta):
    """The geometries that are allowed to CARRY WEIGHT: the SERVED decode M buckets only.

    The heavy oracle records are bs=1 / short-context calls from sglang's own startup warmup (the
    capture budget — 2.4 GiB per record, because both paged KV pools are inputs — is spent before the
    benchmark's first real decode step). They are real and they stay in the correctness oracle, but
    giving each its own self-weighted timing bucket would let a change that only helps a tiny warmup
    shape dominate the reported speedup.
    """
    served = [g for g in _geo() if g.get("source") == "served_analytic"]
    return served or _geo()


def timing_cases(h, meta):
    """DECODE-only timing buckets, one per DECODE_M_BUCKETS entry. `m` = decode batch size, which is
    what the serving weight model multiplies by the decode call count (OSL on the largest bucket)."""
    return [{"sig": g["sig"], "regime": "decode", "m": g["batch_size"],
             "args": _build_args(g, rng=None)}
            for g in timing_buckets(h, meta)]


def _seq_len_variant(g, mode):
    """A copy of geometry `g` with the SAME buffer shapes and the SAME host scalars but DIFFERENT
    per-sequence context lengths.

    batch_size, every tensor shape and every host scalar are held fixed on purpose: batch_size and
    num_kv_heads size the launch grid and max_topk sizes the split-K chunking, so holding them fixed
    is what lets ONE captured graph serve both variants — which is the whole point of the replay
    gate. What changes is exactly what the kernel branches on per row: seq_lens (hence the number of
    valid blocks, the ragged tail mask `off_n < seq_len - c`, and rows with fewer than topk blocks
    where whole split-K chunks fall through empty).
    """
    g = dict(g)
    bs, ctx, blk, topk = g["batch_size"], g["ctx"], g["block_size"], g["topk"]
    if mode == "uniform":
        g["seq_lens"] = [ctx] * bs
    else:  # "boundary": straddle the block_size boundary, include short/trivial-topk rows
        cands = [ctx, ctx - 1, blk - 1, blk, blk + 1, 1, blk * topk, blk * topk + 1]
        cands = [max(1, min(ctx, c)) for c in cands]
        g["seq_lens"] = [cands[i % len(cands)] for i in range(bs)]
    g["sig"] = f"{g['sig']}_{mode}"
    return g


def replay_shapes(h, meta):
    """>=2 boundary cases sharing one buffer set, for the CUDA-graph replay gate (regime.cuda_graph
    is true: the served decode path runs entirely as graph replay)."""
    base = max(timing_buckets(h, meta), key=lambda g: (g["batch_size"], g["ctx"]))
    return [{"sig": f"replay_{v['sig']}", "geo": v,
             "make_inputs": (lambda rng, v=v: _build_args(v, rng=rng))}
            for v in (_seq_len_variant(base, "uniform"), _seq_len_variant(base, "boundary"))]


def random_shapes(h, meta):
    """Fixed online dims, fresh random VALUES per draw — the value-parity leg.

    The replay boundary cases are included on purpose: the baseline leg then records their outputs in
    its OWN process (h.baseline_random_outputs), giving the graph-replay bundle a golden reference
    produced by the frozen live kernel rather than by the candidate itself.
    """
    out = [{"sig": g["sig"], "make_inputs": (lambda rng, g=g: _build_args(g, rng=rng))}
           for g in timing_buckets(h, meta)]
    out += [{"sig": s["sig"], "make_inputs": s["make_inputs"]} for s in replay_shapes(h, meta)]
    return out


# ---------------------------------------------------------------- recorded oracle (lazy)
def eager_cases(h, meta, device="cuda"):
    """Stream the RECORDED inputs + golden outputs one at a time. Each record is ~2.4 GiB (both paged
    KV pools are inputs), so materializing all of them at once would OOM — hence the lazy iterator
    plus h.check_correct_multi_lazy."""
    path = os.path.join(_HERE, "reference_io.pt")
    for rec in h.iter_eager_cases_from_oracle(path, device=device):
        rec["regime"] = "decode"
        yield rec
