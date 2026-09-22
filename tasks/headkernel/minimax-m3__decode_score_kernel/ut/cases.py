"""Case builders for the MiniMax-M3 sparse-attention DECODE index-score seam.

Loaded by BOTH legs (leg_runner.py) and by unittest.py. `call` must resolve the target through the
NORMAL import path so the overlay on PYTHONPATH decides which implementation runs — never import
kernel_src/ directly.

Regime note: this seam is reached ONLY from `minimax_sparse_decode`; prefill has its OWN kernel
(`flash_prefill_with_gqa_share_sparse`, extracted as a separate task). Every case here is therefore
tagged regime="decode" and served_regimes is ["decode"].

Return-shape note: with disable_index_value=True and use_dense_main_attn=False (the served config on
every sparse layer of this checkpoint) the seam returns `(None, topk_idx, None)` — exactly ONE
tensor. `call` unwraps that to the tensor, because the harness's shared-output-buffer check calls
`.detach()` on what `call` returns. The unwrap is a pure projection: if a candidate ever returns
more than one live tensor, the tuple is passed through unchanged and the oracle compare fails loudly
instead of silently checking only the first component.
"""
import ast
import importlib
import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEO = None
_META = None


def _meta():
    global _META
    if _META is None:
        with open(os.path.join(_HERE, "meta.json")) as fh:
            _META = json.load(fh)
    return _META


def _geo():
    """Compact per-call online geometry from build_task.py: integers, the real paging rows for the
    RECORDED calls, and the served decode M buckets taken from the capture's shape histogram."""
    global _GEO
    if _GEO is None:
        _GEO = torch.load(os.path.join(_HERE, "timing_geometry.pt"), map_location="cpu",
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
    out = _resolve()(**args)
    if isinstance(out, (tuple, list)):
        live = [t for t in out if t is not None]
        if len(live) == 1:
            return live[0]
    return out


# ---------------------------------------------------------------- synthetic (timing / random) inputs
def _dt(name):
    return {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
            "torch.float32": torch.float32, "torch.int32": torch.int32,
            "torch.int64": torch.int64}.get(name, torch.bfloat16)


def _paged_rows(g, device):
    """A REAL-shaped paged token map for a synthesized served bucket.

    page_size is 1 here, so req_to_token maps token position -> KV slot directly. The server
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


def _build_args(g, rng=None, device="cuda"):
    """Materialize one call's kwargs at EXACTLY the recorded/served online geometry.

    q and k_cache carry random VALUES (the kernel's cost is value-independent; correctness against
    the recorded golden is checked separately on the recorded values). Everything that steers control
    flow or memory addressing — batch, context, seq_lens, req_to_token, slot_ids, block_size, topk,
    init/local blocks — is the live geometry.
    """
    qd, kd = _dt(g["q_dtype"]), _dt(g["kv_dtype"])

    def randn(shape, dt):
        t = torch.empty(shape, device=device, dtype=torch.float32)
        t.normal_(generator=rng) if rng is not None else t.normal_()
        return t.to(dt)

    bs = g["batch_size"]
    if g.get("req_rows") is not None:
        rows = g["req_rows"].to(device)                # REAL recorded paging rows
    else:
        rows = _paged_rows(g, device)
    slot_ids = torch.arange(bs, device=device, dtype=_dt(g["slot_ids_dtype"]))
    return {
        "q": randn((bs, g["num_q_heads"], g["head_dim"]), qd),
        "sink": None,
        "k_cache": randn((g["max_slots"], g["num_kv_heads"], g["head_dim"]), kd),
        "v_cache": None,
        "req_to_token": rows,
        "seq_lens": torch.tensor(g["seq_lens"], device=device, dtype=_dt(g["seq_lens_dtype"])),
        "max_seqlen": g["max_seqlen"],
        "slot_ids": slot_ids,
        "block_size": g["block_size"],
        "topk": g["topk"],
        "init_blocks": g["init_blocks"],
        "local_blocks": g["local_blocks"],
        "sm_scale": g["sm_scale"],
        "score_type": g["score_type"],
        "disable_index_value": g["disable_index_value"],
        "use_dense_main_attn": g["use_dense_main_attn"],
        "page_size": g["page_size"],
        "q_scale": g["q_scale"],
        "k_scale": g["k_scale"],
        "v_scale": g["v_scale"],
    }


def timing_buckets(h, meta):
    """The geometries that are allowed to CARRY WEIGHT: the SERVED decode M buckets only.

    The five heavy oracle records are bs=1 / ctx~190 calls from sglang's own startup warmup (the
    capture budget is spent before the benchmark's first real decode step). They are real and they
    stay in the correctness oracle, but giving each its own self-weighted timing bucket would let a
    change that only helps a 190-token toy dominate the reported speedup.
    """
    served = [g for g in _geo() if g.get("source") == "served_histogram"]
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

    batch_size, max_seqlen and every tensor shape are held fixed on purpose: `max_seqlen` sizes the
    internally allocated `score` tensor AND (with batch_size) the launch grid, so holding both fixed
    is what lets ONE captured graph serve both variants — which is the whole point of the replay
    gate. What changes is exactly what the kernel branches on per row: seq_lens, i.e. the number of
    valid 128-token blocks each (head, batch) row scans, including lengths that are NOT multiples of
    block_size (the ragged tail) and rows with fewer than topk blocks (the trivial-topk path).
    """
    g = dict(g)
    bs, ctx, blk = g["batch_size"], g["max_seqlen"], g["block_size"]
    if mode == "uniform":
        g["seq_lens"] = [ctx] * bs
    else:  # "boundary": straddle the block_size boundary, include the short/trivial-topk rows
        cands = [ctx, ctx - 1, blk - 1, blk, blk + 1, 1, blk * g["topk"], blk * g["topk"] + 1]
        cands = [max(1, min(ctx, c)) for c in cands]
        g["seq_lens"] = [cands[i % len(cands)] for i in range(bs)]
    g["sig"] = f"{g['sig']}_{mode}"
    return g


def replay_shapes(h, meta):
    """>=2 boundary cases sharing one buffer set, for the CUDA-graph replay gate (regime.cuda_graph
    is true: the served decode path runs entirely as graph replay)."""
    base = max(timing_buckets(h, meta), key=lambda g: (g["batch_size"], g["max_seqlen"]))
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
def _unrepr(x):
    """Undo capture_shapes' `{"__repr__": repr(x)}` fallback for plain strings.

    capture_shapes passes int/float/bool/None through but summarizes anything else by repr, so the
    recorded `score_type='max'` reloads as the 5-character string "'max'" and the seam's
    `assert score_type in ('max','lse')` fires. This is a serialization artifact of the capture, not
    kernel data: restoring the literal is what makes the replayed call identical to the live one.
    Anything that is not a quoted literal is left untouched."""
    if isinstance(x, str) and len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'":
        try:
            v = ast.literal_eval(x)
            if isinstance(v, str):
                return v
        except (ValueError, SyntaxError):
            pass
    return x


def eager_cases(h, meta, device="cuda"):
    """Stream the RECORDED inputs + golden outputs one at a time. Each record is ~1.3 GiB (the paged
    index KV pool is an input), so materializing all of them would OOM — hence the lazy iterator plus
    h.check_correct_multi_lazy."""
    path = os.path.join(_HERE, "reference_io.pt")
    for rec in h.iter_eager_cases_from_oracle(path, device=device):
        rec["args"] = {k: _unrepr(v) for k, v in rec["args"].items()}
        rec["regime"] = "decode"
        yield rec
