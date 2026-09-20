"""Case builders for the MiniMax-M3 block-sparse GQA PREFILL seam.

Loaded by BOTH legs (leg_runner.py) and by unittest.py. `call` must resolve the target through the
NORMAL import path so the overlay on PYTHONPATH decides which implementation runs — never import
kernel_src/ directly.

Regime note: this seam is reached ONLY from `minimax_sparse_prefill`; decode has its OWN kernel
(`flash_decode_with_gqa_share_sparse`). So every case here is tagged regime="prefill" and there is
deliberately NO decode M-bucket, even though the run-level DECODE_M_BUCKETS hint asks for one — a
self-weighted decode case on a prefill-only kernel is exactly the mis-weighting to avoid.
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
    """The compact per-call online geometry recorded by build_task.py (integers + the real
    req_to_token rows + the real topk_idx). Value tensors are synthesized fresh; only the SHAPE and
    the INDEX/paging structure have to be the live ones."""
    global _GEO
    if _GEO is None:
        import importlib
        helper = importlib.import_module("generated_contract")
        _GEO = helper.load_geometry(_HERE, torch)
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
            "torch.float32": torch.float32}.get(name, torch.bfloat16)


def _build_args(g, rng=None, device="cuda"):
    """Build auxiliary validation kwargs with compact request rows/slot IDs.

    Exact scored prefill uses the full captured record through timing_cases().

    q / k_cache / v_cache carry random values (the kernel's cost is value-independent; correctness
    against the oracle is checked separately on the RECORDED values). Everything that steers control
    flow or memory addressing — cu_seqlens, seq_lens, prefix_lens, topk_idx, req_to_token — is the
    REAL recorded data, so block sparsity, ragged lengths and the paging scatter all match the server.
    """
    qd, kd = _dt(g["q_dtype"]), _dt(g["kv_dtype"])

    def randn(shape, dt):
        t = torch.empty(shape, device=device, dtype=torch.float32)
        t.normal_(generator=rng) if rng is not None else t.normal_()
        return t.to(dt)

    n_seq = g["num_seqs"]
    req_rows = g["req_rows"].to(device)                 # [n_seq, orig_cols], real slot numbers
    slot_ids = torch.arange(n_seq, device=device, dtype=torch.int32)   # remapped to the compact rows
    return {
        "q": randn((g["total_q"], g["num_q_heads"], g["qk_head_dim"]), qd),
        "k_cache": randn((g["max_slots"], g["num_kv_heads"], g["qk_head_dim"]), kd),
        "v_cache": randn((g["max_slots"], g["num_kv_heads"], g["v_head_dim"]), kd),
        "sink": None,
        "req_to_token": req_rows,
        "slot_ids": slot_ids,
        "topk_idx": g["topk_idx"].to(device),
        "block_size_q": g["block_size_q"],
        "block_size_k": g["block_size_k"],
        "cu_seqlens": torch.tensor(g["cu_seqlens"], device=device, dtype=torch.int32),
        "seq_lens": torch.tensor(g["seq_lens"], device=device, dtype=torch.int32),
        "prefix_lens": torch.tensor(g["prefix_lens"], device=device, dtype=torch.int32),
        "max_seqlen_q": g["max_seqlen_q"],
        "sm_scale": g["sm_scale"],
        "cu_seqblocks_q": (None if g["cu_seqblocks_q"] is None else
                           torch.tensor(g["cu_seqblocks_q"], device=device, dtype=torch.int32)),
        "max_seqblock_q": g["max_seqblock_q"],
        "q_scale": g["q_scale"],
        "k_scale": g["k_scale"],
        "v_scale": g["v_scale"],
    }


def _dedup(geos):
    """One timing bucket per distinct online geometry, keyed by the capture signature."""
    seen, out = set(), []
    for g in geos:
        key = (g["sig"], g["total_q"], g["num_seqs"])
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


TIMING_MIN_M = 1024


def timing_buckets(h, meta):
    """Only an observed ISL=8192 call is eligible; no warmup fallback."""
    selected = [g for g in _dedup(_geo()) if int(g["total_q"]) == 8192]
    if len(selected) != 1:
        raise RuntimeError("missing/ambiguous observed M=8192 prefill geometry; no timing fallback")
    return selected


def timing_cases(h, meta):
    """Reconstruct the full captured call, including table extent and slot dtype/IDs."""
    import importlib
    helper = importlib.import_module("generated_contract")
    declarations = meta.get("performance_contract") or {}
    wanted = declarations.get("source_signatures") or []
    if declarations.get("status") != "ready" or len(wanted) != 1:
        raise RuntimeError("exact prefill performance contract is missing")
    geometries = timing_buckets(h, meta)
    if [g["sig"] for g in geometries] != wanted:
        raise RuntimeError("scored geometry differs from the observed prefill control record")
    captured = helper.load_contract(_HERE)["records"]
    matches = [record for record in captured if record["sig"] in wanted]
    if len(matches) != len(wanted):
        raise RuntimeError("observed prefill control record is missing; no timing fallback")
    out = []
    for case_id, record in zip(declarations["case_ids"], matches):
        positional, kwargs = helper.build_record(record, 0, torch, "cuda")
        if positional:
            raise RuntimeError("captured prefill positional ABI requires an explicit adapter")
        out.append({"sig": case_id, "regime": "prefill", "m": 8192,
                    "source_sig": record["sig"], "scenario_evidence": "captured_control_record",
                    "args": kwargs})
    return out


def _ragged_variant(g, mode):
    """A copy of geometry `g` with the SAME buffer shapes but a DIFFERENT ragged split.

    Buffer shapes (total_q, num_seqs, max_slots, req_to_token width) and the host ints that set the
    launch grid (max_seqlen_q / max_seqblock_q) are held fixed, so both variants can be driven through
    ONE set of static buffers and ONE captured graph — which is the whole point of the replay check.
    What changes is exactly what the kernel branches on: per-sequence q_len, prefix_len and seq_len,
    including a split whose seq_lens are NOT multiples of block_size_k (the ragged tail path).
    """
    g = dict(g)
    n, total = g["num_seqs"], g["total_q"]
    cols = int(g["req_rows"].shape[1])
    bk = g["block_size_k"]
    rec_q = [g["cu_seqlens"][i + 1] - g["cu_seqlens"][i] for i in range(n)]
    prefix = [int(x) for x in list(g["prefix_lens"])[:n]] or [0] * n

    if mode == "recorded" or n < 2:
        q_lens = list(rec_q)
    elif mode == "even":
        base, rem = divmod(total, n)
        q_lens = [base + (1 if i < rem else 0) for i in range(n)]
    else:  # "skew": first sequence takes the bulk, the rest straddle the block-size boundary
        small = [bk + 1 + i for i in range(n - 1)]
        q_lens = [total - sum(small)] + small
        if q_lens[0] < 1:                                  # degenerate (tiny total_q) -> even split
            base, rem = divmod(total, n)
            q_lens = [base + (1 if i < rem else 0) for i in range(n)]

    # Prefix (= KV context already in the pool) shift. This is the ONLY boundary lever available when
    # the capture window only ever saw single-sequence chunks (n == 1): total_q must stay fixed for the
    # static buffers, but moving prefix_len moves BOTH the causal mask offset (`off_q_k` adds
    # prefix_len) and seq_len, so the last K block becomes a partial one. Only ever INCREASED: shrinking
    # seq_len below the recorded value would strand recorded topk block indices past the end.
    delta = 0 if mode in ("recorded", "even") else (bk // 2 + 1)
    seq_lens, prefix_lens = [], []
    for i, ql in enumerate(q_lens):
        p = int(prefix[i % len(prefix)]) + delta
        p = max(0, min(p, cols - ql))
        prefix_lens.append(p)
        seq_lens.append(p + ql)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    g["cu_seqlens"] = cu
    g["cu_seqblocks_q"] = cu if g["block_size_q"] == 1 else None
    g["seq_lens"] = seq_lens
    g["prefix_lens"] = prefix_lens
    # max_seqlen_q / max_seqblock_q stay at the RECORDED values: they only size the grid, and an upper
    # bound is safe (surplus q-blocks early-return on `pid_q*num_q_loop >= q_block_len`). Holding them
    # fixed is what lets one captured graph serve both variants.
    return g


def replay_shapes(h, meta):
    """>=2 boundary cases sharing one buffer set, for the CUDA-graph replay gate (regime.cuda_graph)."""
    geos = timing_buckets(h, meta)
    base = max(geos, key=lambda g: (g["total_q"], g["num_seqs"]))
    if base["num_seqs"] < 2:
        # Single-sequence chunk: vary the KV context / causal offset instead of the ragged split.
        variants = [("recorded", _ragged_variant(base, "recorded")),
                    ("shift", _ragged_variant(base, "shift"))]
    else:
        variants = [("even", _ragged_variant(base, "even")),
                    ("skew", _ragged_variant(base, "skew"))]
    return [{"sig": f"replay_{name}_m{g['total_q']}_s{g['num_seqs']}",
             "geo": g,
             "make_inputs": (lambda rng, g=g: _build_args(g, rng=rng))}
            for name, g in variants]


def random_shapes(h, meta):
    """Fixed online dims, fresh random VALUES per draw — the value-parity leg.

    The replay boundary cases are included here on purpose: the baseline leg then records their
    outputs in its OWN process (h.baseline_random_outputs), which gives the graph-replay bundle a
    golden reference produced by the frozen live kernel rather than by the candidate itself.
    """
    out = []
    for g in timing_buckets(h, meta):
        out.append({"sig": f"prefill_m{g['total_q']}_s{g['num_seqs']}",
                    "make_inputs": (lambda rng, g=g: _build_args(g, rng=rng))})
    for s in replay_shapes(h, meta):
        out.append({"sig": s["sig"], "make_inputs": s["make_inputs"]})
    return out


# ---------------------------------------------------------------- recorded oracle (lazy)
def eager_cases(h, meta, device="cuda"):
    """Stream the RECORDED inputs+golden outputs one at a time. Each record of this op is ~2.4 GiB
    (the paged KV pool is an input), so materializing them all would OOM — hence the lazy iterator
    plus h.check_correct_multi_lazy."""
    path = os.path.join(_HERE, "reference_io.pt")
    for rec in h.iter_eager_cases_from_oracle(path, device=device):
        rec["regime"] = "prefill"
        yield rec
