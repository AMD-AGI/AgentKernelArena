# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Build ``kernel_census.json`` from a capture: every GPU kernel's op, shapes, calls and time.

    python3 -m src.tools.agentx_workloads census <out>/<capture-id> [--rank 0]

Scope: device time on one TP rank (``--rank``), summed over the profiler
windows of the *measured* leg. The eager leg is measured when present: every
kernel, decode included, is an ordinary launch linked (``External id``) to the
CPU op that issued it, whose ``Input Dims`` give the shapes. Kernels that a Python
wrapper launches without an op (Triton, Gluon, FlyDSL) are linked the same way
when the capture ran with shape hooks: the wrapper's ``pyhook::<label>`` scope is
a recorded op with the call's input shapes (``shape_hooks.py``). A graph-leg window
is summarized next to it for comparison but is not used for the shares,
because the ROCm profiler records the kernels of only a few HIP-graph replays.

Census rows:

``ops``
    One row per (CPU op, input dims, dtypes, phase): calls, GPU time, kernels.
    Kernels with no linked CPU op are one row per kernel name.
``gemm``
    Dense GEMM launches as (M, N, K, dtype, phase).
``gemm_log_weighted``
    The token-indexed GEMMs rebuilt over the whole steady state from the
    production leg's scheduler log and a per-call cost model fitted to the
    windows (``log_weighting.py``); shapes no window caught still appear.
``attention``
    Attention launches with batch, query tokens and KV tokens where known:
    prefill from the op's ragged-index dims, decode from the scheduler log
    (mean KV tokens per sequence of the logged iteration nearest in time with
    the same batch size).

The scheduler log of every leg is summarized as the observed workload
(prefill batches, decode iterations, batch and KV-length distributions) over the
steady-state window. Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import serving_log
from . import log_weighting as weighting

SCHEMA = "agentx-kernel-census/1"
KV_BUCKET = 512  # decode KV tokens per sequence are rounded to this many tokens
FAMILY_RULES = [  # (regex over op and kernel names, family); first match wins
    (r"cross_device_reduce|all_reduce|allreduce|allgather|all_gather|reduce_scatter|nccl|rccl", "comm"),
    (r"rmsnorm|rms_norm|layernorm|layer_norm", "norm"),
    (r"rotary|rope", "rope"),
    # Gated DeltaNet / Mamba / conv1d of hybrid models (before activation: conv1d kernels may say silu)
    (r"causal_conv1d|conv1d|gated_delta|chunk_gated|gdn|fused_recurrent|selective_scan|mamba|ssm_", "state_space"),
    # before activation: fused MoE kernels such as fmoe_*_silu name their activation
    (r"fmoe|moe_stage|moe_sorting|moe_align|topk_softmax|grouped_topk|fused_moe|moe", "moe"),
    (r"act_and_mul|silu|gelu|swiglu", "activation"),
    (r"store_kvcache|store_cache|kv_indices|reshape_and_cache", "kv_cache"),
    (r"fmha|paged_attention|mha_|attention|flash|pa_fwd|mla_|decode_attention|extend_attention", "attention"),
    (r"Cijk_|gemm|hgemm|aten::mm|aten::addmm|aten::bmm|aten::linear|aten::matmul|wvSplitK|LLMM", "gemm"),
    (r"quant", "quant"),
    (r"sample|topk|top_k|argmax|softmax|logits", "sampling"),
    (r"memcpy|memset", "memory"),
]
# Ops whose first two inputs are [M, K] activations and an [N, K] or [K, N] weight.
GEMM_OP_RE = re.compile(r"gemm|aten::mm$|aten::addmm$|aten::bmm$|aten::linear$|aten::matmul$|tgemm", re.I)
PACKED_DTYPES = ("fp4", "uint8", "unsigned char", "Byte")  # two values per byte along K
HOOK_PREFIX = "pyhook::"  # ops recorded by the capture's shape hooks (shape_hooks.PREFIX)


def family(*names: str) -> str:
    for rx, fam in FAMILY_RULES:
        if any(n and re.search(rx, n, re.I) for n in names):
            return fam
    return "other"


def _mangled_names(symbol: str) -> list[str]:
    """Length-prefixed identifiers of an Itanium-mangled symbol, in order."""
    names, i = [], 0
    while i < len(symbol):
        if symbol[i].isdigit() and (i == 0 or not symbol[i - 1].isdigit()):
            j = i
            while j < len(symbol) and symbol[j].isdigit():
                j += 1
            length = int(symbol[i:j])
            name = symbol[j:j + length]
            if length and len(name) == length and re.fullmatch(r"[A-Za-z_]\w*", name):
                names.append(name)
                i = j + length
                continue
        i += 1
    return names


def short_name(name: str) -> str:
    """Readable kernel name: namespace and longest identifier of a mangled symbol, else its first identifier."""
    n = re.sub(r"^void\s+", "", name)
    n = re.sub(r"\(anonymous namespace\)", "", n).lstrip(": ")
    if n.startswith("_Z"):
        names = _mangled_names(n[2:])
        if names:
            longest = max(names, key=len)
            return (names[0] + "::" + longest if longest != names[0] else longest)[:96]
    m = re.match(r"[\w:]+", n)
    return (m.group(0) if m else n).split("::")[-1][:96]


def load_events(path: Path) -> tuple[list, int | None]:
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as fh:
        doc = json.load(fh)
    events = doc.get("traceEvents", []) if isinstance(doc, dict) else doc
    base = doc.get("baseTimeNanoseconds") if isinstance(doc, dict) else None
    return [e for e in events if isinstance(e, dict)], base


def gemm_shape(op: str, dims: list, types: list) -> tuple[int, int, int, str, int] | None:
    """(M, N, K, dtype, batch) of a GEMM op from its first two input dims, else None.

    The activation is [M, K] (or [batch, M, K]); the weight is [N, K] or [K, N]
    (with the same leading batch). Packed 4-bit inputs hold two values per byte along K.
    """
    if not GEMM_OP_RE.search(op):
        return None
    shapes = [d for d in dims if isinstance(d, list) and d]
    # addmm / baddbmm take the bias first: (bias, activation, weight)
    if re.search(r"addmm|baddbmm", op) and len(shapes) >= 3 and len(shapes[0]) < len(shapes[1]):
        shapes = shapes[1:]
    if len(shapes) < 2 or len(shapes[0]) not in (2, 3) or len(shapes[1]) != len(shapes[0]):
        return None
    a, b = shapes[0], shapes[1]
    batch = 1
    if len(a) == 3:
        if a[0] != b[0]:
            return None
        batch, a, b = a[0], a[1:], b[1:]
    (m, k) = a
    dtype = next((t for t in types if t and t != "Scalar"), "?")
    packed = any(p in dtype for p in PACKED_DTYPES)
    if b[1] == k:
        n = b[0]
    elif b[0] == k:
        n = b[1]
    else:
        return None
    return m, n, k * (2 if packed else 1), dtype, batch


def attention_dims(op: str, dims: list) -> dict:
    """Batch, query tokens and KV tokens recorded in an attention op's dims, where the layout is known."""
    if op == "aiter::mha_batch_prefill" and len(dims) > 5 and dims[3]:
        # q [tokens, heads, dim], k_cache, v_cache, cu_seqlens_q [batch+1], kv_indptr, kv_page_indices [kv tokens]
        return {"q_tokens": dims[0][0], "batch": dims[3][0] - 1,
                "kv_tokens_total": dims[5][0] if dims[5] else None, "heads": dims[0][1], "head_dim": dims[0][2]}
    if (op.endswith("extend_attention_fwd") and len(dims) > 8 and len(dims[0]) == 3 and len(dims[4]) == 3
            and len(dims[6]) == 1 and len(dims[8]) == 1):
        # SGLang's Triton extend attention (a shape hook): q [tokens, heads, dim], k, v, o, token-major k_buffer,
        # v_buffer, qo_indptr [batch+1], kv_indptr, kv_indices [prefix KV tokens]
        return {"q_tokens": dims[0][0], "batch": dims[6][0] - 1, "kv_tokens_total": dims[8][0],
                "heads": dims[0][1], "head_dim": dims[0][2]}
    if dims and isinstance(dims[0], list) and len(dims[0]) == 3:
        return {"q_tokens": dims[0][0], "heads": dims[0][1], "head_dim": dims[0][2]}
    return {}


class DecodeKV:
    """Mean KV tokens per sequence of logged decode iterations, looked up by time and batch."""

    def __init__(self, decode_records: list):
        self.by_bs = defaultdict(list)
        for d in decode_records:
            if d["running"]:
                self.by_bs[d["running"]].append((d["ts"].timestamp(), d["token"] / d["running"]))
        for v in self.by_bs.values():
            v.sort()

    def lookup(self, wall: float | None, bs: int, tolerance: float = 2.0) -> float | None:
        rows = self.by_bs.get(bs)
        if wall is None or not rows:
            return None
        i = bisect.bisect_left(rows, (wall,))
        best = min((rows[j] for j in (i - 1, i) if 0 <= j < len(rows)), key=lambda r: abs(r[0] - wall))
        return best[1] if abs(best[0] - wall) <= tolerance else None


def census_window(path: Path, acc: dict, kv: DecodeKV | None, phases: set | None = None,
                  weights: dict | None = None) -> dict:
    """Add one trace window to ``acc``: kernels of ``phases`` only (all when None), time and calls times
    ``weights[phase]`` (1 when None). Returns the window's own statistics, unweighted."""
    events, base_ns = load_events(path)
    ops = defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op":
            args = e.get("args", {})
            if args.get("External id") is not None:
                ops[args["External id"]].append((e.get("ts", 0), e.get("name", ""), args.get("Input Dims"),
                                                 args.get("Input type")))
    for v in ops.values():
        v.sort(key=lambda r: r[0])
    spans = []
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and "dur" in e:
            step = serving_log.parse_step(str(e.get("name", "")))
            if step:
                spans.append((e["ts"], e["ts"] + e["dur"], *step))
    spans.sort()
    starts = [s[0] for s in spans]
    cpu_decode = sum(1 for e in events if e.get("cat") == "user_annotation"
                     and (serving_log.parse_step(str(e.get("name", ""))) or ("",))[0] == "decode")
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    stats = {"trace": str(path), "kernel_us": 0.0, "linked_us": 0.0, "hooked_us": 0.0,
             "prefill_steps": sum(1 for s in spans if s[2] == "prefill"),
             "decode_steps_recorded": sum(1 for s in spans if s[2] == "decode"), "decode_steps_cpu": cpu_decode}
    steps_with_kernels = set()
    seen_ops = set()  # (row key, External id): one op launches several kernels; op_calls counts ops
    for e in kernels:
        dur = float(e.get("dur", 0) or 0)
        ts = e.get("ts", 0)
        i = bisect.bisect_right(starts, ts) - 1
        step = spans[i] if i >= 0 and ts < spans[i][1] else None
        phase = step[2] if step else "outside_step"
        if phases is not None and phase not in phases:
            continue
        if step:
            steps_with_kernels.add(i)
        link = next(((nm, dims, types) for _, nm, dims, types in ops.get(e.get("args", {}).get("External id"), [])
                     if dims and any(dims)), None)
        kname = e.get("name", "?")
        stats["kernel_us"] += dur
        if link:
            stats["linked_us"] += dur
            if link[0].startswith(HOOK_PREFIX):
                stats["hooked_us"] += dur
        w = weights.get(phase, 1.0) if weights else 1.0
        dur *= w
        acc["total_us"] += dur
        acc["phase_us"][phase] += dur
        if e.get("cat") != "kernel":  # copies issued by a collective (eager all-reduce staging) count as comm
            fam = "comm" if link and family(link[0]) == "comm" else "memory"
        else:
            fam = family(link[0] if link else "", kname)
        acc["family_us"][fam] += dur
        k = acc["kernels"].setdefault(kname, {"family": fam, "calls": 0, "total_us": 0.0, "phase_us": Counter()})
        k["calls"] += w
        k["total_us"] += dur
        k["phase_us"][phase] += dur
        step_size = {}
        if link:
            op, dims, types = link
            dims = [d for d in dims if d != []]
            types = [t for t in (types or []) if t]
            key = (op, json.dumps(dims), json.dumps(types), phase)
            # Attention-like ops of speculative rounds (verify/draft) read the KV cache, whose length the
            # recorded dims do not carry (only pool sizes): add the step's logged KV tokens per sequence.
            if (step and kv and phase in ("verify", "draft")
                    and (fam == "attention" or op.startswith("pyhook::"))):
                wall = None if base_ns is None else base_ns / 1e9 + ts / 1e6
                kv_mean = kv.lookup(wall, step[3])
                if kv_mean is not None:
                    step_size = {"kv_tokens_per_seq": int(round(kv_mean / KV_BUCKET)) * KV_BUCKET}
                    key = key + (step_size["kv_tokens_per_seq"],)
        else:
            # No recorded op (Python-launched Triton/Gluon kernels): the step gives the work size.
            op, dims, types = "", [], []
            if step:
                step_size = ({"step_tokens": step[4]} if phase == "prefill" else {"step_batch": step[3]})
            key = ("", kname, json.dumps(step_size, sort_keys=True), phase)
        row = acc["ops"].setdefault(key, {"op": op, "family": fam, "phase": phase, "dims": dims, "dtypes": types,
                                          "calls": 0, "op_calls": 0, "total_us": 0.0, "kernels": Counter(),
                                          **step_size})
        row["calls"] += w
        xid = e.get("args", {}).get("External id")
        if link and xid is not None:
            if (key, xid) not in seen_ops:
                seen_ops.add((key, xid))
                row["op_calls"] += w
        else:
            row["op_calls"] += w
        row["total_us"] += dur
        row["kernels"][short_name(kname)] += 1
        if not link and fam != "attention":
            continue
        g = gemm_shape(op, dims, types) if link and fam == "gemm" else None
        if g:
            gphase = phase
            # After a prefill the draft model runs once over the same new tokens (draft extend of the prefill):
            # a DRAFT step whose GEMM M is far beyond r x batch belongs to the prefill, not to the decode round.
            if phase == "draft" and step and g[0] > serving_log.MAX_ROUND_RATIO * max(1, step[3]):
                gphase = "draft_prefill"
            gkey = (*g, gphase)
            r = acc["gemm"].setdefault(gkey, {"calls": 0, "total_us": 0.0, "ops": Counter(), "kernels": Counter(),
                                              "token_matched": 0, "ratios": Counter()})
            r["calls"] += w
            # M equals the step's tokens (prefill) or batch (decode): the GEMM runs over every token of the step.
            # Verify and draft steps run r tokens per sequence (r = draft tokens on verify; 1 or more on the
            # draft model's steps): M = r x batch, with r recorded per call.
            if gphase == "draft_prefill":
                r["token_matched"] += w  # runs over the prefill's new tokens, like the target's prefill
            elif step and phase in serving_log.ROUND_PHASES and step[3] > 0 and g[0] % step[3] == 0:
                r["ratios"][g[0] // step[3]] += w
                if phase != "decode" or g[0] == step[3]:
                    r["token_matched"] += w
            elif step and phase == "prefill" and g[0] == step[4]:
                r["token_matched"] += w
            r["total_us"] += dur
            r["ops"][op] += 1
            r["kernels"][short_name(kname)] += 1
        elif fam == "attention":
            # Decode attention is often launched outside a recorded CPU op; the step gives its batch.
            att = attention_dims(op, dims) if link else {}
            if phase == "decode" and step:
                att["batch"] = step[3]
                wall = None if base_ns is None else base_ns / 1e9 + ts / 1e6
                kv_mean = kv.lookup(wall, step[3]) if kv else None
                if kv_mean is not None:
                    att["kv_tokens_per_seq"] = int(round(kv_mean / KV_BUCKET)) * KV_BUCKET
            akey = (op, phase, json.dumps(att, sort_keys=True))
            r = acc["attention"].setdefault(akey, {"op": op, "phase": phase, **att, "calls": 0, "total_us": 0.0,
                                                   "kernels": Counter()})
            r["calls"] += w
            r["total_us"] += dur
            r["kernels"][short_name(kname)] += 1
    for i in steps_with_kernels:
        acc["steps"][spans[i][2]] += 1
        if spans[i][2] == "decode":
            acc["decode_bs_hist"][spans[i][3]] += 1
        elif spans[i][2] == "prefill":
            acc["prefill_tokens_hist"][spans[i][4]] += 1
        else:
            acc["round_bs_hist"][f"{spans[i][2]}:{spans[i][3]}"] += 1
    stats["decode_steps_with_kernels"] = sum(1 for i in steps_with_kernels if spans[i][2] == "decode")
    stats["kernel_ms"] = round(stats.pop("kernel_us") / 1e3, 3)
    stats["linked_share"] = round(stats.pop("linked_us") / max(1e-9, stats["kernel_ms"] * 1e3), 4)
    stats["hooked_share"] = round(stats.pop("hooked_us") / max(1e-9, stats["kernel_ms"] * 1e3), 4)
    if phases is not None:
        stats["phases_used"] = sorted(phases)
    return stats


def new_acc() -> dict:
    return {"total_us": 0.0, "phase_us": Counter(), "family_us": Counter(), "kernels": {}, "ops": {}, "gemm": {},
            "attention": {}, "steps": Counter(), "decode_bs_hist": Counter(), "prefill_tokens_hist": Counter(),
            "round_bs_hist": Counter()}


def leg_traces(leg_dir: Path, rank: int, record: dict | None = None) -> list[Path]:
    """Rank traces of the leg's finalized windows (a window cut by a server crash is left out)."""
    ok = None
    if record and record.get("windows") and all(isinstance(w, dict) and "dir" in w for w in record["windows"]):
        ok = {Path(w["dir"]).name for w in record["windows"] if w.get("status") == "ok"}
    name = re.compile(rf"-TP-{rank}(?:-[A-Z]+-\d+)*\.trace\.json(?:\.gz)?$")  # -TP-<r>[-DP-<d>...]
    return sorted(p for p in leg_dir.glob(f"traces/*/*-TP-{rank}*.trace.json*")
                  if name.search(p.name) and (ok is None or p.parent.name in ok))


def steady_segments(record: dict) -> list[dict]:
    """Steady-state segments of a leg, each with its server log (one per server start)."""
    if record.get("segments"):
        return record["segments"]
    return [dict(record["steady_state"], server_log=record.get("server_log", "logs/server.log"))]


def leg_observed(leg_dir: Path, record: dict, rank: int) -> tuple[dict, list, list]:
    """Observed workload, prefill and decode log records of a leg's steady-state segments."""
    prefill, decode, windows = [], [], []
    for seg in steady_segments(record):
        window = tuple(serving_log.parse_timestamp(seg[k]) for k in ("start", "end"))
        if window[1] <= window[0]:
            continue
        with open(leg_dir / seg["server_log"], errors="replace") as fh:
            p, d = serving_log.parse_log(fh, rank=rank, window=window)
        prefill += p
        decode += d
        windows.append(window)
    if not windows:
        raise SystemExit(f"census: {leg_dir} has no steady-state segment")
    return serving_log.observed_workload(prefill, decode, windows), prefill, decode


def share(us: float, total: float) -> float:
    return round(us / total, 6) if total else 0.0


def finish(acc: dict) -> dict:
    total = acc["total_us"]
    return {
        "total_gpu_ms": round(total / 1e3, 3),
        "phase_ms": {k: round(v / 1e3, 3) for k, v in sorted(acc["phase_us"].items())},
        "family_share": {k: share(v, total) for k, v in sorted(acc["family_us"].items(), key=lambda kv: -kv[1])},
        "steps_with_kernels": dict(acc["steps"]),
        "decode_bs_hist": dict(sorted(acc["decode_bs_hist"].items())),
        "prefill_tokens_hist": dict(sorted(acc["prefill_tokens_hist"].items())),
        "speculative_bs_hist": dict(sorted(acc["round_bs_hist"].items())),
        "kernels": [{"name": n, "short_name": short_name(n), "family": k["family"],
                     "calls": round(k["calls"], 1),
                     "total_us": round(k["total_us"], 2), "share": share(k["total_us"], total),
                     "phase_us": {p: round(v, 2) for p, v in k["phase_us"].items()}}
                    for n, k in sorted(acc["kernels"].items(), key=lambda kv: -kv[1]["total_us"])],
        "ops": [dict(r, calls=round(r["calls"], 1), op_calls=round(r.get("op_calls", r["calls"]), 1),
                     total_us=round(r["total_us"], 2),
                     share=share(r["total_us"], total),
                     kernels=dict(r["kernels"].most_common(4)))
                for r in sorted(acc["ops"].values(), key=lambda r: -r["total_us"])],
        "gemm": [{"M": k[0], "N": k[1], "K": k[2], "dtype": k[3], "B": k[4], "phase": k[5],
                  "calls": round(r["calls"], 1),
                  "total_us": round(r["total_us"], 2), "avg_us": round(r["total_us"] / r["calls"], 3),
                  "share": share(r["total_us"], total), "token_matched": round(r["token_matched"], 1),
                  "ratios": {m: round(c, 1) for m, c in sorted(r["ratios"].items())},
                  "ops": dict(r["ops"]),
                  "kernels": dict(r["kernels"].most_common(3))}
                 for k, r in sorted(acc["gemm"].items(), key=lambda kv: -kv[1]["total_us"])],
        "attention": [dict(r, calls=round(r["calls"], 1), total_us=round(r["total_us"], 2),
                           avg_us=round(r["total_us"] / r["calls"], 3),
                           share=share(r["total_us"], total), kernels=dict(r["kernels"].most_common(3)))
                      for r in sorted(acc["attention"].values(), key=lambda r: -r["total_us"])],
    }


def _leg_dir(capture_dir: Path, name: str, record: dict) -> Path:
    leg_dir = Path(record.get("dir") or name)
    if not leg_dir.is_absolute():
        leg_dir = capture_dir / leg_dir
    if not leg_dir.is_dir():  # a capture moved after it was recorded
        leg_dir = capture_dir / name
    return leg_dir


def _server_info(capture_dir: Path, record: dict, name: str) -> dict:
    path = _leg_dir(capture_dir, name, record) / "server_info.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


# Server settings that change which kernels run and at which shapes (from the server's /server_info).
DEPLOYMENT_KEYS = ("version", "tp_size", "dp_size", "ep_size", "enable_dp_attention", "page_size",
                   "kv_cache_dtype", "chunked_prefill_size", "max_prefill_tokens", "mem_fraction_static",
                   "context_length", "attention_backend", "moe_runner_backend", "quantization",
                   "speculative_algorithm", "speculative_num_steps", "speculative_num_draft_tokens",
                   "speculative_eagle_topk", "speculative_draft_model_path",
                   "max_running_requests", "max_total_num_tokens", "disable_cuda_graph", "cuda_graph_max_bs")


# Deployment settings that name a model by its path on the capture host.
MODEL_PATH_KEYS = ("speculative_draft_model_path",)


def model_ref(value):
    """A model reference without the capture host's directory layout: a Hugging Face cache snapshot
    path becomes ``org/name@revision``, any other absolute path its last component; anything else is
    returned unchanged."""
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    match = re.search(r"models--(.+?)--([^/]+)/snapshots/([^/]+)", value)
    if match:
        return f"{match[1]}/{match[2]}@{match[3]}"
    return Path(value.rstrip("/")).name or value


def deployment(manifest: dict, server_info: dict) -> dict:
    """The effective serving deployment of a capture: what the server reports, else what was requested.
    Model paths are reduced to model references (``model_ref``)."""
    replay = manifest.get("replay") or {}
    out = {"image": manifest.get("image")}
    if server_info:
        out.update({k: server_info.get(k) for k in DEPLOYMENT_KEYS if k in server_info})
    else:
        out.update({"tp_size": manifest.get("tp"), "chunked_prefill_size": manifest.get("chunked_prefill_size"),
                    "context_length": manifest.get("context_length"), "source": "capture arguments"})
    if manifest.get("server_env"):
        out["server_env"] = " ".join(manifest["server_env"])
    out.update({"replay_lanes": replay.get("lanes"), "replay_think_cap_s": replay.get("think_cap_s"),
                "replay_duration_s": replay.get("duration_s"), "replay_warmup_s": replay.get("warmup_s")})
    out.update({k: model_ref(out[k]) for k in MODEL_PATH_KEYS if k in out})
    return {k: v for k, v in out.items() if v is None or type(v) in (str, int, float, bool)}


def graph_batch_sizes(capture_dir: Path, legs: dict) -> list[int]:
    """Decode batch sizes the graph leg captured CUDA graphs for: from the server log line when it lists them,
    else from the server's own config (``server_info.json`` ``cuda_graph_config.decode.bs``; newer SGLang
    no longer logs the list)."""
    record = legs.get("graph") or {}
    if record.get("cuda_graph_bs"):
        return list(record["cuda_graph_bs"])
    if not record:
        return []
    decode = ((_server_info(capture_dir, record, "graph").get("cuda_graph_config") or {}).get("decode") or {})
    if decode.get("backend") in (None, "disabled") or not isinstance(decode.get("bs"), list):
        return []
    return sorted(int(b) for b in decode["bs"])


def build(capture_dir: Path, rank: int = 0, log_weighting: bool = True) -> dict:
    """The census of a capture.

    Per-step device costs come from the profiler windows: decode from the eager leg, prefill from the
    windows of every leg (prefill runs eagerly in both). With ``log_weighting`` each phase's rows are
    scaled from the steps the windows recorded to the steps the production (graph) leg's scheduler log
    counts over its steady state, so a few windows with few prefill steps do not skew the mix.
    """
    manifest = json.loads((capture_dir / "capture.json").read_text())
    legs = manifest.get("legs", {})
    measured = "eager" if legs.get("eager", {}).get("windows") else "graph"
    if not legs.get(measured, {}).get("windows"):
        raise SystemExit(f"census: no profiler windows in {capture_dir}")
    production = "graph" if legs.get("graph", {}).get("steady_state") else measured
    out = {"schema": SCHEMA, "capture_id": manifest["capture_id"], "model": manifest["model"],
           "corpus": manifest["corpus"], "image": manifest["image"], "tp": manifest["tp"], "rank": rank,
           "measured_leg": measured, "production_leg": production,
           "cuda_graph_bs": graph_batch_sizes(capture_dir, legs), "legs": {}}
    sources, observed_steps, production_log, spec_production = [], {}, ([], []), False
    for name, record in legs.items():
        leg_dir = _leg_dir(capture_dir, name, record)
        observed, prefill_log, decode = leg_observed(leg_dir, record, rank)
        kv = DecodeKV(decode)
        traces = leg_traces(leg_dir, rank, record)
        acc = new_acc()
        windows = [census_window(t, acc, kv) for t in traces]
        out["legs"][name] = {"cuda_graph": record.get("cuda_graph"), "server_crashed": record.get("server_crashed"),
                             "crashes": record.get("crashes", []), "restarts": record.get("restarts", 0),
                             "replay_start": record.get("replay_start"), "observed": observed, "windows": windows,
                             "summary": {k: v for k, v in finish(acc).items() if k in (
                                 "total_gpu_ms", "phase_ms", "family_share", "steps_with_kernels", "decode_bs_hist")}}
        if name == production:
            observed_steps = {"prefill": observed["prefill_batches"], "decode": observed["decode_iterations"]}
            spec_production = observed.get("speculative", False)
            production_log = (prefill_log, decode)
        phases = None if name == measured else {"prefill"}
        sources += [(t, kv, phases) for t in traces]
    count = new_acc()
    windows = [census_window(t, count, kv, phases) for t, kv, phases in sources]
    recorded = dict(count["steps"])
    measured_rows = finish(count)["gemm"]
    # A logged decode iteration is one decode round: one DECODE step, or, with speculative decoding, the draft
    # model's steps and one VERIFY step. Verify and draft time is scaled per round, by the recorded verify steps.
    rounds_recorded = recorded.get("verify") or recorded.get("decode", 0)
    round_steps = dict(recorded)
    if recorded.get("verify"):
        round_steps["draft"] = recorded["verify"]
    weights = {}
    if log_weighting:
        rounds = observed_steps.get("decode", 0)
        if recorded.get("prefill"):
            weights["prefill"] = observed_steps.get("prefill", 0) / recorded["prefill"]
        if recorded.get("decode"):
            weights["decode"] = rounds / recorded["decode"]
        if recorded.get("verify"):
            weights["verify"] = weights["draft"] = rounds / recorded["verify"]
        total_recorded = recorded.get("prefill", 0) + rounds_recorded
        weights["outside_step"] = ((observed_steps.get("prefill", 0) + rounds) / total_recorded
                                   if total_recorded else 1.0)
        acc = new_acc()
        for t, kv, phases in sources:
            census_window(t, acc, kv, phases, weights)
    else:
        acc = count
    out.update(finish(acc))
    out["windows"] = windows
    out["steps_recorded"] = recorded
    out["steps_observed"] = observed_steps
    out["speculative"] = bool(recorded.get("verify")) or spec_production
    if out["speculative"] and not recorded.get("verify"):
        out.setdefault("warnings", []).append(
            "the server log shows speculative decoding but the windows recorded no VERIFY step: verify and draft "
            "kernels are counted outside a step")
    out["phase_weights"] = {k: round(v, 4) for k, v in weights.items()}
    if log_weighting:
        chunk = manifest.get("chunked_prefill_size") or _server_info(capture_dir, legs.get(production, {}),
                                                                     production).get("chunked_prefill_size")
        forwards = weighting.logged_forwards(production_log[0], production_log[1], out["cuda_graph_bs"], chunk)
        sizes = {"prefill": set(count["prefill_tokens_hist"]), "decode": set(count["decode_bs_hist"])}
        rows, groups = weighting.weighted_gemm(measured_rows, round_steps, sizes, forwards)
        weighted = {(g["N"], g["K"], g["dtype"], g["B"], ph) for g in groups
                    for ph, info in g["phases"].items() if info["log_weighted"]}
        # Groups or phases that are not token-indexed keep their window-scaled rows.
        rows += [dict(r, window_scaled=True) for r in out["gemm"]
                 if (r["N"], r["K"], r["dtype"], r["B"], r["phase"]) not in weighted]
        out["gemm_log_weighted"] = sorted(rows, key=lambda r: -r["total_us"])
        out["gemm_log_weighting"] = {"method": weighting.METHOD, "chunked_prefill_size": chunk,
                                     "logged_forwards": {p: sum(c.values()) for p, c in forwards.items()},
                                     "groups": groups}
    window = (out["legs"].get(production, {}).get("observed") or {}).get("window") or {}
    out["steady_state_s"] = window.get("duration_s")
    out["deployment"] = deployment(manifest, _server_info(capture_dir, legs.get(production, {}), production))
    if manifest.get("corpus_coverage"):
        out["corpus_coverage"] = manifest["corpus_coverage"]
    out["shape_hooks"] = manifest.get("shape_hooks", [])
    out["scope"] = (f"device time of TP rank {rank}: decode kernels from the {measured} leg's profiler windows, "
                    f"prefill kernels from every leg's windows"
                    + (f", each phase scaled to the {production} leg's logged steady-state step count"
                       if log_weighting else ", unscaled"))
    return out


def markdown(c: dict) -> str:
    kernel_ms = sum(w["kernel_ms"] for w in c["windows"])
    linked = sum(w["linked_share"] * w["kernel_ms"] for w in c["windows"]) / max(1e-9, kernel_ms)
    hooked = sum(w.get("hooked_share", 0) * w["kernel_ms"] for w in c["windows"]) / max(1e-9, kernel_ms)
    L = [f"# Kernel census: {c['model']} ({c['capture_id']})", "",
         f"- Scope: {c['scope']}; {c['total_gpu_ms']:.1f} ms over {len(c['windows'])} windows.",
         f"- Corpus {c['corpus']['id']}@{c['corpus']['revision']}; image `{c['image']}`; TP {c['tp']}.",
         f"- Deployment (effective): {c.get('deployment')}",
         f"- Corpus coverage under the context window: {c.get('corpus_coverage', 'not recorded')}",
         "- Phase ms: " + ", ".join(f"{k} {v:.1f}" for k, v in c["phase_ms"].items()),
         f"- Steps recorded in the windows {c['steps_recorded']}; logged in the steady state {c['steps_observed']}; "
         f"phase weights {c['phase_weights']}",
         "- Share of device time linked to a CPU op with shapes: %.1f%%" % (100 * linked)
         + (", %.1f%% through shape hooks (%d hooked wrappers)" % (100 * hooked, len(c["shape_hooks"]))
            if c.get("shape_hooks") else " (no shape hooks)"),
         "", "## Observed workload (scheduler log, steady state)", ""]
    for name, leg in c["legs"].items():
        o = leg["observed"]
        for crash in leg.get("crashes", []):
            L.append(f"- {name}: server exited at {crash.get('time')}: {crash.get('error')}")
        L.append(f"- {name}: {o['window']['duration_s']:.0f} s in {o['window'].get('segments', 1)} segment(s), "
                 f"{leg.get('restarts', 0)} restart(s), {o['prefill_batches']} prefill batches, "
                 f"{o['decode_iterations']} decode iterations, prefix-cache hit {o['prefix_cache_hit_pct']}%, "
                 f"decode batch {o['decode_running_quantiles']}, "
                 f"KV tokens/seq {o['decode_kv_tokens_per_seq_quantiles']}"
                 + (" (server exited during the replay)" if leg.get("server_crashed") else ""))
    L += ["", "## GPU time by family", "", "| family | share % |", "|---|---|"]
    L += [f"| {k} | {100 * v:.2f} |" for k, v in c["family_share"].items()]
    L += ["", "## Top kernels", "", "| share % | ms | calls | family | kernel |", "|---|---|---|---|---|"]
    L += [f"| {100 * k['share']:.2f} | {k['total_us'] / 1e3:.1f} | {k['calls']} | {k['family']} | `{k['short_name']}` |"
          for k in c["kernels"][:25]]
    L += ["", "## GEMM (M, N, K), top 40", "", "| M | N | K | dtype | phase | share % | calls | avg us |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {g['M']} | {g['N']} | {g['K']} | {g['dtype']} | {g['phase']} | {100 * g['share']:.3f} | {g['calls']} | "
          f"{g['avg_us']:.1f} |" for g in c["gemm"][:40]]
    if c.get("gemm_log_weighted"):
        lw = c["gemm_log_weighting"]
        L += ["", "## GEMM, log-weighted over the steady state, top 40", "",
              f"Method: {lw['method']}. Chunked prefill {lw['chunked_prefill_size']}; logged forwards "
              f"{lw['logged_forwards']}. `x` marks a cost extrapolated beyond the measured M range.", "",
              "| M | N | K | phase | forwards | ms | cost us | |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {g['M']} | {g['N']} | {g['K']} | {g['phase']} | {g.get('forwards', '')} | "
              f"{g['total_us'] / 1e3:.1f} | {g.get('cost_us', g.get('avg_us', 0)):.1f} | "
              f"{'x' if g.get('extrapolated') else ('window' if g.get('window_scaled') else '')} |"
              for g in c["gemm_log_weighted"][:40]]
    L += ["", "## Attention, top 20", "", "| op | phase | batch | q tokens | KV tokens | share % | calls | avg us |",
          "|---|---|---|---|---|---|---|---|"]
    for a in c["attention"][:20]:
        kv = a.get("kv_tokens_total", a.get("kv_tokens_per_seq", ""))
        L.append(f"| {a['op'] or '-'} | {a['phase']} | {a.get('batch', '')} | {a.get('q_tokens', '')} | {kv} | "
                 f"{100 * a['share']:.3f} | {a['calls']} | {a['avg_us']:.1f} |")
    for name, leg in c["legs"].items():
        if "summary" in leg:
            s = leg["summary"]
            L += ["", f"## {name} leg, windows as recorded", "",
                  f"- {s['total_gpu_ms']:.1f} ms recorded; phase ms {s['phase_ms']}; steps with kernels "
                  f"{s['steps_with_kernels']}; CPU-side decode steps "
                  f"{sum(w['decode_steps_cpu'] for w in leg['windows'])}",
                  "- family share: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["family_share"].items())]
    return "\n".join(L) + "\n"


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("capture_dir", help="<out>/<capture-id> written by capture")
    ap.add_argument("--rank", type=int, default=0, help="TP rank whose traces and log lines are used")
    ap.add_argument("--no-log-weighting", action="store_true",
                    help="sum the windows as recorded instead of scaling each phase to the logged step count")
    ap.add_argument("--out", help="census JSON (default <capture_dir>/kernel_census.json; census.md beside it)")


def main(a: argparse.Namespace) -> int:
    capture_dir = Path(a.capture_dir)
    c = build(capture_dir, a.rank, log_weighting=not a.no_log_weighting)
    out = Path(a.out) if a.out else capture_dir / "kernel_census.json"
    out.write_text(json.dumps(c, indent=1) + "\n")
    out.with_name("census.md").write_text(markdown(c))
    print(f"census: {c['total_gpu_ms']:.1f} ms, {len(c['gemm'])} GEMM shapes, {len(c['ops'])} op rows -> {out}")
    return 0
