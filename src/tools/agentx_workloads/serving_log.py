# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#
# The SGLang scheduler-log and step-annotation dialects below are adapted from
# GEAK (e2e_workflow/knowledge/kernel_extraction_skills/kernel-extract/scripts/
# observed_workload.py and e2e_workflow/scripts/parse_profile.py, revision
# b8c357d908020729cd1b680aec295be6a3c91200), Copyright (c) 2026 Advanced Micro
# Devices, Inc., licensed under the Apache License, Version 2.0. Modifications:
# only the SGLang dialects are kept; records are reduced to the fields the
# census uses; an unparsable log or trace raises instead of exiting.
"""SGLang scheduler log and profiler step annotations -> the observed serving workload."""
from __future__ import annotations

import datetime as dt
import re
import statistics
from collections import Counter

PREFILL_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?)(?: (?:TP|DP)(?P<rank>\d+)[^\]]*)?\] Prefill batch(?: \[\d+\])?[,.] "
    r"#new-seq: (?P<new_seq>\d+), #new-token: (?P<new_token>\d+), "
    r"#cached-token: (?P<cached_token>\d+), "
    # token usage, or one usage per KV pool on hybrid models (full / window_128 / mamba ...)
    r"(?:(?!#running-req: )[^,]+, )*"
    r"#running-req: (?P<running>\d+), #queue-req: (?P<queue>\d+)")
DECODE_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?)(?: (?:TP|DP)(?P<rank>\d+)[^\]]*)?\] Decode batch(?: \[\d+\])?[,.] "
    r"#running-req: (?P<running>\d+), "
    # KV tokens in use: "#token: N", or per pool on hybrid models ("#full token: N, ..., #window_128 token: M")
    r"(?:#(?:full )?token: (?P<token>\d+), )?"
    r"(?:(?!accept len: |cuda graph: |gen throughput )[^,]+, )*"
    r"(?:accept len: (?P<accept_len>[\d.]+), )?"
    r"(?:accept rate: (?P<accept_rate>[\d.]+), )?"
    # newer releases add speculative fields here (cap len, block accept len, estimates)
    r"(?:(?!cuda graph: |gen throughput )[^,]+, )*"
    r"(?:cuda graph: (?P<graph>\w+), )?"
    r"gen throughput \(token/s\): (?P<gen_tps>[\d.]+), #queue-req: (?P<queue>\d+)")
BATCH_HINT_RE = re.compile(r"\]\s*(?:Prefill|Decode) batch(?: \[\d+\])?[,.]")
# SGLang profiler step spans: step[EXTEND bs=<bs> toks=<n>], step[DECODE bs=<bs>], ... With speculative
# decoding a decode round is the draft model's steps (step[DRAFT ...]; older releases: DRAFT_EXTEND*)
# and one target verify step (step[VERIFY ...]; older releases: TARGET_VERIFY). Detailed annotations
# may add fields after the batch, so the label may continue after bs/toks.
STEP_RE = re.compile(r"step\[(?P<mode>[A-Z_0-9]+) bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?[\] ]")
STEP_PHASE = {"EXTEND": "prefill", "DECODE": "decode", "VERIFY": "verify", "TARGET_VERIFY": "verify",
              "DRAFT": "draft", "DRAFT_EXTEND": "draft", "DRAFT_EXTEND_V2": "draft"}
# Phases whose forwards run once per decode round (one logged decode iteration) over the running batch.
ROUND_PHASES = ("decode", "verify", "draft")
# Largest tokens per sequence of a speculative round (draft tokens); a draft GEMM above r x batch with a
# larger r runs over a prefill's new tokens instead (the draft model's extend after a prefill).
MAX_ROUND_RATIO = 32


def parse_timestamp(text: str) -> dt.datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"unparsable timestamp {text!r}")


def parse_step(name: str) -> tuple[str, int, int] | None:
    """(phase, batch, tokens) of an SGLang step annotation, else None.

    phase: 'prefill', 'decode', 'verify' (target verify of speculative decoding) or 'draft' (any step of the
    draft model). tokens: the extend tokens of a prefill step, else the batch.
    """
    m = STEP_RE.search(name)
    if not m or m.group("mode") not in STEP_PHASE:
        return None
    phase = STEP_PHASE[m.group("mode")]
    bs = int(m.group("bs"))
    return phase, bs, int(m.group("toks") or (0 if phase == "prefill" else bs))


def parse_log(lines, rank: int = 0, window: tuple[dt.datetime, dt.datetime] | None = None) -> tuple[list, list]:
    """Prefill and decode records of one rank, optionally inside [start, end] (naive UTC)."""
    prefill, decode, hints, parsed = [], [], 0, 0
    for line in lines:
        if not BATCH_HINT_RE.search(line):
            continue
        hints += 1
        m = PREFILL_RE.match(line)
        kind = "prefill"
        if not m:
            m, kind = DECODE_RE.match(line), "decode"
        if not m:
            continue
        parsed += 1
        if int(m.group("rank") or 0) != rank:  # a TP1 server logs no rank tag
            continue
        ts = parse_timestamp(m.group("ts"))
        if window and not window[0] <= ts <= window[1]:
            continue
        if kind == "prefill":
            prefill.append({"ts": ts, "new_seq": int(m.group("new_seq")), "new_token": int(m.group("new_token")),
                            "cached_token": int(m.group("cached_token")), "running": int(m.group("running"))})
        else:
            decode.append({"ts": ts, "running": int(m.group("running")), "token": int(m.group("token") or 0),
                           "accept_len": float(m.group("accept_len") or 1.0),
                           "cuda_graph": None if m.group("graph") is None else m.group("graph") == "True"})
    if hints and not parsed:
        raise ValueError(f"{hints} scheduler batch lines found but none matched the SGLang dialect; "
                         "an empty workload would look like an idle server")
    return prefill, decode


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return {"p10": pick(0.1), "p50": statistics.median(v), "p90": pick(0.9), "max": v[-1]}


def observed_workload(prefill: list, decode: list, windows: list[tuple[dt.datetime, dt.datetime]]) -> dict:
    """Summary of the scheduler log over the steady-state segments (decode logged every iteration)."""
    new = sum(p["new_token"] for p in prefill)
    cached = sum(p["cached_token"] for p in prefill)
    kv_per_seq = [d["token"] / d["running"] for d in decode if d["running"] and d["token"]]
    return {
        "window": {"start": windows[0][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "end": windows[-1][1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "segments": len(windows),
                   "duration_s": sum((end - start).total_seconds() for start, end in windows)},
        "prefill_batches": len(prefill),
        "decode_iterations": len(decode),
        "prefill_new_tokens": new,
        "prefill_cached_tokens": cached,
        "prefix_cache_hit_pct": round(100.0 * cached / max(1, new + cached), 2),
        "prefill_new_tokens_quantiles": _quantiles([p["new_token"] for p in prefill]),
        "decode_running_hist": dict(sorted(Counter(d["running"] for d in decode).items())),
        "decode_running_quantiles": _quantiles([d["running"] for d in decode]),
        "decode_kv_tokens_per_seq_quantiles": {k: round(v) for k, v in _quantiles(kv_per_seq).items()},
        "speculative": any(d["accept_len"] != 1.0 for d in decode),
        "accept_len_mean": (round(statistics.fmean(d["accept_len"] for d in decode), 4) if decode else None),
    }
