# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""GEMM time over a whole steady state: logged batches times a measured per-call cost by M.

Profiler windows are short. A shape that the scheduler log shows hundreds of
times (a 16384-token prefill chunk) can be missing from every window, and the
windows' prefill/decode mix is a small sample. So, for every token-indexed
GEMM (N, K), the census rebuilds its time from the production leg's log:

* prefill: every logged prefill batch is one forward with M = its new tokens
  (split into chunks of the server's chunked-prefill size, should a logged
  batch exceed it);
* decode: every logged decode iteration is one forward with M = its running
  batch, padded to the captured CUDA-graph batch size;
* the draft model's extend after a prefill (speculative decoding) runs over
  the same new tokens: once per prefill forward, like the target's prefill;
* speculative decoding: a logged decode iteration is one round, the draft
  model's steps and one target verify step over the padded running batch. A
  GEMM's verify and draft calls run M = r x batch (r = draft tokens per
  sequence on verify; on the draft model 1 per draft step and more on its
  extend step); r and the calls per round of each r are measured in the
  windows, so every logged round adds calls at M = r x padded batch;
* calls per forward: the GEMM's calls per recorded step in the windows (one
  per layer);
* cost per call: piecewise-linear in log M through the per-call times measured
  in the windows. Below the smallest measured M the smallest point's cost is
  used; above the largest, cost grows in proportion to M. Both are marked as
  extrapolated.

A GEMM group is token-indexed when at least ``TOKEN_INDEXED_SHARE`` of its
recorded calls in a phase have M equal to their own step's token count
(prefill) or batch (decode); others (the LM head on prefill, which sees one row per
sequence) keep their window-scaled rows. This follows the log-weighted mapping
of the Qwen3-32B probe (scheduler-log instance counts x measured per-call cost).
"""
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict

TOKEN_INDEXED_SHARE = 0.9
METHOD = ("scheduler-log forwards (prefill M = new tokens per batch, decode M = running batch padded to the "
          "CUDA-graph size; speculative verify/draft M = r x padded batch, r and calls per round measured) x calls "
          "per forward x per-call cost interpolated in log M from the profiler windows")
ROUND_PHASES = ("verify", "draft")


def cost_model(points: list[tuple[int, float]]):
    """[(M, us per call)] -> f(M) -> (us, extrapolated)."""
    pts = sorted(points)
    ms = [m for m, _ in pts]

    def cost(m: int) -> tuple[float, bool]:
        if m <= ms[0]:
            return pts[0][1], m < ms[0]
        if m >= ms[-1]:
            return pts[-1][1] * m / ms[-1], m > ms[-1]
        i = bisect.bisect_left(ms, m)
        if ms[i] == m:
            return pts[i][1], False
        (m0, u0), (m1, u1) = pts[i - 1], pts[i]
        t = (math.log(m) - math.log(m0)) / (math.log(m1) - math.log(m0))
        return u0 + t * (u1 - u0), False
    return cost


def snap_up(value: int, sizes: list[int]) -> int:
    i = bisect.bisect_left(sizes, value)
    return sizes[i] if i < len(sizes) else value


def logged_forwards(prefill: list[dict], decode: list[dict], graph_bs: list[int], chunk: int | None) -> dict:
    """{'prefill': Counter(M), 'decode': Counter(M)} of forwards in the scheduler log."""
    out = {"prefill": Counter(), "decode": Counter()}
    for p in prefill:
        n = p["new_token"]
        while n > 0:
            m = min(n, chunk) if chunk else n
            out["prefill"][m] += 1
            n -= m
    sizes = sorted(graph_bs)
    for d in decode:
        if d["running"] > 0:
            out["decode"][snap_up(d["running"], sizes) if sizes else d["running"]] += 1
    return out


def weighted_gemm(rows: list[dict], steps: dict, step_sizes: dict, forwards: dict) -> tuple[list[dict], list[dict]]:
    """Log-weighted GEMM rows and a per-group summary.

    ``rows``: measured (unweighted) census GEMM rows; ``steps``: recorded steps per phase (for verify and
    draft: recorded decode rounds, i.e. verify steps);
    ``step_sizes``: {'prefill': set of recorded step token counts, 'decode': set of recorded batches}, used
    only for rows without a ``token_matched`` count (calls whose M equalled their own step's size);
    ``forwards``: ``logged_forwards`` output.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["N"], r["K"], r["dtype"], r.get("B", 1))].append(r)
    out, summary = [], []
    for (n, k, dtype, b), grp in groups.items():
        calls = defaultdict(float)
        points = defaultdict(lambda: [0.0, 0.0])
        info = {"N": n, "K": k, "dtype": dtype, "B": b, "phases": {}}
        for r in grp:
            calls[r["phase"]] += r["calls"]
            points[r["M"]][0] += r["total_us"]
            points[r["M"]][1] += r["calls"]
        cost = cost_model([(m, us / c) for m, (us, c) in points.items() if c])
        for phase in ("prefill", "decode", "draft_prefill"):
            recorded = [r for r in grp if r["phase"] == phase]
            # the draft model's prefill extend runs once per target prefill forward, over the same tokens
            base = "prefill" if phase == "draft_prefill" else phase
            if not recorded or not steps.get(base):
                continue
            indexed = sum(r["token_matched"] if "token_matched" in r else
                          (r["calls"] if r["M"] in step_sizes.get(phase, ()) else 0) for r in recorded)
            share = indexed / sum(r["calls"] for r in recorded)
            per_step = sum(r["calls"] for r in recorded) / steps[base]
            ok = share >= TOKEN_INDEXED_SHARE
            info["phases"][phase] = {"token_indexed_share": round(share, 3), "calls_per_step": round(per_step, 3),
                                     "log_weighted": ok}
            if not ok:
                continue
            extrapolated = []
            for m, count in sorted(forwards[base].items()):
                us, extra = cost(m)
                if extra:
                    extrapolated.append(m)
                c = count * per_step
                out.append({"M": m, "N": n, "K": k, "dtype": dtype, "B": b, "phase": phase, "forwards": count,
                            "calls": round(c, 1), "cost_us": round(us, 3), "total_us": round(c * us, 2),
                            "extrapolated": extra})
            info["phases"][phase]["extrapolated_M"] = extrapolated
        for phase in ROUND_PHASES:
            recorded = [r for r in grp if r["phase"] == phase]
            if not recorded or not steps.get(phase):
                continue
            ratios = Counter()
            for r in recorded:
                for ratio, c in (r.get("ratios") or {}).items():
                    ratios[int(ratio)] += c
            total = sum(r["calls"] for r in recorded)
            share = sum(ratios.values()) / total if total else 0.0
            ok = share >= TOKEN_INDEXED_SHARE
            info["phases"][phase] = {"token_indexed_share": round(share, 3),
                                     "calls_per_round": round(total / steps[phase], 3), "log_weighted": ok,
                                     "ratios": {ratio: round(c / steps[phase], 3) for ratio, c in sorted(ratios.items())}}
            if not ok:
                continue
            extrapolated = []
            merged = defaultdict(lambda: [0, 0.0])  # M -> [forwards, calls]
            for ratio, c_ratio in ratios.items():
                per_round = c_ratio / steps[phase]
                for bs, count in forwards["decode"].items():
                    merged[ratio * bs][0] += count
                    merged[ratio * bs][1] += count * per_round
            for m, (count, c) in sorted(merged.items()):
                us, extra = cost(m)
                if extra:
                    extrapolated.append(m)
                out.append({"M": m, "N": n, "K": k, "dtype": dtype, "B": b, "phase": phase, "forwards": count,
                            "calls": round(c, 1), "cost_us": round(us, 3), "total_us": round(c * us, 2),
                            "extrapolated": extra})
            info["phases"][phase]["extrapolated_M"] = extrapolated
        info["measured_points"] = sorted((m, round(us / c, 3)) for m, (us, c) in points.items() if c)
        summary.append(info)
    out.sort(key=lambda r: -r["total_us"])
    return out, summary
