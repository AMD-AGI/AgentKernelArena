# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Closed-loop AgentX corpus replay against an SGLang server's native ``/generate`` API.

Runs inside the serving image (it needs ``aiohttp`` and ``numpy``); ``capture``
starts it in a client container. It keeps the parts of the AgentX recipe that
shape kernel work:

* corpus: the weka cc-traces ``traces.jsonl`` (one trajectory per line);
* lanes: ``--lanes`` closed-loop lanes, each replaying one trajectory at a time,
  trajectories shuffled with ``--seed``;
* trajectory start: uniform in ``[--start-min, --start-max]`` of the trajectory;
* think time between turns from the trace, capped at ``--think-cap`` seconds
  (300 s, aiperf's idle-gap cap in the AgentX recipe);
* prefix reuse: each prompt is built from its weka ``hash_ids``, one fixed
  64-token block per (trajectory, hash id), and sent as ``input_ids``, so the
  input length is exact and the server's radix cache sees the corpus's
  shared-prefix structure;
* subagent groups replay as their own turn sequences; consecutive groups run
  concurrently and the main agent waits for them;
* output length is the trace's ``out`` with ``ignore_eos``.

A request whose input plus output exceeds ``--max-ctx`` (the served context
window) is skipped and its trajectory continues, as in the official AgentX
capped variants (agentic traces compact their context, so later turns can fit
again); ``corpus_coverage`` reports how much of the corpus a window reaches.
Every request is appended to ``--log`` (JSON lines) for the observed-workload
analysis.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

BLOCK = 64  # tokens per weka hash id


def fits(request: dict, max_ctx: int, max_out: int) -> bool:
    """Whether a request's input plus (capped) output fits the served context window."""
    return request["in"] + min(request.get("out", 1), max_out) <= max_ctx


def corpus_coverage(traces: str, max_ctx: int, max_out: int) -> dict:
    """How much of the corpus the served context window reaches (requests over it are skipped)."""
    stats = {"context_cap_tokens": max_ctx, "trajectories": 0, "trajectories_unreachable": 0,
             "requests": 0, "requests_dropped": 0, "input_tokens": 0, "input_tokens_dropped": 0}
    with open(traces, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["trajectories"] += 1
            reachable = 0
            requests = [x for r in rec.get("requests", [])
                        for x in (r.get("requests", []) if r.get("type") == "subagent" else [r])]
            for x in requests:
                if "in" not in x:
                    continue
                stats["requests"] += 1
                stats["input_tokens"] += int(x["in"])
                if fits(x, max_ctx, max_out):
                    reachable += 1
                else:
                    stats["requests_dropped"] += 1
                    stats["input_tokens_dropped"] += int(x["in"])
            stats["trajectories_unreachable"] += reachable == 0
    stats["dropped_share"] = round(stats["requests_dropped"] / stats["requests"], 4) if stats["requests"] else 0.0
    stats["input_tokens_dropped_share"] = (round(stats["input_tokens_dropped"] / stats["input_tokens"], 4)
                                           if stats["input_tokens"] else 0.0)
    return stats


def build_session(rec: dict, max_ctx: int, max_out: int, rng: random.Random,
                  start_min: float, start_max: float) -> list:
    """One trajectory as a list of ("req", request) and ("sub", [request lists]) events."""
    events = []
    for r in rec["requests"]:
        if r.get("type") == "subagent":
            inner = []
            for x in r.get("requests", []):
                if "in" not in x or fits(x, max_ctx, max_out):
                    inner.append(x)
            if inner:
                if events and events[-1][0] == "sub":
                    events[-1][1].append(inner)
                else:
                    events.append(("sub", [inner]))
        else:
            if "in" not in r:
                continue
            if fits(r, max_ctx, max_out):
                events.append(("req", r))
    if not events:
        return []
    start = int(rng.uniform(start_min, start_max) * len(events))
    return events[min(start, len(events) - 1):]


def prompt_ids(seed: int, trajectory: int, request: dict, vocab_hi: int, cache: dict) -> list[int]:
    """Token ids of one request: a fixed block per (trajectory, hash id), then a unique tail."""
    import numpy as np

    n = int(request["in"])
    out: list[int] = []
    for h in request.get("hash_ids", []):
        block = cache.get((trajectory, h))
        if block is None:
            block = np.random.default_rng([seed, trajectory, h]).integers(1000, vocab_hi, BLOCK).tolist()
            cache[(trajectory, h)] = block
        out.extend(block)
        if len(out) >= n:
            break
    if len(out) < n:  # request-unique tail: no false prefix hits
        out.extend(np.random.default_rng([seed, trajectory, -1, n, len(out)]).integers(
            1000, vocab_hi, n - len(out)).tolist())
    return out[:n]


class Replay:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.url = args.base_url.rstrip("/") + "/generate"
        self.deadline = time.time() + args.duration
        self.logf = open(args.log, "a", buffering=1)
        self.block_cache: dict = {}
        self.n_ok = self.n_err = self.inflight = 0

    async def send(self, http, lane: int, tidx: int, kind: str, r: dict) -> bool:
        import aiohttp

        if time.time() >= self.deadline:
            return False
        ids = prompt_ids(self.a.seed, tidx, r, self.a.vocab_hi, self.block_cache)
        osl = max(1, min(int(r.get("out", 1)), self.a.max_out))
        body = {"input_ids": ids, "sampling_params": {
            "max_new_tokens": osl, "ignore_eos": True, "temperature": 0.0}}
        t0 = time.time()
        self.inflight += 1
        rec = {"lane": lane, "trace": tidx, "kind": kind, "in": len(ids), "out": osl, "t0": t0}
        ok = False
        try:
            async with http.post(self.url, json=body,
                                 timeout=aiohttp.ClientTimeout(total=self.a.request_timeout)) as resp:
                js = await resp.json(content_type=None)
                rec["status"] = resp.status
                meta = js.get("meta_info", {}) if isinstance(js, dict) else {}
                rec["cached"] = meta.get("cached_tokens")
                rec["completion_tokens"] = meta.get("completion_tokens")
                ok = resp.status == 200
        except Exception as exc:  # noqa: BLE001 - logged; the lane continues
            rec["status"] = "exc:%s" % type(exc).__name__
        finally:
            self.inflight -= 1
        rec["t1"] = time.time()
        rec["lat"] = rec["t1"] - t0
        self.logf.write(json.dumps(rec) + "\n")
        if ok:
            self.n_ok += 1
        else:
            self.n_err += 1
        return True

    async def turns(self, http, lane: int, tidx: int, kind: str, reqs: list) -> None:
        for i, r in enumerate(reqs):
            if i:
                await asyncio.sleep(min(float(r.get("think_time") or 0.0), self.a.think_cap))
            if not await self.send(http, lane, tidx, kind, r):
                return

    async def lane(self, http, lane: int, order: list[int], lines: list[bytes]) -> None:
        rng = random.Random(self.a.seed * 1000 + lane)
        k = lane
        while time.time() < self.deadline:
            tidx = order[k % len(order)]
            k += self.a.lanes
            events = build_session(json.loads(lines[tidx]), self.a.max_ctx, self.a.max_out, rng,
                                   self.a.start_min, self.a.start_max)
            for i, (kind, payload) in enumerate(events):
                if time.time() >= self.deadline:
                    break
                if kind == "req":
                    if i:
                        await asyncio.sleep(min(float(payload.get("think_time") or 0.0), self.a.think_cap))
                    await self.send(http, lane, tidx, "main", payload)
                else:
                    await asyncio.gather(*(self.turns(http, lane, tidx, "sub", g) for g in payload))

    async def stats(self) -> None:
        t_start = time.time()
        while time.time() < self.deadline:
            await asyncio.sleep(30)
            print("[replay] t=%4.0fs ok=%d err=%d inflight=%d" % (
                time.time() - t_start, self.n_ok, self.n_err, self.inflight), flush=True)

    async def run(self, lines: list[bytes]) -> None:
        import aiohttp

        order = list(range(len(lines)))
        random.Random(self.a.seed).shuffle(order)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
            stats = asyncio.create_task(self.stats())
            lanes = []
            for i in range(self.a.lanes):
                lanes.append(asyncio.create_task(self.lane(http, i, order, lines)))
                await asyncio.sleep(self.a.ramp_s / max(1, self.a.lanes))
            await asyncio.gather(*lanes)
            stats.cancel()
        print("[replay] done ok=%d err=%d" % (self.n_ok, self.n_err), flush=True)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--traces", required=True, help="the corpus traces.jsonl")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--lanes", type=int, default=32, help="closed-loop lanes (concurrent trajectories)")
    ap.add_argument("--duration", type=float, default=1200, help="seconds after which no request starts")
    ap.add_argument("--max-ctx", type=int, default=131072, help="served context window (tokens)")
    ap.add_argument("--max-out", type=int, default=16384)
    ap.add_argument("--think-cap", type=float, default=300.0, help="cap on the trace's think time (seconds)")
    ap.add_argument("--start-min", type=float, default=0.25)
    ap.add_argument("--start-max", type=float, default=0.75)
    ap.add_argument("--ramp-s", type=float, default=60.0, help="stagger lane starts over this many seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-hi", type=int, default=150000, help="exclusive upper token id of prompts")
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--log", required=True, help="JSON-lines request log (appended)")


def main(args: argparse.Namespace) -> int:
    t = time.time()
    with open(args.traces, "rb") as fh:
        lines = [ln for ln in fh if ln.strip()]
    print("[replay] loaded %d trajectories in %.1fs" % (len(lines), time.time() - t), flush=True)
    replay = Replay(args)
    replay.deadline = time.time() + args.duration
    asyncio.run(replay.run(lines))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    sys.exit(main(parser.parse_args()))
