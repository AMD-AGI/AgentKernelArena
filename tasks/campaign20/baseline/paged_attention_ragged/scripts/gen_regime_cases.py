#!/usr/bin/env python3
"""Inject the arena WORKLOAD REGIME perf cases into test_cases.json.

Regime: decode-style paged attention (q_len=1), context/kv_len = 1024,
concurrency (batch = number of concurrent sequences) B in {2, 32, 64}.
Three perf_only cases with ids c2 / c32 / c64.

Model dims (heads, kv_heads, head_size, block_size, partition, layout) are kept
from the captured base case via gen_perf_cases.build_case -> only S (batch) and
L (context length) vary. Legality of the ragged page table is guaranteed at
replay time by _runtime.make_consistent_paged_attention_ragged.

Idempotent: removes any existing perf_only cases before appending the 3 regime
cases.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_perf_cases as g  # noqa: E402

TEST_CASES = os.path.join(os.path.dirname(HERE), "test_cases.json")

CTX = 1024          # input/output regime seqlen (decode: context = kv_len = 1024)
CONCURRENCY = [2, 32, 64]


def main():
    with open(TEST_CASES) as f:
        cases = json.load(f)
    captured = [c for c in cases if not c.get("perf_only")]
    template = captured[0]
    new_cases = []
    for B in CONCURRENCY:
        tc = g.build_case(B, CTX, template)
        tc["test_case_id"] = f"c{B}"            # arena id convention
        new_cases.append(tc)
    out = captured + new_cases
    with open(TEST_CASES, "w") as f:
        json.dump(out, f, indent=2)
    for tc in new_cases:
        p = tc["params_repr"]
        print(f"[gen_regime_cases] {tc['test_case_id']}: B={p['S_seqs']} "
              f"ctx={p['ctx_len']} heads={p['heads']} kvh={p['kv_heads']} "
              f"gqa={p['gqa']} head={p['head_size']} bs={p['block_size']} "
              f"KV~{p['kv_alloc_gb']}GB")
    print(f"[gen_regime_cases] wrote {len(captured)} captured + {len(new_cases)} perf_only -> {TEST_CASES}")


if __name__ == "__main__":
    main()
