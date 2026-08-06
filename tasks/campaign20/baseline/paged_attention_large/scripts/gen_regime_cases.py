#!/usr/bin/env python3
"""Set test_cases.json perf_only cases to the arena WORKLOAD REGIME.

Regime: decode-style paged_attention. ctx_len (KV length) L = 1024, one query
token per sequence (q_len=1). Concurrency B in {2,32,64} maps to S = num_seqs
(batch). Model dims kept from the captured base case: head_size=128, block_size=16,
H=16 query heads, KVH=1 (GQA 16:1), X=8, PARTITION_SIZE=256.

Emits exactly 3 perf_only cases with ids c2,c32,c64. Captured (correctness)
cases are preserved untouched.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_perf_cases as g

TEST_CASES = os.path.join(os.path.dirname(HERE), "test_cases.json")

L = 1024            # ctx / KV length per sequence (= arena seqlen)
H = g.DEFAULT_H     # 16 query heads (captured)
KVH = g.DEFAULT_KVH # 1 kv head (captured GQA 16:1)
REGIME = [2, 32, 64]

def main():
    with open(TEST_CASES) as f:
        cases = json.load(f)
    captured = [c for c in cases if not c.get("perf_only")]
    # template with matching GQA layout
    def heads_of(c):
        q = c["args_sig"][4]; kv = c["args_sig"][7]
        return (q.get("shape", [0, 0])[1], kv.get("value"))
    tmpl = next((c for c in captured if heads_of(c) == (H, KVH)), captured[0])
    new = []
    for B in REGIME:
        tc = g.build_case(B, L, H, KVH, tmpl)
        tc["test_case_id"] = f"c{B}"
        tc["params_repr"]["concurrency_B"] = B
        new.append(tc)
    out = captured + new
    with open(TEST_CASES, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen_regime_cases] {len(captured)} captured + {len(new)} perf_only:")
    for c in new:
        print(f"    {c['test_case_id']:6s} S={c['params_repr']['S_seqs']} L={c['params_repr']['ctx_len']} KV~{c['params_repr']['kv_alloc_gb']}GB")

if __name__ == "__main__":
    main()
