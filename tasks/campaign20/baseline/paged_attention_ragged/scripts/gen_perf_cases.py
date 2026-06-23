#!/usr/bin/env python3
"""Inject a memory-bound performance case into the ragged ``test_cases.json``.

Unlike the vLLM paged_attention task, the ragged runner already builds *legal,
self-consistent* KV bookkeeping at replay time via
``_runtime.make_consistent_paged_attention_ragged`` — it overwrites kv_indptr /
kv_page_indices / kv_last_page_lens deterministically from the tensor *shapes*
(even split of the page table across sequences, in-bounds page ids). So there is
no need to store literal index ``data`` here: the workload is controlled purely
by the shapes, and legality is guaranteed by that repair pass.

This script appends one ``perf_only`` case scaled from a captured template:
each of S sequences gets a context of L tokens (block_size==1 ⇒ L pages/seq),
sized so the streamed KV (~S*L*kv_heads*head*2*2 bytes) far exceeds MI300X's
256MB last-level cache and the kernel must stream from HBM (memory-bound).

  python3 scripts/gen_perf_cases.py            # default S=512, ctx=4096 (~4.3GB KV)

Idempotent: existing ``perf_only`` cases are removed before re-appending.
"""
import argparse
import copy
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CASES = os.path.join(os.path.dirname(HERE), "test_cases.json")

# ragged positional layout (see src/.../pa/pa_ragged.py):
# 0 out 1 workspace 2 query 3 key_cache 4 value_cache 5 scale 6 kv_indptr
# 7 kv_page_indices 8 kv_last_page_lens 9 block_size 10 max_num_partitions
# 11 alibi 12 kv_cache_dtype 13 layout 14 logits_soft_cap 15 k_scale 16 v_scale
# 17 fp8_out_scale 18 partition_size
I_OUT, I_WS, I_Q, I_KC, I_VC = 0, 1, 2, 3, 4
I_INDPTR, I_PAGEIDX, I_LASTLEN = 6, 7, 8
I_BLOCK, I_MAXPART, I_PART = 9, 10, 18


def _shape(sig):
    return list(sig.get("shape", []))


def _scalar(sig):
    return sig.get("value")


def build_case(S: int, L: int, template: dict) -> dict:
    tc = copy.deepcopy(template)
    a = tc["args_sig"]

    # Structural params come from the captured template so the compiled kernel
    # variant (gqa_ratio, head_size, block_size, layout) is unchanged.
    H = _shape(a[I_OUT])[1]
    head = _shape(a[I_OUT])[2]
    layout = a[13].get("repr", "'NHD'")
    kc = _shape(a[I_KC])                       # [num_blocks, block_size, kv_heads, head] (NHD)
    block_size = int(_scalar(a[I_BLOCK]) or kc[1])
    kvh = kc[2] if "NHD" in layout else kc[1]
    part = int(_scalar(a[I_PART]) or 256)

    pages_per_seq = math.ceil(L / block_size)
    total_pages = S * pages_per_seq
    num_blocks = total_pages                   # unique page per slot -> real HBM
    max_part = math.ceil((pages_per_seq * block_size) / part)

    # Workspace is raw scratch (kernel casts the uint8 buffer). Captured cases
    # over-allocate hugely; size generously (>> S*H*max_part*(2*f32 + head*dtype)).
    ws_bytes = max(S * H * max_part * 4096, 256 * 1024 * 1024)

    a[I_OUT]["shape"] = [S, H, head]
    a[I_Q]["shape"] = [S, H, head]
    a[I_WS]["shape"] = [ws_bytes]
    a[I_KC]["shape"] = [num_blocks, block_size, kvh, head] if "NHD" in layout \
        else [num_blocks, kvh, block_size, head]
    a[I_VC]["shape"] = list(a[I_KC]["shape"])
    a[I_INDPTR]["shape"] = [S + 1]
    a[I_PAGEIDX]["shape"] = [total_pages]      # length drives ctx; values filled by make_consistent
    a[I_LASTLEN]["shape"] = [S]
    a[I_MAXPART]["value"] = max_part

    kv_gb = 2 * num_blocks * block_size * kvh * head * 2 / 1e9  # K+V, bf16
    tc["test_case_id"] = f"perf_S{S}_ctx{L}"
    tc["perf_only"] = True
    tc["count"] = 1
    tc["params_repr"] = {
        "S_seqs": S, "ctx_len": L, "out_len": 1, "heads": H, "kv_heads": kvh,
        "gqa": H // max(1, kvh), "head_size": head, "block_size": block_size,
        "partition": part, "max_num_partitions": max_part, "layout": layout.strip("'"),
        "num_blocks": num_blocks, "kv_alloc_gb": round(kv_gb, 1),
        "note": "decode; even-split ragged page table (make_consistent) -> HBM-streaming",
    }
    return tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=512, help="concurrent sequences (batch)")
    ap.add_argument("-L", type=int, default=4096, help="context length per sequence (tokens)")
    args = ap.parse_args()

    with open(TEST_CASES) as f:
        cases = json.load(f)
    captured = [c for c in cases if not c.get("perf_only")]
    new = build_case(args.S, args.L, captured[0])
    out = captured + [new]
    with open(TEST_CASES, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen_perf_cases] wrote {len(captured)} captured + 1 perf_only "
          f"({new['test_case_id']}, KV~{new['params_repr']['kv_alloc_gb']}GB, "
          f"gqa {new['params_repr']['gqa']}:1) -> {TEST_CASES}")


if __name__ == "__main__":
    main()
