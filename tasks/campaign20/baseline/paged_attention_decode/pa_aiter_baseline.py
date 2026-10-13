#!/usr/bin/env python3
"""aiter decode paged-attention baseline (production CK/HIP kernel) at MiniMax bf16 shape.
Validates the shared golden and gives the apply-back bar (us/call) for the CK rewrite.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# import-isolation shim (consistent with K13 drivers; harmless here).
import importlib.util as _ilu
_orig = _ilu.find_spec
def _hidden(name, *a, **k):
    if name == "flydsl" or name.startswith("flydsl."):
        return None
    return _orig(name, *a, **k)
_ilu.find_spec = _hidden

import torch
import pa_harness as H
import aiter
_ilu.find_spec = _orig


def run_aiter(prep, partition_size=H.PARTITION_SIZE):
    query, kc, vc = prep["query"], prep["kc_nhd"], prep["vc_nhd"]
    num_seqs, num_heads, head_size = query.shape
    output = torch.empty_like(query)
    max_num_partitions = (prep["max_seq_len"] + partition_size - 1) // partition_size
    nbytes = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * num_heads * max_num_partitions * head_size) * nbytes
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8, device=output.device)
    k_scale = v_scale = torch.tensor([1.0], dtype=torch.float32, device=output.device)
    torch.ops.aiter.paged_attention_v1(
        output, workspace_buffer, query, kc, vc, prep["scale"],
        prep["block_tables"], prep["cu_query_lens"], prep["seq_lens"], prep["max_seq_len"],
        None, "auto", "NHD", 0.0, k_scale, v_scale, None, partition_size, sliding_window=0)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-seqs", type=int, default=H.NUM_SEQS)
    ap.add_argument("--ctx-len", type=int, default=H.CTX_LEN)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    prep = H.prepare(args.num_seqs, args.ctx_len)
    g = H.golden(prep)
    out = run_aiter(prep)
    torch.cuda.synchronize()
    snr = H.snr_db(g, out)
    cos = H.cosine_diff(g, out)
    print(f"aiter paged_attention_v1  num_seqs={args.num_seqs} ctx={args.ctx_len} "
          f"q={H.NUM_Q_HEADS} kv={H.NUM_KV_HEADS} head={H.HEAD_SIZE} block={H.BLOCK_SIZE} bf16-KV")
    print(f"aiter SNR vs torch golden: {snr:.2f} dB  cosine_diff={cos:.4e}")
    for _ in range(args.warmup):
        run_aiter(prep)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(args.reps):
        run_aiter(prep)
    en.record(); torch.cuda.synchronize()
    us = st.elapsed_time(en) * 1000.0 / args.reps
    print(f"aiter median_us: {us:.2f}  (= {us/1000:.5f} ms) over {args.reps} reps")


if __name__ == "__main__":
    main()
