#!/usr/bin/env python3
"""ASM .co baseline (fmoe_fp8_blockscale_g1u1) at MiniMax TP=2 per-GPU MoE shape.
Reference bar for apply-back: a FlyDSL/CK rewrite must beat this to be worth shipping.
Reuses moe_harness.prepare() so quant layouts match exactly the production path.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# import-isolation: hide flydsl from importlib during `import aiter` (proven driver pattern).
import importlib.util as _ilu
_orig = _ilu.find_spec
def _hidden(name, *a, **k):
    if name == "flydsl" or name.startswith("flydsl."):
        return None
    return _orig(name, *a, **k)
_ilu.find_spec = _hidden

import torch
import moe_harness as H
import aiter
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest
_ilu.find_spec = _orig


def asm_call(a1_q, w1_s, w2_s, topk_weights, topk_ids, w1_scale, w2_scale, a1_scale_t,
             E, model_dim, dtype, scale_blk=(128, 128)):
    topk = topk_ids.shape[-1]
    sorted_ids, sorted_w, sorted_eids, num_valid, out_asm = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype)
    bn, bk = scale_blk
    aiter.fmoe_fp8_blockscale_g1u1(
        out_asm, a1_q, w1_s, w2_s, sorted_ids, sorted_w, sorted_eids, num_valid,
        topk, a1_scale_t, w1_scale, w2_scale, "", bn, bk, None)
    return out_asm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=int, default=64)
    args = ap.parse_args()
    prep = H.prepare(args.token)
    dtype = torch.bfloat16
    w1_s = shuffle_weight(prep["w1_q"], (16, 16))
    w2_s = shuffle_weight(prep["w2_q"], (16, 16))
    a1_scale_t = prep["a1_scale"].t().contiguous()
    out, us = run_perftest(
        asm_call, prep["a1_q"], w1_s, w2_s, prep["topk_weights"], prep["topk_ids"],
        prep["w1_scale"], prep["w2_scale"], a1_scale_t, prep["expert"], prep["model_dim"],
        dtype, num_warmup=10, num_iters=50)
    gout = H.golden(prep)
    snr = H.snr_db(gout, out)
    cos = H.cosine_diff(gout, out)
    print(f"ASM fmoe_fp8_blockscale_g1u1  token={args.token}  "
          f"model_dim={prep['model_dim']} inter_dim={prep['inter_dim']} E={prep['expert']} topk={prep['topk']}")
    print(f"ASM median_us: {us:.2f}  (= {us/1000:.5f} ms)")
    print(f"ASM SNR vs torch golden: {snr:.2f} dB")
    print(f"ASM cosine_diff vs torch golden: {cos:.6e}")


if __name__ == "__main__":
    main()
