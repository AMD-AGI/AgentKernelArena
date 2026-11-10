# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 reference of one DeepSeek-V4.1-Flash K1 launch on a TP4 group (logical inputs and outputs; the
definition is dsv41_math's docstring). It never imports the baseline."""
from dsv41_math import encode_records, k1_front, key_table, seam, write_records

SWA_BLOCK = 64


def reference(*, res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_norm, wqkv, wqkv_s, q_norm,
              kv_norm, wqb, wqb_s, cos_sin, pos, slot, swa, swa_idx, swa_lens, topk, topk_lens):
    sm = seam(res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_norm)
    front = k1_front(sm["y"], dict(wqkv=wqkv, wqkv_s=wqkv_s, q_norm=q_norm, kv_norm=kv_norm, wqb=wqb, wqb_s=wqb_s,
                                   cos_sin=cos_sin, pos=pos))
    cache = swa.clone()
    live = slot >= 0
    data, scales = encode_records(front["kn"][live])
    write_records(cache, slot[live], data, scales, SWA_BLOCK)
    kt, klen = key_table(swa_idx, swa_lens, topk, topk_lens, slot)
    # bf16 outputs are returned before their final rounding (the kernel's rounding is the comparison's own)
    return {
        "q": front["q_raw"],
        "res_out": sm["r_raw"],
        "post_out": sm["post_out"],
        "comb_out": sm["comb_out"],
        "pre_out": sm["pre_out"],
        "kt": kt,
        "klen": klen,
        "swa": cache,
    }
