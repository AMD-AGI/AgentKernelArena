# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 reference of one DeepSeek-V4.1-Flash K2 launch on a TP4 group (logical inputs and outputs;
the definition is dsv41_math's docstring). It never imports the baseline."""
from dsv41_math import k2_ffn_input, moe_out

SWA_BLOCK, COMP_BLOCK = 64, 128


def reference(*, q, swa, comp, kt, klen, pos, sink, cos_sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in,
              pre_in, hc_fn, hc_scale, hc_base, ffn_norm, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2,
              sw2_s):
    inp = dict(q=q, swa=swa, comp=comp, kt=kt, klen=klen, pos=pos, sink=sink, cos_sin=cos_sin, woa=woa,
               woa_s=woa_s, wob=wob, wob_s=wob_s, res_in=res_in, post_in=post_in, comb_in=comb_in, pre_in=pre_in,
               hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base, ffn_norm=ffn_norm, gate_w=gate_w, bias=bias,
               w13=w13, w13_s=w13_s, w2=w2, w2_s=w2_s, sgu=sgu, sgu_s=sgu_s, sw2=sw2, sw2_s=sw2_s,
               swa_block=SWA_BLOCK, comp_block=COMP_BLOCK)
    ffn = k2_ffn_input(inp)
    # bf16 outputs are returned before their final rounding (the kernel's rounding is the comparison's own)
    return {
        "out": moe_out(ffn["y"], inp),
        "res_out": ffn["r_raw"],
        "post_out": ffn["post_out"],
        "comb_out": ffn["comb_out"],
        "pre_out": ffn["pre_out"],
    }
