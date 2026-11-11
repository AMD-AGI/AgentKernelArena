# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""High-precision reference of one Kimi-K3 layer-post launch on a TP8 group, used only for correctness.

It receives the logical inputs (inputs.py: the TP-sharded weights at their full logical shapes, the routed
experts as hash keys) and returns the logical outputs, both replicated on every rank: ``out`` (the layer's
MoE output plus the updated residual prefix) and ``prefix`` (the updated attention-residual prefix), each
before its final bfloat16 rounding. The numerics are in k3_numerics.layer (float64, the production
tensors' bfloat16 hand-offs and the MXFP4 quantizations of the latent and of the experts' SiTUv2 output
applied exactly). It never imports the baseline.
"""
from k3_numerics import layer


def reference(*, core, w_o, prefix_in, blocks, ares_nw, ares_qk, in_nw, w_gate, bias, w_ld, w_sgu, w_sd,
              w13, w13s, w2, w2s, ln_w, w_up, guard):
    inp = {"core": core, "w_o": w_o, "prefix_in": prefix_in, "blocks": blocks, "ares_nw": ares_nw,
           "ares_qk": ares_qk, "in_nw": in_nw, "w_gate": w_gate, "bias": bias, "w_ld": w_ld, "w_sgu": w_sgu,
           "w_sd": w_sd, "w13": w13, "w13s": w13s, "w2": w2, "w2s": w2s, "ln_w": ln_w, "w_up": w_up}
    got = layer(inp)
    return {"out": got["out"], "prefix": got["prefix"]}
