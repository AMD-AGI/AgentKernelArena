# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""High-precision reference of one GLM-5.3-Flash sparse-attention decoder layer, used only for correctness.

The layer is defined in ``dsa_layer_math.py`` (float64 PyTorch, with the operator's quantizations
emulated exactly); this module returns its outputs, then the final caches, in the task's order. It
computes the logical (whole-model) layer: every attention head and the whole expert width, i.e. the
sum over the four tensor-parallel ranks the kernel reduces in place.
"""
from dsa_layer_math import layer


def reference(**inputs):
    out = layer(inputs)
    return (out["x_out"], out["res_out"], out["post_out"], out["comb_out"], out["kv_cache"],
            out["index_cache"], out["kpool_tail_k"], out["kpool_tail_gate"])
