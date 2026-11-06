"""Float64 reference for the MXFP4 MoE expert GEMM, used only for correctness.

c[t, k] = dequant(a[t]) @ dequant(b[topk_ids[t, k]])^T for every token t and slot k, with MXFP4 decoded
as in mxfp4_numerics.py. The sorted-token layout inputs only describe how a kernel may tile this work;
the result depends on topk_ids alone.
"""
import torch

from mxfp4_numerics import dequantize


def reference(*, a, a_scale, b, b_scale, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
    n_tokens, topk = topk_ids.shape
    x = dequantize(a, a_scale)
    out = torch.zeros((n_tokens, topk, b.shape[1]), dtype=torch.float64, device=a.device)
    ids = topk_ids.long()
    for expert in torch.unique(ids).tolist():
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        out[tokens, slots] = x[tokens] @ dequantize(b[expert], b_scale[expert]).T
    return out
