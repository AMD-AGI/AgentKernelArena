"""Float64 absorbed MLA decode attention over the FP8 latent KV cache, used only for correctness.

The worklist inputs (work_*, reduce_*) only schedule the computation and are not read here.
"""
import torch

KV_LORA_RANK = 512


def reference(*, q, kv_cache, kv_indptr, kv_page_indices, work_indptr, work_info, reduce_indptr, reduce_final_map,
              reduce_partial_map, softmax_scale):
    num_seqs, num_heads, head_size = q.shape
    rows = kv_cache.view(kv_cache.shape[0], head_size)
    out = torch.empty((num_seqs, num_heads, KV_LORA_RANK), dtype=torch.float64, device=q.device)
    bounds = kv_indptr.tolist()
    for b in range(num_seqs):
        pages = kv_page_indices[bounds[b]:bounds[b + 1]].long()
        kv = rows[pages].double()  # [tokens, 576]; the value is its first 512 columns
        scores = (q[b].double() @ kv.T) * float(softmax_scale)  # [heads, tokens]
        out[b] = torch.softmax(scores, dim=-1) @ kv[:, :KV_LORA_RANK]
    return out
