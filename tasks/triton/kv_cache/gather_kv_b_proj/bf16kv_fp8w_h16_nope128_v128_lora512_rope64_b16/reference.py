"""Float64 reference: gather the latent rows of every request and expand them with the dequantized kv_b weight."""
import torch

_BLOCK = 128


def reference(*, k_buffer, kv_indptr, kv_indices, kv_prefix_sum_context_lens, kv_proj_weight, kv_proj_scale):
    _, block_size, latent = k_buffer.shape
    rows, k = kv_proj_weight.shape
    total = int(kv_prefix_sum_context_lens[-1])
    pe_dim = latent - k
    # The task fixes 16 heads of 128 + 128 outputs; derive the head count from the output rows.
    head_out = 256
    heads = rows // head_out
    nope = head_out // 2
    scale = kv_proj_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    weight = kv_proj_weight.double() * scale
    k_prefix = torch.empty((total, heads, nope + pe_dim), dtype=torch.float64, device=k_buffer.device)
    v_prefix = torch.empty((total, heads, head_out - nope), dtype=torch.float64, device=k_buffer.device)
    starts = kv_prefix_sum_context_lens.tolist()
    block_starts = kv_indptr.tolist()
    for b in range(len(starts) - 1):
        first, last = starts[b], starts[b + 1]
        blocks = kv_indices[block_starts[b]:block_starts[b + 1]].long()
        data = k_buffer[blocks].reshape(-1, latent)[:last - first].double()
        proj = (data[:, :k] @ weight.T).view(last - first, heads, head_out)
        k_prefix[first:last, :, :nope] = proj[..., :nope]
        k_prefix[first:last, :, nope:] = data[:, None, k:]
        v_prefix[first:last] = proj[..., nope:]
    return {"k_prefix": k_prefix, "v_prefix": v_prefix}
