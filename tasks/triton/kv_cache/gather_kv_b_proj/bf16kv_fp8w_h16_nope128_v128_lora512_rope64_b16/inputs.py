"""Block-quantized fp8 kv_b projection, request partition of the prefix and paged-cache routing."""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude
_BLOCK = 128      # weight scale granularity along n and k


def make_inputs(case, inputs, generator, device):
    """Weight W ~ N(0, 0.03^2) quantized per 128 x 128 block with scale = block amax / 448, as the checkpoint stores it.

    The batch's total_kv tokens are split into `batch` requests of unequal length (every request has at least one
    token); the cache rows of all tokens are distinct random blocks of the pool, so requests never share a row.
    """
    axes = case["axes"]
    batch, total = axes["batch"], axes["total_kv"]
    rows = axes["num_heads"] * (axes["qk_nope_head_dim"] + axes["v_head_dim"])
    k = axes["kv_lora_dim"]
    w = 0.03 * torch.randn(rows, k, generator=generator, device=device, dtype=torch.float32)
    blocks = w.view(rows // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-10)
    scale = amax / _FP8_MAX
    codes = (blocks / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

    share = 0.5 + torch.rand(batch, generator=generator, device=device, dtype=torch.float64)
    lens = 1 + torch.floor((total - batch) * share / share.sum()).to(torch.int64)
    lens[torch.argmax(share)] += total - int(lens.sum())
    indptr = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(lens, dim=0)
    slots = torch.randperm(axes["num_blocks"], generator=generator, device=device)[:total]
    return {
        "kv_proj_weight": codes.reshape(rows, k),
        "kv_proj_scale": scale.contiguous(),
        "kv_indptr": indptr.to(torch.int32),
        "kv_prefix_sum_context_lens": indptr.to(torch.int32),
        "kv_indices": slots.to(torch.int32),
    }


def to_kernel_layout(case, inputs):
    """Preshuffle the fp8 weight like aiter.ops.shuffle.shuffle_weight(layout=(16, 16)) for one-byte elements.

    Logical W[16 * i + r, 32 * j + 16 * h + c] sits at flat position 512 * (kblocks * i + j) + 256 * h + 16 * r + c.
    """
    inputs = dict(inputs)
    w = inputs["kv_proj_weight"]
    n, k = w.shape
    shuffled = w.view(torch.uint8).reshape(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
    inputs["kv_proj_weight"] = shuffled.reshape(n, k).view(torch.float8_e4m3fn)
    return inputs
