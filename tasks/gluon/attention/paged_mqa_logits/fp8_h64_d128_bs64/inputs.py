"""Decode-step inputs of the DeepSeek-V3.2 indexer (fp8 queries, a scattered 64-token-block fp8 cache, ragged
contexts) and the AITER (16, 16) preshuffle of the cache blocks."""
import torch

FP8_MAX = 448.0  # float8_e4m3fn saturation magnitude
HEAD_SIZE = 128
SOFTMAX_SCALE = HEAD_SIZE ** -0.5
MIN_CTX = 128  # the axis minimum; shorter contexts would leave rows with too few elements to bound
_QUANT_CHUNK = 1 << 19  # cache tokens quantized per step, bounding the float32 temporaries


def _quantize_rows(values):
    """fp8 codes and float32 scales (row amax / 448) of the rows of a float32 [..., HEAD_SIZE] tensor."""
    scale = values.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / scale).to(torch.float8_e4m3fn), scale


def _normal_bf16(shape, generator, device):
    """Standard normal activations as the model holds them (bfloat16), widened for quantization."""
    return torch.randn(shape, generator=generator, device=device).to(torch.bfloat16).float()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, next_n, ctx = int(axes["batch_size"]), int(axes["next_n"]), int(axes["ctx"])
    heads, block_size = int(axes["num_heads"]), int(axes["block_size"])
    max_model_len = int(axes["max_model_len"])
    num_blocks = int(axes["num_blocks"])
    blocks_per_seq = -(-ctx // block_size)

    # Query rows are quantized per head and the dequantization scale joins the head weight, as the model's
    # indexer does: weights = weights_proj(x) * q_scale * head_size**-0.5 * heads**-0.5. The projection
    # output is drawn with a positive mean so the summed score has the nonzero mean of a trained indexer.
    q_codes, q_scale = _quantize_rows(_normal_bf16((batch, next_n, heads, HEAD_SIZE), generator, device))
    head_weight = 0.5 + 0.5 * torch.randn(batch, next_n, heads, generator=generator, device=device)
    weights = head_weight * q_scale.squeeze(-1) * (SOFTMAX_SCALE * heads ** -0.5)

    # Context lengths: the longest sequence has exactly ctx tokens, the others lie between max(MIN_CTX, ctx / 2)
    # and ctx, so partial chunks and uneven loads occur in every case.
    low = max(MIN_CTX, ctx // 2)
    draw = torch.rand(batch, generator=generator, device=device)
    lengths = low + (draw * (ctx - low + 1)).to(torch.int64).clamp(max=ctx - low)
    lengths[0] = ctx

    # Each sequence owns distinct, randomly placed cache blocks; unlisted table entries are 0 as in a
    # serving block table.
    placement = torch.randperm(batch * blocks_per_seq, generator=generator, device=device).to(torch.int32)
    listed = (torch.arange(blocks_per_seq, device=device)[None, :] * block_size) < lengths[:, None]
    table = torch.zeros(batch, max_model_len // block_size, dtype=torch.int32, device=device)
    table[:, :blocks_per_seq] = torch.where(listed, placement.reshape(batch, blocks_per_seq),
                                            torch.zeros_like(listed, dtype=torch.int32))

    # Cache block: block_size * HEAD_SIZE fp8 codes, then block_size float32 scales (amax / 448 per token).
    row_bytes = block_size * (HEAD_SIZE + 4)
    cache = torch.empty(num_blocks, row_bytes, dtype=torch.uint8, device=device)
    step = max(1, _QUANT_CHUNK // block_size)
    for start in range(0, num_blocks, step):
        stop = min(num_blocks, start + step)
        codes, scale = _quantize_rows(_normal_bf16((stop - start, block_size, HEAD_SIZE), generator, device))
        cache[start:stop, : block_size * HEAD_SIZE] = codes.reshape(stop - start, -1).view(torch.uint8)
        cache[start:stop, block_size * HEAD_SIZE:] = scale.reshape(stop - start, block_size).view(torch.uint8) \
            .reshape(stop - start, -1)
    return {
        "q": q_codes,
        "kv_cache": cache.view(num_blocks, block_size, 1, HEAD_SIZE + 4),
        "weights": weights.reshape(batch * next_n, heads).contiguous(),
        "context_lens": lengths.to(torch.int32),
        "block_tables": table,
    }


def preshuffle_block_data(data):
    """AITER shuffle_weight(x, layout=(16, 16)) for one-byte elements, per cache block.

    data [n, block_size * HEAD_SIZE] holds each block's token rows [block_size, HEAD_SIZE]. The rows are cut
    into tiles of 16 tokens x 32 bytes, and inside a block the byte order becomes (token group of 16,
    32-byte column tile, 16-byte half, token of the group, byte of the half).
    """
    n = data.shape[0]
    block_size = data.shape[1] // HEAD_SIZE
    tiles = data.contiguous().view(n, block_size // 16, 16, HEAD_SIZE // 32, 2, 16)
    return tiles.permute(0, 1, 3, 4, 2, 5).reshape(n, block_size * HEAD_SIZE)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    cache = inputs["kv_cache"]
    num_blocks, block_size = cache.shape[0], cache.shape[1]
    rows = cache.view(num_blocks, block_size * (HEAD_SIZE + 4))
    step = max(1, _QUANT_CHUNK // block_size)
    for start in range(0, num_blocks, step):
        data = rows[start:start + step, : block_size * HEAD_SIZE]
        data.copy_(preshuffle_block_data(data))
    return inputs
