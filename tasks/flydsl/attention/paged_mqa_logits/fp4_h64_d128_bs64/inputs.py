"""Decode-step inputs of the FP4 (MXFP4) indexer: packed fp4 queries and a scattered 64-token-block fp4 cache with
E8M0 block scales, ragged contexts, the persistent-grid schedule, and the AITER kernel layout of queries scales
and cache blocks."""
import torch

HEAD_SIZE = 128
GROUP = 32  # elements per E8M0 scale (MXFP4 block)
PACKED = HEAD_SIZE // 2
GROUPS = HEAD_SIZE // GROUP
BLOCK_SIZE = 64
BLOCK_K = 256  # keys per CTA step of the kernel
MFMA_M = 16
MIN_CTX = 128
FP4_MAX = 6.0
_QUANT_BLOCKS = 8192  # cache blocks quantized per step, bounding the float32 temporaries
_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # e2m1 magnitudes; the code is the index, bit 3 the sign
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_mxfp4(values):
    """Packed e2m1 codes [rows, HEAD_SIZE // 2] and E8M0 scales [rows, GROUPS] of a float32 [rows, HEAD_SIZE] tensor.

    Per 32 elements: scale = 2 ** ceil(log2(amax / 6)) stored as a biased exponent, then round to nearest e2m1.
    Element 2i is the low nibble of byte i, element 2i + 1 the high nibble.
    """
    rows = values.shape[0]
    groups = values.reshape(rows, GROUPS, GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    biased = (torch.ceil(torch.log2(amax / FP4_MAX)) + 127.0).clamp(0.0, 255.0)
    scaled = groups / torch.exp2(biased - 127.0)
    midpoints = torch.tensor(_MIDPOINTS, dtype=torch.float32, device=values.device)
    magnitude = torch.bucketize(scaled.abs().contiguous(), midpoints).to(torch.uint8)
    nibble = torch.where(magnitude == 0, torch.zeros_like(magnitude), magnitude | ((scaled < 0).to(torch.uint8) << 3))
    nibble = nibble.reshape(rows, HEAD_SIZE)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    return packed.contiguous(), biased.reshape(rows, GROUPS).to(torch.uint8)


def _normal_bf16(shape, generator, device):
    """Standard normal activations as the model holds them (bfloat16), widened for quantization."""
    return torch.randn(shape, generator=generator, device=device).to(torch.bfloat16).float()


def build_schedule(context_lens, total_ctas, next_n, max_seq_len):
    """The persistent-grid schedule of AITER's compute_varctx_schedule: one [batch_packed, chunk_start,
    chunk_count, ctx_len] row per CTA (a torch port of its Triton _varctx_cta_info_kernel)."""
    device = context_lens.device
    batch = context_lens.shape[0]
    chunks = (context_lens.long() + BLOCK_K - 1) // BLOCK_K
    s_max = max(1, (max_seq_len + BLOCK_K - 1) // BLOCK_K)
    slots = total_ctas // next_n
    lo, hi = 1, s_max
    for _ in range(32):
        mid = (lo + hi) // 2
        feasible = int(((chunks + mid - 1) // mid).sum()) * next_n <= total_ctas
        if lo < hi:
            if feasible:
                hi = mid
            else:
                lo = mid + 1
    fits = int(((chunks + s_max - 1) // s_max).sum()) * next_n <= total_ctas
    safe = lo if fits else max(int(chunks.max()), 1)
    per_batch = (chunks + safe - 1) // safe
    inclusive = torch.cumsum(per_batch, dim=0)
    exclusive = inclusive - per_batch
    total_splits = int(per_batch.sum())
    slot = torch.arange(slots, device=device)
    owner = (inclusive[None, :] <= slot[:, None]).sum(dim=1).clamp(max=batch - 1)
    valid = slot < total_splits
    start = (slot - exclusive[owner]) * safe
    count = (chunks[owner] - start).clamp(min=0).clamp(max=safe)
    packed_batch = torch.where(valid, owner, torch.zeros_like(owner))
    start = torch.where(valid, start, torch.zeros_like(start))
    count = torch.where(valid, count, torch.ones_like(count))
    ctx = torch.where(valid, context_lens.long()[owner], torch.zeros_like(owner))
    rows = []
    for n in range(next_n):
        batch_packed = torch.where(valid, packed_batch * next_n + n, torch.zeros_like(owner))
        rows.append(torch.stack([batch_packed, start, count, ctx], dim=1))
    return torch.stack(rows, dim=1).reshape(slots * next_n, 4).to(torch.int32).contiguous()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, next_n, ctx = int(axes["batch_size"]), int(axes["next_n"]), int(axes["ctx"])
    heads, num_blocks = int(axes["num_heads"]), int(axes["num_blocks"])
    blocks_per_seq = int(axes["blocks_per_seq"])
    max_seq_len, total_ctas = int(axes["max_seq_len"]), int(axes["total_ctas"])

    q_packed, q_scale = quantize_mxfp4(_normal_bf16((batch * next_n * heads, HEAD_SIZE), generator, device))

    # The longest sequence has exactly ctx keys, the others lie between max(MIN_CTX, ctx / 2) and ctx.
    low = max(MIN_CTX, ctx // 2)
    draw = torch.rand(batch, generator=generator, device=device)
    lengths = low + (draw * (ctx - low + 1)).to(torch.int64).clamp(max=ctx - low)
    lengths[0] = ctx
    lengths = lengths.to(torch.int32)

    # Distinct, randomly placed cache blocks per sequence; unlisted entries (and the guard pages) are 0.
    placement = torch.randperm(num_blocks, generator=generator, device=device).to(torch.int32)
    listed = (torch.arange(blocks_per_seq, device=device)[None, :] * BLOCK_SIZE) < lengths[:, None]
    table = torch.zeros(batch, int(axes["max_blocks"]), dtype=torch.int32, device=device)
    table[:, :blocks_per_seq] = torch.where(listed, placement.reshape(batch, blocks_per_seq),
                                            torch.zeros_like(listed, dtype=torch.int32))

    kv_packed = torch.empty(num_blocks, BLOCK_SIZE, PACKED, dtype=torch.uint8, device=device)
    kv_scale = torch.empty(num_blocks, BLOCK_SIZE, GROUPS, dtype=torch.uint8, device=device)
    for start in range(0, num_blocks, _QUANT_BLOCKS):
        stop = min(num_blocks, start + _QUANT_BLOCKS)
        packed, scale = quantize_mxfp4(_normal_bf16(((stop - start) * BLOCK_SIZE, HEAD_SIZE), generator, device))
        kv_packed[start:stop] = packed.reshape(stop - start, BLOCK_SIZE, PACKED)
        kv_scale[start:stop] = scale.reshape(stop - start, BLOCK_SIZE, GROUPS)
    return {
        "q_fp4": q_packed.reshape(batch, next_n, heads, PACKED),
        "q_scale": q_scale.reshape(batch, next_n, heads, GROUPS),
        "kv_cache": kv_packed,
        "kv_scale": kv_scale,
        "context_lens": lengths,
        "block_tables": table,
        "cta_info": build_schedule(lengths, total_ctas, next_n, max_seq_len),
    }


def to_kernel_layout(case, inputs):
    """Natural layouts -> the kernel ABI (aiter's create_paged_preshuffle_kv_fp4 and the preshuffled q scales)."""
    inputs = dict(inputs)
    batch, next_n, heads = inputs["q_scale"].shape[:3]
    tiles = heads // MFMA_M
    pad = (tiles + 3) // 4 * 4
    # q scales [batch, next_n, heads, groups] -> [batch, next_n, k_tiles, 4 K chunks, 16 rows, m tiles padded].
    q_scale = (inputs["q_scale"].reshape(batch, next_n, tiles, MFMA_M, HEAD_SIZE // 128, 4)
               .permute(0, 1, 4, 5, 3, 2).contiguous())
    inputs["q_scale"] = torch.nn.functional.pad(q_scale, (0, pad - tiles)).contiguous()

    packed, scale = inputs["kv_cache"], inputs["kv_scale"]
    num_blocks = packed.shape[0]
    k_tiles = HEAD_SIZE // 128
    cache = torch.empty(num_blocks, k_tiles, 4, BLOCK_SIZE, 16, dtype=torch.uint8, device=packed.device)
    block_scale = torch.empty(num_blocks, k_tiles, 4, BLOCK_SIZE, dtype=torch.uint8, device=packed.device)
    ntpw = 4  # four 16-token tiles of a block are read by one thread, so their scale bytes are interleaved
    for start in range(0, num_blocks, _QUANT_BLOCKS):
        stop = min(num_blocks, start + _QUANT_BLOCKS)
        n = stop - start
        cache[start:stop] = packed[start:stop].reshape(n, BLOCK_SIZE, k_tiles, 4, 16).permute(0, 2, 3, 1, 4)
        by_token = scale[start:stop].reshape(n, BLOCK_SIZE, k_tiles, 4).permute(0, 2, 3, 1)
        block_scale[start:stop] = (by_token.reshape(n, k_tiles, 4, ntpw, BLOCK_SIZE // ntpw)
                                   .transpose(-1, -2).reshape(n, k_tiles, 4, BLOCK_SIZE))
    inputs["kv_cache"] = cache
    inputs["kv_scale"] = block_scale
    return inputs
