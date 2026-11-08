"""Token-granular cache routing for decode: shuffled cache rows, ragged sequence lengths."""
import torch

# The kernel asserts (gl.assume) at least three 64-token iterations per sequence; the wrapper
# requires more than 3 * 64 tokens.
MIN_SEQ_LEN = 193


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, ctx = int(axes["batch"]), int(axes["ctx"])
    # Every sequence owns ctx distinct cache rows, scattered over the whole cache the way a token
    # allocator hands them out after many requests have come and gone.
    rows = torch.randperm(batch * ctx, generator=generator, device=device).to(torch.int32)
    block_tables = rows.reshape(batch, ctx).contiguous()
    # The longest sequence is exactly ctx; the others are between half of it (but at least
    # MIN_SEQ_LEN) and ctx tokens, so partial tiles and short rows occur with a full one.
    low = max(MIN_SEQ_LEN, ctx // 2)
    draw = torch.rand(batch, generator=generator, device=device, dtype=torch.float32)
    lengths = low + (draw * (ctx - low + 1)).to(torch.int64).clamp(max=ctx - low)
    lengths[0] = ctx
    return {"block_tables": block_tables, "seq_lens": lengths.to(torch.int32)}
