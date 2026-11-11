"""Inputs of one DSpark verify step of the DeepSeek-V4.1-Flash FP4 indexer: MXFP4 queries of R requests x 6
rows, SGLang's FP4 index plane (64-row pages, payload and E8M0 scales apart) with each request's pages
scattered over it, the per-row page table, the rows' visible-column bounds and their requests' first rows."""
import torch

HEADS = 32
DIM = 128
GROUP = 32  # elements per E8M0 scale (MXFP4 block)
CHUNKS = DIM // GROUP  # the scorer's K chunks (16 bytes of E2M1 a chunk)
PAGE = 64  # rows of an index-plane page
TOPK = 512  # a row that sees at most this many columns is not scored (SGLang's top-k takes them all)
FP4_MAX = 6.0
_QUANT_PAGES = 8192  # plane pages quantized per step, bounding the float32 temporaries
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # between the e2m1 magnitudes 0 .. 6


def quantize_mxfp4(values):
    """Packed e2m1 codes [n, DIM // 2] and E8M0 scales [n, CHUNKS] of a float32 [n, DIM] tensor.

    Per 32 elements: scale = 2 ** ceil(log2(amax / 6)) stored as a biased exponent, then round to
    nearest e2m1 (the code is the magnitude's index, bit 3 the sign). Element 2i is the low nibble of
    byte i, element 2i + 1 the high nibble. (The quantizer of the sibling task fp4_h64_d128_bs64.)
    """
    n = values.shape[0]
    groups = values.reshape(n, CHUNKS, GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    biased = (torch.ceil(torch.log2(amax / FP4_MAX)) + 127.0).clamp(0.0, 255.0)
    scaled = groups / torch.exp2(biased - 127.0)
    midpoints = torch.tensor(_MIDPOINTS, dtype=torch.float32, device=values.device)
    magnitude = torch.bucketize(scaled.abs().contiguous(), midpoints).to(torch.uint8)
    nibble = torch.where(magnitude == 0, torch.zeros_like(magnitude), magnitude | ((scaled < 0).to(torch.uint8) << 3))
    nibble = nibble.reshape(n, DIM)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    return packed.contiguous(), biased.reshape(n, CHUNKS).to(torch.uint8)


def _normal_bf16(shape, generator, device):
    """Standard normal activations as the model holds them (bfloat16), widened for quantization."""
    return torch.randn(shape, generator=generator, device=device).to(torch.bfloat16).float()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    requests, ctx = int(axes["num_requests"]), int(axes["ctx"])
    per_req, ppr = int(axes["rows_per_request"]), int(axes["pages_per_request"])
    rows, num_pages = requests * per_req, requests * ppr

    # queries [rows, 32, 64]; their scales in the kernel layout [rows, 1, 4 (K chunk), 16 (head % 16),
    # 4 (head // 16, padded with 127)]
    q_packed, q_e8m0 = quantize_mxfp4(_normal_bf16((rows * HEADS, DIM), generator, device))
    q_scale = torch.full((rows, 1, CHUNKS, 16, 4), 127, dtype=torch.uint8, device=device)
    q_scale[:, 0, :, :, : HEADS // 16] = q_e8m0.reshape(rows, HEADS // 16, 16, CHUNKS).permute(0, 3, 2, 1)

    # request 0 sees exactly ctx columns on its last row, the others between max(TOPK + 6, ctx / 2) and ctx;
    # row i of a request sees one column fewer than row i + 1 (its draft position is one earlier), so every
    # row sees more than TOPK columns and is scored
    low = max(TOPK + per_req, ctx // 2)
    draw = torch.rand(requests, generator=generator, device=device)
    lengths = low + (draw * (ctx - low + 1)).to(torch.int64).clamp(max=ctx - low)
    lengths[0] = ctx
    step = torch.arange(per_req, device=device)
    bounds = (lengths[:, None] - (per_req - 1 - step)[None, :]).reshape(rows).to(torch.int32)
    owners = (torch.arange(rows, device=device) // per_req * per_req).to(torch.int32)

    # each request owns ppr distinct pages scattered over the plane; its rows share one table row (SGLang's
    # per-row page table); entries past the request's own pages are 0
    placement = torch.randperm(num_pages, generator=generator, device=device).to(torch.int32).reshape(requests, ppr)
    listed = (torch.arange(ppr, device=device)[None, :] * PAGE) < lengths[:, None]
    table = torch.where(listed, placement, torch.zeros_like(placement))
    page_table = table.repeat_interleave(per_req, dim=0).contiguous()

    # SGLang's FP4 index plane: payload [pages, 1, 4 (K chunk), 64 (row), 16 B], scales [pages, 1, 4 (K chunk),
    # 64] with row 16 j + l's scale at (row % 16) * 4 + row // 16
    kv_cache = torch.empty(num_pages, 1, CHUNKS, PAGE, 16, dtype=torch.uint8, device=device)
    kv_scale = torch.empty(num_pages, 1, CHUNKS, PAGE, dtype=torch.uint8, device=device)
    for start in range(0, num_pages, _QUANT_PAGES):
        stop = min(num_pages, start + _QUANT_PAGES)
        n = stop - start
        packed, e8m0 = quantize_mxfp4(_normal_bf16((n * PAGE, DIM), generator, device))
        kv_cache[start:stop, 0] = packed.reshape(n, PAGE, CHUNKS, 16).permute(0, 2, 1, 3)
        by_row = e8m0.reshape(n, PAGE, CHUNKS).permute(0, 2, 1)  # [n, chunk, row = 16 j + l]
        kv_scale[start:stop, 0] = by_row.reshape(n, CHUNKS, 4, 16).transpose(-1, -2).reshape(n, CHUNKS, PAGE)
    return {
        "q_fp4": q_packed.reshape(rows, HEADS, DIM // 2),
        "q_scale": q_scale,
        "kv_cache": kv_cache,
        "kv_scale": kv_scale,
        "page_table": page_table,
        "bounds": bounds,
        "owners": owners,
    }
