"""Float64 SiLU-and-mul with exact MXFP4 quantization and sorted scale tiles, used only for correctness."""
import torch

GROUP = 32
ROW_CHUNK = 1 << 16
GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)          # E2M1 magnitudes by code
MIDPOINTS = tuple((GRID[i] + GRID[i + 1]) / 2 for i in range(7))


def _quantize(x):
    """x [R, 2 * I] bfloat16 -> (packed codes [R, I // 2] uint8, E8M0 scale codes [R, I // 32] uint8)."""
    rows, width = x.shape
    inter = width // 2
    gate, up = x[:, :inter].double(), x[:, inter:].double()
    act = gate * torch.sigmoid(gate) * up
    blocks = act.reshape(rows, inter // GROUP, GROUP)
    peak = blocks.abs().amax(dim=-1)
    exponent = torch.floor(torch.log2(peak.clamp(min=1e-300)))
    exponent = exponent + (peak / torch.exp2(exponent) >= 1.5)      # round the exponent at mantissa 1.5
    biased = torch.where(peak > 0, exponent + 127 - 2, torch.zeros_like(exponent)).clamp(0, 255)
    scaled = blocks * torch.exp2(127.0 - biased).unsqueeze(-1)
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int64)
    for index, mid in enumerate(MIDPOINTS):                          # round to nearest, ties to the even code
        code += (magnitude > mid) | ((magnitude == mid) & (index % 2 == 1))
    code = code | (torch.signbit(scaled).long() << 3)
    code = code.reshape(rows, inter).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, biased.to(torch.uint8)


def reference(*, x, sorted_ids, num_valid_ids, token_num):
    rows, width = x.shape
    inter = width // 2
    top_k = rows // token_num
    codes = torch.empty(rows, inter // 2, dtype=torch.uint8, device=x.device)
    scale_rows = torch.empty(rows, inter // GROUP, dtype=torch.uint8, device=x.device)
    for start in range(0, rows, ROW_CHUNK):
        codes[start:start + ROW_CHUNK], scale_rows[start:start + ROW_CHUNK] = _quantize(x[start:start + ROW_CHUNK])
    # Scales follow the sorted order; sentinel rows (token >= token_num or beyond num_valid_ids) hold zero.
    ids = sorted_ids.to(torch.int64)
    token, slot = ids & 0xFFFFFF, ids >> 24
    valid = (token < token_num) & (slot < top_k) & (torch.arange(ids.numel(), device=x.device) < num_valid_ids[0])
    source = torch.where(valid, token * top_k + slot, torch.zeros_like(token))
    sorted_rows = torch.where(valid[:, None], scale_rows[source], torch.zeros((), dtype=torch.uint8, device=x.device))
    length, groups = sorted_rows.shape
    tiles = sorted_rows.reshape(length // 32, 2, 16, groups // 8, 2, 4)      # R, a, m, Cb, c, n
    tiles = tiles.permute(0, 3, 5, 2, 4, 1)                                    # R, Cb, n, m, c, a
    return codes, tiles.reshape(-1).contiguous()
