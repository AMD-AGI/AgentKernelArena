"""Inputs of the Muse Spark DFlash2 draft attention front (D-pre) that a built-in distribution cannot
express: the RoPE table, the rows' positions and cache slots, and the (stale) fp8 KV cache.

* ``cos_sin``: SGLang's neox rotary cache layout, float32 ``[positions, 128]``: columns 0..63 cos and
  64..127 sin of ``pos * 10000 ** (-i / 64)`` (the base the repository's D-pre test uses; the
  checkpoint's base is not in the source). The table holds 65,536 positions; the draft's table holds
  max_position rows, which does not change the work (one row is read a token).
* ``positions``: each 8-row draft block (one request) sits at consecutive positions ``p + 0 .. p + 7``
  with p uniform in [0, 65536 - 8], int64 as SGLang passes them.
* ``slots``: distinct random cache slots (int64, as SGLang's ``out_cache_loc``).
* ``k_cache`` / ``v_cache``: unit-normal float8_e4m3fn codes; the launch overwrites the rows' slots and
  must leave the others unchanged.

No discrete decision depends on these values, so no margins are needed.
"""

import torch

BLOCK = 8
HEAD_DIM = 128
ROPE_BASE = 10000.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    rows, positions_n, cache_slots = axes["rows"], axes["rope_positions"], axes["cache_slots"]
    g = generator
    half = HEAD_DIM // 2
    inv = ROPE_BASE ** (-torch.arange(half, dtype=torch.float64, device=device) / half)
    angle = torch.arange(positions_n, dtype=torch.float64, device=device)[:, None] * inv[None, :]
    cos_sin = torch.cat([angle.cos(), angle.sin()], dim=1).to(torch.float32)
    starts = torch.randint(0, positions_n - BLOCK + 1, (rows // BLOCK, 1), generator=g, device=device)
    positions = (starts + torch.arange(BLOCK, device=device)[None, :]).reshape(rows).to(torch.int64)
    slots = torch.randperm(cache_slots, generator=g, device=device)[:rows].to(torch.int64)
    k_cache = torch.randn(cache_slots, 1, HEAD_DIM, generator=g, device=device).to(torch.float8_e4m3fn)
    v_cache = torch.randn(cache_slots, 1, HEAD_DIM, generator=g, device=device).to(torch.float8_e4m3fn)
    return {"cos_sin": cos_sin, "positions": positions, "slots": slots, "k_cache": k_cache, "v_cache": v_cache}
