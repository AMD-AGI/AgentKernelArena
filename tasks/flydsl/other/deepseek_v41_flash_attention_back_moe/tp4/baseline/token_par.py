# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 5e8d913, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/token_par.py and peer_layout.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/token_par.py). Modified for
# AgentKernelArena: imports are package-relative.
"""Token parallel: rank r owns rows r S / TP .. (r + 1) S / TP of an S-row step.

The sublayers' all-reduces become reduce-scatters to each row's owner, the per-row
seam, norm and qkv stages run on the owner alone, and the rows every rank needs (the
MoE input, the q latent, the window record, an index source's normed row and qr)
reach the other ranks through the peer buffer's gather regions. The sums keep the
all-reduce's rank order, so every output is the same to the bit."""

import flydsl.expr as fx

from .plan import pair_layout

from .attn_plan import DATA, Q_RANK
from .seam import HC, HIDDEN

# below this many rows the gathers' extra hop costs more than a rank's few rows save (ROCm/ATOM)
TOKEN_PAR_FROM = 12
QX_WORDS = Q_RANK // 4  # the q latent's e4m3 words
QX_GROUPS = Q_RANK // 32
KG_WORDS = DATA // 4 + 2  # a window record's data words, then its two scale words
LINE = 128  # every peer region starts on a cache line

# a row's bytes in each per-row seam argument (``own_view``)
ROW_BYTES = {
    "res_in": HC * HIDDEN * 2,
    "res_out": HC * HIDDEN * 2,
    "pend": HIDDEN * 2,
    "post_in": HC * 4,
    "pre_in": HC * 4,
    "comb_in": HC * HC * 4,
    "post_out": HC * 4,
    "pre_out": HC * 4,
    "comb_out": HC * HC * 4,
}


def half_pair(n: int, chunk, h: int):
    """Where half ``h`` of a lane's 4-pair chunk goes in a gathered row of ``n`` pairs,
    so each wave store fills whole lines."""
    return h * (n // 2) + 2 * chunk


def chunk_pairs(n: int, q):
    """The first logical pair at a gathered row's 2-pair group ``q`` (inverts ``half_pair``)."""
    return 4 * (q % (n // 4)) + 2 * (q // (n // 4))


def token_par(s: int, tp: int) -> bool:
    """Whether an s-row target step runs token parallel."""
    return tp > 1 and s >= TOKEN_PAR_FROM and s % tp == 0


def peer_layout(s: int, tp: int) -> dict[str, tuple[int, int]]:
    """Return the peer buffer's regions -> (byte offset, bytes), 8 B a (word, tag) pair.

    ATTN and FFN hold the sublayers' partials ``[source rank][row][hidden / 2]``; the
    token-parallel gathers follow, ``[row][..]`` each: XG the MoE input rows, QG / QGS
    the q latent's MXFP8 words / codes, KG the window records, NBG and QRG an index
    source's normed rows and qr, and IWG its index head weights (bf16 pairs)."""
    row = s * HIDDEN // 2
    return pair_layout(
        [
            ("attn", tp * row),
            ("ffn", tp * row),
            ("xg", row),
            ("qg", s * QX_WORDS),
            ("qgs", s * QX_GROUPS),
            ("kg", s * KG_WORDS),
            ("nbg", row),
            ("qrg", s * Q_RANK // 2),
            ("iwg", s * 16),
        ],
        align=LINE,
    )


def peer_offsets(s: int, tp: int) -> dict[str, int]:
    return {region: off for region, (off, _) in peer_layout(s, tp).items()}


def own_view(c: dict, t0, n: int) -> dict:
    """The seam context ``c`` on rows t0 .. t0 + n: S their count, each per-row
    argument moved to row t0."""
    lo = fx.Int64(t0)
    args = {
        k: v + lo * fx.Int64(ROW_BYTES[k]) if k in ROW_BYTES else v
        for k, v in c["args"].items()
    }
    if "aux_out" in args:  # its rows ``aux_stride`` bf16 apart
        args["aux_out"] = args["aux_out"] + lo * fx.Int64(args["aux_stride"]) * 2
    return {**c, "S": n, "args": args}
