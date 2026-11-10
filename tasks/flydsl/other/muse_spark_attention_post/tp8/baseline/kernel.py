# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark K2 launch (sgl-amd-mono ``muse_spark/gfx950/build_k2.py::build_k2``) in the arena
``run()`` interface, one rank of the TP8 / EP8 group.

One persistent 256-CTA launch takes this rank's attention output through the experts' input: the
per-head output norm-gate of its 16 query heads, its o_proj K shard (bf16 MFMA GEMV), the bf16 partial
rows pushed into every rank's peer buffer and summed in rank order (the in-kernel all-reduce), the
attention boundary (weightless post norm, the gated fp32 residual stream updated in place, the
pre-FFN norm), the sigmoid router's top-8 of 256 experts on the bf16 hi + lo router, the pooler
(8192 -> 4096; at S <= 16 rows each rank computes 1/8 of the pooled rows and pushes them to every
rank) and the pre-expert norm.

The build is the one ``MuseMonoRunner.k2_kernel`` makes with the runner's default environment
(``SGLANG_AMD_MONO_MUSE_PSHARD=1``, ``..._MUSE_FP8_DENSE`` off, so a bf16 pooler, row-sharded at
S <= 16 only): the standalone K2 launch of the runner's non-merged path. Its arguments are
``MuseMonoRunner.launch``'s, with two differences of placement:

- the peer buffer is the framework's ``symmetric`` buffer (the runner's ``PeerBuffer``, the same
  uncached allocation): its first ``sym_bytes(npes) + FLAG_BYTES`` bytes are the peer regions and the
  guard's flags;
- the runner's local K2 scratch buffer (``scratch_k2``, ordinary device memory there) is carved out of
  ``symmetric`` past the peer regions. The framework zeroes the whole buffer before every call, which
  stands in for the runner's per-step epoch advance: a call's mailbox tags never meet a previous call's
  words.

``guard`` is the runner's epoch / guard-word buffer (epoch 1, the wait deadline, clean error words).
"""

import functools

import torch

from .build_k2 import K2Build, build_k2
from .config import MOE_HIDDEN
from .guard import FLAG_BYTES
from .layout import scratch_bytes
from .post_attn import k2_scratch_layout, sym_bytes

LAYER = 9  # the layer index: only the mailbox tags (epoch * LAYER_SLOTS + layer + 1) read it
_ALIGN = 4096


def _round_up(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


PEER_BYTES = sym_bytes(8) + FLAG_BYTES  # the peer regions and the guard's flags (TP 8)
SCRATCH_OFFSET = _round_up(PEER_BYTES)  # the runner's scratch_k2, sized for MAX_TOKENS rows
SYMMETRIC_BYTES = SCRATCH_OFFSET + _round_up(scratch_bytes(k2_scratch_layout()))


def _pshard(rows: int, npes: int) -> bool:
    """``MuseMonoRunner._pshard``: the row-sharded pooler at TP 8 and S <= 16 rows only."""
    return npes == 8 and rows <= 16


@functools.lru_cache(maxsize=None)
def _launcher(rows: int, npes: int):
    """``MuseMonoRunner.k2_kernel(rows)`` with the runner's defaults."""
    return build_k2(
        K2Build(
            tokens=rows,
            npes=npes,
            fp8_pool=False,
            defer_k1=False,
            xq_pre=False,
            pshard=_pshard(rows, npes),
        )
    )


def run(
    *,
    o, g, res, w_o, alpha, beta, pre_w, router_hi, router_lo, expert_bias, w_pool, xn_w, guard,
    ids, scores, xin,
    rank, world_size, peers, symmetric,
) -> None:  # fmt: skip
    """One rank's K2 launch; every rank of the group calls it with its own shard."""
    rows = o.shape[0]
    npes = int(world_size)
    rank = int(rank)
    assert npes == 8 and symmetric.numel() >= SYMMETRIC_BYTES, (npes, symmetric.numel(), SYMMETRIC_BYTES)
    launch = _launcher(rows, npes)
    # MuseMonoRunner._pool_w: the rank's 1 / npes of the pooler rows when sharded, else the whole pooler
    rows_a_rank = MOE_HIDDEN // npes
    pool = w_pool[rank * rows_a_rank : (rank + 1) * rows_a_rank] if _pshard(rows, npes) else w_pool
    sym = symmetric.data_ptr()
    launch(
        o.data_ptr(), g.data_ptr(), res.data_ptr(), w_o.data_ptr(), alpha.data_ptr(), beta.data_ptr(),
        pre_w.data_ptr(), router_hi.data_ptr(), router_lo.data_ptr(), expert_bias.data_ptr(), pool.data_ptr(),
        xn_w.data_ptr(), ids.data_ptr(), scores.data_ptr(), xin.data_ptr(), sym + SCRATCH_OFFSET,
        sym, peers.data_ptr(), rank, guard.data_ptr(), LAYER,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
