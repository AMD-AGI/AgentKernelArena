# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark DFlash2 draft's D-MLP launch (sgl-amd-mono ``muse_spark/gfx950/draft_mlp.py::build_dmlp``)
in the arena ``run()`` interface, one rank of the TP8 group.

One persistent 256-CTA launch runs one draft layer's MLP half for S rows (blocks of 8 rows a request):
the post-attention fused add and RMSNorm, the convolution's kernel projection (the per-token tap
coefficients), the grouped two-tap convolution within each 8-row block (conv0), this rank's gate / up
rows (3584 + 3584 of 28672 + 28672) on bf16 MFMA GEMVs, silu(g) u, this rank's down K shard, the bf16
partial rows pushed into every rank's peer buffer and summed in rank order (the in-kernel all-reduce),
and the finish convolution (conv1) with side 1's coefficients.

The build is the one ``DraftRunner.kernel`` makes (``DMLPBuild(tokens=rows, npes=8)``, all eight
stages). Its arguments are ``DraftRunner.launch``'s, with two differences of placement:

- the peer buffer is the framework's ``symmetric`` buffer (the runner's ``PeerBuffer``, the same
  uncached allocation): its first ``dm_sym_bytes(npes) + FLAG_BYTES`` bytes are the peer region and
  the (unused: D-MLP's guard has no peers) flags;
- the runner's local scratch buffer (``DraftRunner.scratch``, ordinary device memory there) is carved
  out of ``symmetric`` past the peer region. The framework zeroes the whole buffer before every call,
  which stands in for the runner's per-step epoch advance: a call's mailbox tags never meet a previous
  call's words.

``guard`` is the runner's epoch / guard-word buffer (epoch 1, the wait deadline, clean error words).
"""

import functools

import torch

from .draft_mlp import DMLPBuild, build_dmlp, dm_scratch_layout, dm_sym_bytes
from .guard import FLAG_BYTES
from .layout import scratch_bytes

LAYER = 3  # the draft layer index: only the mailbox tags (epoch * LAYER_SLOTS + layer + 1) read it
_ALIGN = 4096


def _round_up(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


PEER_BYTES = dm_sym_bytes(8) + FLAG_BYTES  # the peer region and the flags (TP 8)
SCRATCH_OFFSET = _round_up(PEER_BYTES)  # the runner's scratch, sized for MAX_TOKENS rows
SYMMETRIC_BYTES = SCRATCH_OFFSET + _round_up(scratch_bytes(dm_scratch_layout()))


@functools.lru_cache(maxsize=None)
def _launcher(rows: int, npes: int):
    """``DraftRunner.kernel(rows)``."""
    return build_dmlp(DMLPBuild(tokens=rows, npes=npes))


def run(
    *,
    res, attn_out, w_post, w_kp, base, w_gu, w_down, guard,
    res_out, out,
    rank, world_size, peers, symmetric,
) -> None:  # fmt: skip
    """One rank's D-MLP launch; every rank of the group calls it with its own shard."""
    rows = res.shape[0]
    npes = int(world_size)
    rank = int(rank)
    assert npes == 8 and symmetric.numel() >= SYMMETRIC_BYTES, (npes, symmetric.numel(), SYMMETRIC_BYTES)
    launch = _launcher(rows, npes)
    sym = symmetric.data_ptr()
    launch(
        res.data_ptr(), attn_out.data_ptr(), w_post.data_ptr(), w_kp.data_ptr(), base.data_ptr(),
        w_gu.data_ptr(), w_down.data_ptr(), res_out.data_ptr(), out.data_ptr(), sym + SCRATCH_OFFSET,
        sym, peers.data_ptr(), rank, guard.data_ptr(), LAYER,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
