# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark K3 launch (sgl-amd-mono ``muse_spark/gfx950/build_k3.py::build_k3``) in the arena
``run()`` interface, one rank of the TP8 / EP8 group with TP experts.

One persistent 256-CTA launch takes the step's routing (K2's ids and scores) and expert input through
the next layer's attention input: the expert input quantized to MXFP4 (aiter's ``_mxfp4_quant_op``
rule), gate / up on the scaled E2M1 x E2M1 MFMA against this rank's slices of every picked expert,
bf16(silu(gate) up) re-quantized to MXFP4, the mids exchanged with every rank, the down projection's
output columns of this rank, the finalize (per-(token, slot) post norm with the input scale, the
score-weighted slot sum) over values pushed by every rank, the unpooler rows of this rank pushed to
every rank, and the FFN boundary (post norm, the gated fp32 residual stream updated in place, the next
layer's pre-attention norm) on every rank.

The build is the one ``MuseMonoRunner.k3_kernel`` makes with the runner's default environment
(``SGLANG_AMD_MONO_MUSE_TPX=1``: TP experts; ``..._MUSE_FP8_DENSE`` off, so a bf16 unpooler): the
standalone K3 launch of the runner's non-merged path (no ``xq_pre`` / ``ug_dyn``, which only the
merged launch uses), for a middle layer (``last`` = 0). Its arguments are ``MuseMonoRunner.launch``'s,
with two differences of placement:

- the peer buffer is the framework's ``symmetric`` buffer (the runner's ``PeerBuffer``, the same
  uncached allocation): its first ``sym_bytes(npes) + FLAG_BYTES`` bytes are the peer regions and the
  guard's flags;
- the runner's local K3 scratch buffer (``scratch_k3``, ordinary device memory there) is carved out of
  ``symmetric`` past the peer regions. The framework zeroes the whole buffer before every call, which
  stands in for the runner's per-step epoch advance: a call's mailbox tags never meet a previous call's
  words.

``guard`` is the runner's epoch / guard-word buffer (epoch 1, the wait deadline, clean error words).
"""

import functools

import torch

from .build_k3 import K3Build, build_k3
from .config import LOCAL_EXPERTS
from .ffn import k3_scratch_layout, k3_sym_layout
from .guard import FLAG_BYTES
from .layout import scratch_bytes

LAYER = 9  # the layer index: only the mailbox tags (epoch * LAYER_SLOTS + layer + 1) read it
LAST = 0  # a middle layer: the last-layer boundary rule is off
_ALIGN = 4096


def _round_up(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


PEER_BYTES = scratch_bytes(k3_sym_layout(8)) + FLAG_BYTES  # the peer regions and the guard's flags (TP 8)
SCRATCH_OFFSET = _round_up(PEER_BYTES)  # the runner's scratch_k3, sized for MAX_TOKENS rows
SYMMETRIC_BYTES = SCRATCH_OFFSET + _round_up(scratch_bytes(k3_scratch_layout()))


@functools.lru_cache(maxsize=None)
def _launcher(rows: int, npes: int):
    """``MuseMonoRunner.k3_kernel(rows)`` with the runner's defaults."""
    return build_k3(K3Build(tokens=rows, npes=npes, last=False, fp8_unpool=False, tpx=True))


def run(
    *,
    xin, ids, scores, w13, s13, w2, s2, pen_w, res, pfw, alpha, beta, npw, w_unpool, guard,
    attn_out,
    rank, world_size, peers, symmetric,
) -> None:  # fmt: skip
    """One rank's K3 launch; every rank of the group calls it with its own shard."""
    rows = xin.shape[0]
    npes = int(world_size)
    rank = int(rank)
    assert npes == 8 and symmetric.numel() >= SYMMETRIC_BYTES, (npes, symmetric.numel(), SYMMETRIC_BYTES)
    launch = _launcher(rows, npes)
    sym = symmetric.data_ptr()
    launch(
        # local_start: the runner passes the layer's EP start (32 rank); TP experts do not read it
        xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), rank * LOCAL_EXPERTS,
        w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(), pen_w.data_ptr(), res.data_ptr(),
        pfw.data_ptr(), alpha.data_ptr(), beta.data_ptr(), npw.data_ptr(), w_unpool.data_ptr(),
        attn_out.data_ptr(), sym + SCRATCH_OFFSET, sym, peers.data_ptr(), rank, guard.data_ptr(), LAYER, LAST,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
