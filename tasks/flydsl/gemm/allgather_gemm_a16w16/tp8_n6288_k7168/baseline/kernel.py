# SPDX-License-Identifier: Apache-2.0
"""The fused all-gather -> bf16 tile GEMM of sgl-amd-mono (``kimi_k31/gfx950/stages/gemm_coll.py::ag_gemm``)
in the arena ``run()`` interface, one rank of the TP8 group: Kimi-K3.1's in_proj shape (6288 rows a rank,
K = 7168).

One persistent 256-CTA launch per rank: every rank pushes its x slice [m, 896] into every rank's ag region
of the peer buffer, and the rank's GEMM tasks (``tile_mma.mma_tile``, K split over ``ks`` CTAs) start as
soon as the owners of their K range have landed; the K slices merge into the zeroed bf16 ``out`` with
packed-bf16 atomic adds ("linbf"), in arrival order.

The launch configuration is ``test/kernels/kimi_k31/ag_gemm_bench.py``'s default ``--cfgs`` (the in_proj
entries of ``kimi_k31/wide.py``'s tuned table at the same rows): a call with m rows takes the smallest of
the bench's 16-, 32-, 128- and 384-row builds that holds m. As the bench does, ``run`` zeroes ``out``
before the launch (linbf adds into it) and passes epoch word 0 = 1, slot 0 of a one-launch step.

Placement: the peer buffer (``coll.peer_bytes(bucket, 7168, "flag")``: two parity halves of rs / ag / flag
regions, then the guard's error flags) is the start of the framework's ``symmetric`` buffer, and the
arrival counter (``cnt``, one int32 that the bench zeroes once) follows it. The framework zeroes the
whole buffer before every call, which stands in for the bench's zeroed-once peer buffer and counter plus
its per-launch epoch bump: a call's mailbox tags never meet a previous call's words.
"""

import functools

import torch

from . import coll
from .gemm_coll import ag_gemm

TP = 8
N = 6288  # a rank's in_proj rows (Kimi-K3.1 KDA in_proj at TP8, 6284 padded to 6288)
K = 7168  # hidden size
# test/kernels/kimi_k31/ag_gemm_bench.py --cfgs default (m:bm:bn:wm:wn:ks), the build's rows -> config
CFGS = {
    16: (16, 128, 1, 8, 4),
    32: (32, 128, 2, 4, 4),
    128: (128, 128, 2, 4, 4),
    384: (192, 128, 4, 2, 2),
}
_ALIGN = 4096  # the counter's alignment past the peer region


def _bucket(m):
    """The smallest bench build that holds m rows."""
    return next(b for b in sorted(CFGS) if b >= m)


@functools.lru_cache(maxsize=None)
def _layout(bucket):
    """(byte offset of the arrival counter in symmetric, bytes used) of a build."""
    cnt = -(-coll.peer_bytes(bucket, K, "flag") // _ALIGN) * _ALIGN
    return cnt, cnt + _ALIGN


def run(*, x, w, epoch, out, rank, world_size, peers, symmetric) -> None:
    """out[m, 6288] (this rank's in_proj rows) = AllGather(x slices) @ w^T; every rank of the group calls it."""
    m = x.shape[0]
    assert int(world_size) == TP and 1 <= m <= max(CFGS), (world_size, m)
    bucket = _bucket(m)
    bm, bn, wm, wn, ks = CFGS[bucket]
    cnt, need = _layout(bucket)
    assert need <= symmetric.numel(), (need, symmetric.numel())
    launch = ag_gemm(bucket, N, K, bm, bn, wm, wn, ks)
    out.zero_()  # linbf: every configuration splits K (ks > 1) and adds its slices into a zeroed out
    launch(
        x.data_ptr(),
        w.data_ptr(),
        out.data_ptr(),
        symmetric.data_ptr() + cnt,
        peers.data_ptr(),
        int(rank),
        epoch.data_ptr(),
        0,
        m,
        stream=torch.cuda.current_stream(),
    )
