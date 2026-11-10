# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark K-attn launch (sgl-amd-mono ``muse_spark/gfx950/layer.py::build_kattn``, stage code
``stages/attention.py::run_kattn``) in the arena ``run()`` interface: one full-attention layer's
verify attention on one TP8 rank (16 query heads over 2 KV heads, head dim 64) over the fp8 page-16
NHD cache, for ``requests`` x 8 rows.

One persistent launch of 256 CTAs x 512 threads: split-KV attention tasks (request, KV head, two
row tiles, partition of 256-key chunks), each publishing its rows' partial (m, l, O) with a mailbox
flag, then one- or two-level merge jobs that fold the partitions into ``out``.

The build and the arguments are the runner's (``MuseMonoRunner.kattn_kernel`` and ``_attn_args``,
gfx950/runner.py) for a target-verify step of a full-attention layer: ``KAttnBuild(rows, 8)``,
``extra`` = 8 (the step's rows the kernel adds to ``seq_lens``), window 0, the partition sized by the
kernel, the layer's softmax scale 64 ** -0.5. Placement differs from the runner, host-side only:

- the runner keeps one scratch for every launch and moves the step epoch on after each step, so a
  mailbox flag of an earlier step never carries the current tag. Here the epoch stays 0 (tag 1) and
  ``run()`` zeroes the scratch's flag span (the task flags, the level-2 records and their flags) before
  every launch; the partial-record region is zeroed once, when the buffer is created, and afterwards
  holds only finite records this kernel wrote, which the merges weigh by zero when they skip them;
- the scratch and the epoch / guard words (word 0 the epoch, words 1-2 the wait deadline) share one
  device buffer per row count, created at the first (eager) call and reused, as the runner keeps its
  scratch.
"""

import functools

import torch

from .guard import GUARD_DEADLINE, deadline_ticks
from .layer import KAttnBuild, build_kattn
from .layout import kattn_scratch_layout, scratch_bytes

Q_LEN = 8  # rows a request: the DFlash2 verify's 8 draft tokens
LAYER = 0  # the mailbox tag is epoch * LAYER_SLOTS + layer + 1 = 1 (zeroed flags carry tag 0)
SCALE_BITS = 0x3E000000  # float32 bits of the softmax scale 64 ** -0.5 = 0.125 (RadixAttention.scaling)
GUARD_BYTES = 256  # the epoch / guard words, then the scratch (256-byte aligned)
DEADLINE_S = 30.0  # a wait that never completes stops the launch
_BUFFERS = {}  # (rows, device index) -> (buffer, its flag span)


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    return build_kattn(KAttnBuild(tokens=rows, q_len=Q_LEN))


def _buffers(rows: int, device):
    key = (rows, device.index)
    found = _BUFFERS.get(key)
    if found is None:
        layout = kattn_scratch_layout(rows, Q_LEN)
        buf = torch.zeros(GUARD_BYTES + scratch_bytes(layout), dtype=torch.uint8, device=device)
        words = buf[:GUARD_BYTES].view(torch.int32)
        ticks = deadline_ticks(DEADLINE_S)
        low = ticks & 0xFFFFFFFF
        words[GUARD_DEADLINE:GUARD_DEADLINE + 1].fill_(low - (1 << 32) if low >= 1 << 31 else low)
        words[GUARD_DEADLINE + 1:GUARD_DEADLINE + 2].fill_(ticks >> 32)
        start = GUARD_BYTES + layout["attn_pflag"][0]
        end = GUARD_BYTES + layout["attn_l2flag"][0] + layout["attn_l2flag"][1]
        found = _BUFFERS[key] = (buf, buf[start:end])
    return found


def run(*, q, k_cache, v_cache, page_table, seq_lens, out) -> None:
    rows = q.shape[0]
    assert rows % Q_LEN == 0 and seq_lens.shape[0] * Q_LEN == rows, (rows, tuple(seq_lens.shape))
    launch = _launcher(rows)
    buf, flags = _buffers(rows, q.device)
    flags.zero_()  # no flag of an earlier launch carries this launch's tag
    base = buf.data_ptr()
    launch(
        q.data_ptr(), out.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), page_table.data_ptr(),
        page_table.shape[1], page_table.numel() * 4, seq_lens.data_ptr(),
        int(seq_lens.dtype == torch.int64), Q_LEN, 0, 0, SCALE_BITS, base + GUARD_BYTES, base, LAYER,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
