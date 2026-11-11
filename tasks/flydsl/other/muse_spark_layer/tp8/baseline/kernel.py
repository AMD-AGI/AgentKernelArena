# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark whole-layer launch (sgl-amd-mono ``muse_spark/gfx950/layer.py::build_layer``) in the
arena ``run()`` interface, one rank of the TP8 / EP8 group.

One persistent 256-CTA launch runs this layer's verify attention over the fp8 page-16 KV cache, K2
(o-norm-gate, o_proj with the in-kernel all-reduce over the peers' symmetric buffers, the attention
boundary, the router's top-8, the row-sharded pooler, the pre-expert norm), K3 (the MXFP4 experts as TP
slices of every expert, the finalize, the sharded unpooler, the FFN boundary) and the next layer's K1 (the
fused q | k | v | gate projection, the q / k norms, the query scale, RoPE and the fp8 cache write through the
sliding pool's slot map).

The build is the one ``MuseMonoRunner.layer_kernel`` makes with the runner's default environment
(``SGLANG_AMD_MONO_MUSE_TPX=1``, ``..._XQ_PRE=1``, ``..._UG_DYN=1``, ``..._PSHARD=1``, ``..._DEFER_K1=0``,
``..._MUSE_FP8_DENSE=0``) for a middle MXFP4 layer whose next layer also runs the kernels. The launch's
arguments are the runner's (``MuseMonoRunner.launch``, merged path), with two differences of placement:

- the peer buffer is the framework's ``symmetric`` buffer (the runner's ``PeerBuffer``, the same uncached
  allocation): its first ``sym_bytes(npes) + FLAG_BYTES`` bytes are the peer regions and the guard's flags;
- the runner's four local scratch buffers (``scratch_attn``, ``scratch_k2``, ``scratch_k3``, ``scratch_k1``,
  ordinary device memory there) are carved out of ``symmetric`` past the peer regions. The framework zeroes
  the whole buffer before every call, which stands in for the runner's per-step epoch advance: a call's
  mailbox tags never meet a previous call's words.

``guard`` is the runner's epoch / guard-word buffer (epoch 1, the wait deadline, clean error words).
"""

import functools

import torch

from .config import LOCAL_EXPERTS, MAX_TOKENS, MOE_HIDDEN
from .ffn import k3_scratch_layout
from .layer import LayerBuild, build_layer
from .guard import FLAG_BYTES
from .layout import k1_scratch_layout, kattn_scratch_layout, scratch_bytes
from .post_attn import k2_scratch_layout, sym_bytes

Q_LEN = 8  # rows a request: the DFlash2 verify's 8 draft tokens
LAYER = 9  # the layer index: only the mailbox tags (epoch * LAYER_SLOTS + layer + 1) read it
SCALE_BITS = 0x3E000000  # float32 bits of the softmax scale 64 ** -0.5 = 0.125 (RadixAttention.scaling)
_ALIGN = 4096


def _local_regions(start: int):
    """The runner's four scratch buffers (sized as the runner sizes them, for MAX_TOKENS rows) as byte
    offsets into the symmetric buffer from ``start``, and the end of the last."""
    sizes = (
        ("attn", scratch_bytes(kattn_scratch_layout(MAX_TOKENS, Q_LEN))),
        ("k2", scratch_bytes(k2_scratch_layout())),
        ("k3", scratch_bytes(k3_scratch_layout())),
        ("k1", scratch_bytes(k1_scratch_layout())),
    )
    offsets, end = {}, start
    for name, nbytes in sizes:
        offsets[name] = end
        end += -(-nbytes // _ALIGN) * _ALIGN
    return offsets, end


PEER_BYTES = sym_bytes(8) + FLAG_BYTES  # the peer regions and the guard's flags (TP 8)
REGIONS, SYMMETRIC_BYTES = _local_regions(-(-PEER_BYTES // _ALIGN) * _ALIGN)


@functools.lru_cache(maxsize=None)
def _launcher(rows: int, npes: int):
    """``MuseMonoRunner.layer_kernel(rows, 8, next_k1=True)`` with the runner's defaults."""
    return build_layer(
        LayerBuild(
            tokens=rows,
            q_len=Q_LEN,
            npes=npes,
            next_k1=True,
            last=False,
            fp8_dense=False,
            tpx=True,
            defer_k1=False,
            xq_pre=True,
            ug_dyn=True,
            # MuseMonoRunner._pshard: the row-sharded pooler at TP 8, S <= 16 only
            pshard=npes == 8 and rows <= 16,
        )
    )


def run(
    *,
    q, g, k_cache, v_cache, page_table, seq_lens, res,
    w_o, alpha, beta, pre_w, router_hi, router_lo, expert_bias, w_pool, xn_w,
    w13, s13, w2, s2, pen_w, pfw, alpha2, beta2, npw, w_unpool,
    w_qkvg1, cos_sin, positions, slots, swa_map, k_cache1, v_cache1,
    guard,
    rank, world_size, peers, symmetric,
    o, ids, scores, xin, attn_out, k_out, v_out,
) -> None:  # fmt: skip
    """One rank's merged layer launch; every rank of the group calls it with its own shard."""
    rows = q.shape[0]
    npes = int(world_size)
    rank = int(rank)
    assert rows % Q_LEN == 0 and 1 <= rows <= MAX_TOKENS, rows
    assert npes == 8 and symmetric.numel() >= SYMMETRIC_BYTES, (npes, symmetric.numel(), SYMMETRIC_BYTES)
    launch = _launcher(rows, npes)
    pshard = npes == 8 and rows <= 16
    pool = w_pool[rank * (MOE_HIDDEN // npes) : (rank + 1) * (MOE_HIDDEN // npes)] if pshard else w_pool
    sym = symmetric.data_ptr()
    launch(
        # the attention (seq_lens: the keys before this step's rows; extra: the step's Q_LEN rows)
        q.data_ptr(), o.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), page_table.data_ptr(),
        page_table.shape[1], page_table.numel() * 4, seq_lens.data_ptr(), 0, Q_LEN, 0, 0, SCALE_BITS,
        sym + REGIONS["attn"],
        # K2
        g.data_ptr(), res.data_ptr(), w_o.data_ptr(), alpha.data_ptr(), beta.data_ptr(), pre_w.data_ptr(),
        router_hi.data_ptr(), router_lo.data_ptr(), expert_bias.data_ptr(), pool.data_ptr(), xn_w.data_ptr(),
        ids.data_ptr(), scores.data_ptr(), xin.data_ptr(), sym + REGIONS["k2"],
        # K3 (TP experts: local_start is not read; the runner passes the EP start)
        rank * LOCAL_EXPERTS, w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(), pen_w.data_ptr(),
        pfw.data_ptr(), alpha2.data_ptr(), beta2.data_ptr(), npw.data_ptr(), w_unpool.data_ptr(),
        attn_out.data_ptr(), sym + REGIONS["k3"],
        # the next layer's K1: its q / gate land in q / g, a sliding layer with RoPE, int64 slots and slot map
        w_qkvg1.data_ptr(), cos_sin.data_ptr(), positions.data_ptr(), slots.data_ptr(), swa_map.data_ptr(),
        k_cache1.data_ptr(), v_cache1.data_ptr(), k_out.data_ptr(), v_out.data_ptr(), sym + REGIONS["k1"],
        1, 1, 1, 1,
        # the launch
        sym, peers.data_ptr(), rank, guard.data_ptr(), LAYER, 0,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
