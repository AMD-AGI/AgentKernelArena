# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 layer-post launch (sgl-amd-mono ``kimi_k3/gfx950/layer.py``, K2) in the arena ``run()``
interface, one rank of the TP8 group.

One persistent 256-CTA launch takes the layer from the attention's core output to the layer output:
o_proj (this rank's 1536 input columns) with its all-reduce over the peers' symmetric buffers, the
residual add and the MLP AttnRes seam (the live bank rows and the updated prefix, softmax-mixed, then
post_attention_layernorm); the sigmoid router's top-16 of 896 (with the score-correction bias), the
latent down projection (7168 -> 3584, an eighth a rank, all-gathered and MXFP4-quantized), the shared
experts' gate / up, SiTU and down (TP slices), the routed experts on MXFP4 weights (aiter's A16W4 layout,
gate / up interleaved; SiTUv2 and the MXFP4 requantization of its output), the latent all-reduce and
RMSNorm, the latent up projection of this rank's 896 hidden columns, and the final all-reduce, written
with the updated residual prefix added (``fold``).

The build is the one ``kimi_k3/spec.py::_weights`` makes for a covered middle layer that is not a
block-write layer (``reset=False``), with the serving defaults: ``fold``, ``reduce``, gate / up
interleaved experts (``guint``, the layout ``spec.py`` requires) and the OPT bits ``layer.K2_OPT`` minus
the three sentinel all-reduces (``OPT_SENTS``). The sentinel regions must be armed with 0xFF bytes once
(``init_peer``) and are re-armed by the readers, which the framework's zeroing of ``symmetric`` before
every call would undo; ``check_k2.py`` checks every OPT bit bit for bit against the launch without it.

Placement. The peer buffer is the framework's ``symmetric`` buffer (the runner's ``PeerBuffer``, the same
uncached allocation): the two epoch-parity halves of the widest width, then the guard's flags at
``layer.flags_offset()``. The runner's launch scratch (``K3MonoRunner.scratch``, a one-rank uncached
``PeerBuffer`` there) sits past them in the same buffer, ``layer.scratch_alloc_bytes()`` bytes (the tagged
x rows of OPT_TAG_X included). The framework zeroes the whole buffer before every call, which stands in
for the runner's per-step epoch bump: a call's mailbox tags never meet a previous call's words. ``guard``
is the runner's epoch buffer (epoch 1, the wait deadline, clean error words).
"""

import functools

import torch

from .guard import FLAG_BYTES
from .layer import K2_OPT, OPT_SENTS, K2Build, build, flags_offset, scratch_alloc_bytes
from .model import MAX_ROWS

NBLOCKS = 4  # bank rows live at this layer's MLP seam (prev_valid_blocks of a mid-model layer)
LAYER = 128 + 32  # spec.py's launch slot of layer 32: only the mailbox tags read it
OPT = K2_OPT & ~OPT_SENTS
TP = 8
_ALIGN = 4096

SCRATCH_OFFSET = -(-(flags_offset() + FLAG_BYTES) // _ALIGN) * _ALIGN
SYMMETRIC_BYTES = SCRATCH_OFFSET + -(-scratch_alloc_bytes() // 256) * 256


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    """``spec._weights``' K2Build for a plain covered layer at ``rows`` rows."""
    return build(K2Build(tokens=rows, nblocks=NBLOCKS, reset=False, fold=True, guint=True, opt=OPT))


def run(
    *,
    core, w_o, prefix_in, blocks, ares_nw, ares_qk, in_nw, w_gate, bias, w_ld, w_sgu, w_sd,
    w13, w13s, w2, w2s, ln_w, w_up, guard,
    out, prefix,
    rank, world_size, peers, symmetric,
) -> None:  # fmt: skip
    """One rank's layer-post launch; every rank of the group calls it with its own shard."""
    rows = core.shape[0]
    assert 1 <= rows <= MAX_ROWS, rows
    assert int(world_size) == TP and symmetric.numel() >= SYMMETRIC_BYTES, (world_size, symmetric.numel())
    launch = _launcher(rows)
    launch(
        core.data_ptr(), w_o.data_ptr(), prefix.data_ptr(), prefix_in.data_ptr(),
        blocks.data_ptr(), blocks.stride(0), blocks.stride(1),
        ares_nw.data_ptr(), ares_qk.data_ptr(), in_nw.data_ptr(),
        w_gate.data_ptr(), bias.data_ptr(), w_ld.data_ptr(), w_sgu.data_ptr(), w_sd.data_ptr(),
        w13.data_ptr(), w13s.data_ptr(), w2.data_ptr(), w2s.data_ptr(), ln_w.data_ptr(), w_up.data_ptr(),
        out.data_ptr(), symmetric.data_ptr() + SCRATCH_OFFSET, peers.data_ptr(), int(rank),
        guard.data_ptr(), LAYER, 0,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
