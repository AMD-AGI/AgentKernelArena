# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""One Kimi-K3 KDA layer front (sgl-amd-mono's K1 launch, ``kda_pre``) in the arena run() interface.

The device code is sgl-amd-mono's FlyDSL kernel ``kda.py`` (vendored at 52486703): one persistent grid
of 256 CTAs x 512 threads that runs the attention-side AttnRes seam and ``input_layernorm``, the bf16
in_proj GEMV (6288 rows a rank at TP8), the short conv + SiLU, the f_b gate, the q / k l2 norms, the
KDA delta rule on the request's fp32 [12, 128, 128] state and the gated output RMSNorm, handing values
between its stages through tagged-pair mailboxes in a global scratch.

The build is the one SGLang serves (``kimi_k3/gfx950/k1.py`` ``build``): one request of ``rows`` verify
tokens (tokens = qlen = rows), ``nblocks`` AttnRes bank rows, no delta, no bank write (a KDA layer is
never a block-write layer in Kimi-K3), the snapshot spec state (``ring`` off) with the initial state
read from the committed pool (``init_committed``), conv state length rows + 3, eps 1e-5 everywhere
and the lower-bound gate at -5. Host work mirrors ``k1.launch`` for one layer:

* the conv buffer is handed over as K1 receives its own buffer, a view with a zero slot stride, so the
  slot number the kernel reads (the first token's state row) folds onto the request's single window;
  the task's ``conv_state`` is token-major ([rows + 3, 4608], SGLang's pool order), which the kernel
  reads through its strides;
* the per-token states go to ``ssm_states`` at ``state_rows`` (SGLang's verify-scratch rows);
* ``num_acc`` = 1 (K1's form pins it: prepare() stages the committed window at offset 0), the
  committed state at slot 0 of a one-slot pool view of ``ssm_state``;
* the mailbox scratch and the epoch word are allocated zeroed for every call (sgl-amd-mono's runner
  keeps one scratch and bumps the epoch once a step on the device): every pair of this call carries
  the tag epoch 0 / slot 1, which no zeroed word matches.
"""

import functools

import torch

from . import kda

LAYER_SLOT = 1  # the launch's mailbox slot (tag = epoch * 256 + slot); nonzero, so a zeroed pair is stale
META_WORDS = 16  # int32 words ahead of the scratch: [0] epoch, [1] init_slot, [2] num_acc, rest padding
EPOCH, INIT_SLOT, NUM_ACC = 0, 1, 2


@functools.lru_cache(maxsize=None)
def _key(rows: int, nblocks: int) -> kda.KdaPreBuild:
    return kda.KdaPreBuild(
        tokens=rows,
        qlen=rows,
        nblocks=nblocks,
        delta=False,
        state_len=rows + kda.CONV_W - 1,
        init_committed=True,
    )


def run(*, prefix, blocks, attn_res_norm, attn_res_query, input_norm, w_in, w_fb, conv_weight, a_log,
        dt_bias, o_norm, ssm_state, state_rows, conv_state, core_out, ssm_states) -> None:
    rows, nblocks = prefix.shape[0], blocks.shape[1]
    assert prefix.shape == (rows, kda.HIDDEN) and prefix.is_contiguous() and blocks.is_contiguous()
    assert conv_state.shape == (rows + kda.CONV_W - 1, 3 * kda.PROJ) and conv_state.is_contiguous()
    key = _key(rows, nblocks)
    meta_bytes = META_WORDS * 4
    # epoch / pool-slot / acceptance words and the mailbox scratch, zeroed: no pair of an earlier
    # call carries this call's tag
    buf = torch.zeros(meta_bytes + kda.scratch_bytes(key), dtype=torch.uint8, device=prefix.device)
    meta = buf[:meta_bytes].view(torch.int32)
    meta[NUM_ACC:NUM_ACC + 1].fill_(1)
    kda.kda_pre(
        key,
        prefix=prefix,
        delta=None,
        blocks=blocks,
        ares_nw=attn_res_norm,
        ares_qk=attn_res_query,
        in_nw=input_norm,
        w_in=w_in,
        w_fb=w_fb,
        conv_w=conv_weight,
        # [slots, dim, state_len] with a zero slot stride, as k1.py hands over its own buffer
        conv_state=conv_state.t().unsqueeze(0).expand(2, -1, -1),
        a_log=a_log,
        dt_bias=dt_bias,
        on_w=o_norm,
        rstate=ssm_states,
        st_idx=state_rows.view(1, rows),
        num_acc=meta[NUM_ACC:NUM_ACC + 1],
        core_out=core_out,
        scratch=buf[meta_bytes:],
        layer=LAYER_SLOT,
        epoch=meta[EPOCH:EPOCH + 1],
        rstate0=ssm_state.view(1, kda.NH, kda.HD, kda.HD),
        init_slot=meta[INIT_SLOT:INIT_SLOT + 1],
    )
