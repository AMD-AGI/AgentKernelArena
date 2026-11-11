# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""One Kimi-K3.1 MLA (DSA) layer's projection front (sgl-amd-mono's ``mla_pre`` launch) in the arena
run() interface.

The device code is sgl-amd-mono's FlyDSL kernel ``mla_pre.py`` (vendored at 52486703): one persistent
grid of 256 CTAs x 512 threads that runs the attention-side AttnRes seam and ``input_layernorm``, the
stage-1 bf16 GEMVs of x (fused_qkv_a 2112 rows, g_proj 1536, the indexer's wk 128 and weights_proj
32), the q / kv latent RMSNorms, the stage-2 GEMVs of the normed q latent (q_b 2304 rows, the
indexer's wq_b 4096) and the stage-3 absorb of q_nope into the kv_lora space (W_UK), handing values
between its stages through tagged-pair mailboxes in a global scratch.

The build is the one SGLang serves (``kimi_k3/spec.py`` ``_mla_build``) on a layer that is not a
block-write layer: ``rows`` verify tokens, ``nblocks`` AttnRes bank rows, the absorb on (``qn``: a
bf16 checkpoint's w_kc with w_scale 1, which SGLang's ROCm MLA forward reads as q_nope_out), eps
1e-5 for the seam and input_layernorm and 1e-6 for the latent norms. Host work mirrors
``spec._mla_core`` for one layer; the mailbox scratch and the epoch word are allocated zeroed for
every call (sgl-amd-mono's runner keeps one scratch and bumps the epoch once a step on the device):
every pair of this call carries the tag epoch 0 / slot 1, which no zeroed word matches.
"""

import functools

import torch

from . import mla_pre as mp

LAYER_SLOT = 1  # the launch's mailbox slot (tag = epoch * 256 + slot); nonzero, so a zeroed pair is stale
META_BYTES = 64  # the epoch word (int32 0) and padding ahead of the scratch


@functools.lru_cache(maxsize=None)
def _key(rows: int, nblocks: int) -> mp.MlaPreBuild:
    return mp.MlaPreBuild(
        tokens=rows,
        nblocks=nblocks,
        eps=1e-5,
        out_eps=1e-5,
        q_eps=1e-6,
        kv_eps=1e-6,
        write_idx=-1,
        qn=True,
    )


def run(*, prefix, blocks, attn_res_norm, attn_res_query, input_norm, w_qkv_a, w_gate, w_index_k,
        w_index_weights, w_q_b, w_index_q, q_a_norm, kv_a_norm, w_kc, q, latent, k_nope, gate, index_q,
        index_k, index_weights, q_nope_absorbed) -> None:
    rows, nblocks = prefix.shape[0], blocks.shape[1]
    assert prefix.shape == (rows, mp.HIDDEN) and prefix.is_contiguous() and blocks.is_contiguous()
    key = _key(rows, nblocks)
    # the epoch word and the mailbox scratch, zeroed: no pair of an earlier call carries this call's tag
    buf = torch.zeros(META_BYTES + mp.scratch_bytes(key), dtype=torch.uint8, device=prefix.device)
    w = {
        "ares_nw": attn_res_norm, "ares_qk": attn_res_query, "in_nw": input_norm,
        "w_qkv": w_qkv_a, "w_g": w_gate, "w_k": w_index_k, "w_p": w_index_weights,
        "w_qb": w_q_b, "w_iq": w_index_q, "q_nw": q_a_norm, "kv_nw": kv_a_norm,
        # [heads x kv_lora, nope], as spec._mla_build hands the absorb its weight
        "w_kc": w_kc.view(mp.NHQ * mp.KV, mp.QN),
    }
    out = {
        "q": q, "latent": latent, "k_nope": k_nope, "gate": gate, "idx_q": index_q,
        "idx_k": index_k, "idx_w": index_weights, "qn": q_nope_absorbed,
    }
    mp.mla_pre_launch(
        key,
        prefix=prefix,
        blocks=blocks,
        w=w,
        out=out,
        scratch=buf[META_BYTES:],
        layer=LAYER_SLOT,
        epoch=buf[:META_BYTES].view(torch.int32)[:1],
    )
