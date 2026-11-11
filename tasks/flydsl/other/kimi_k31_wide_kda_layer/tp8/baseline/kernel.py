# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 wide-row KDA layer launch (sgl-amd-mono ``kimi_k31/gfx950/layer.py::kda_layer``) in the
arena ``run()`` interface, one rank of the TP8 group.

One persistent 256-CTA launch per rank runs, for a DSpark verify step padded to a row bucket of ``m`` rows
(R requests x 8 tokens): the previous layer's MoE tail (latent all-reduce, latent RMSNorm, latent up on this
rank's columns + the shared all-reduce + the prefix), the attention AttnRes seam, in_proj, the KDA verify on
the ReplaySSM ring (one task a (request, head)), o_proj with its reduce-scatter into the column-sharded
residual, the MLP AttnRes seam and the MoE front (router logits, latent down, shared gate / up), the in-launch
top-16 route, the shared SiTU and the shared down. The ranks exchange slices through each other's symmetric
buffers inside the kernel.

The build is the one ``wide.Wide._build_launch`` makes for a covered KDA layer in the middle of a run (the
previous layer's MoE tail in this launch, ``t1``; not a block-write layer; no DSpark capture), with the
bucket's ``wide.cfg_rows`` GEMM configurations, SGLang's up-projection rows read as is (``up_norm``, the
latent norm weight after the score-correction bias in ``bias``), a rank's contiguous front rows
(``front_rows`` None) and no in-launch expert sort (``SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT=0``; see
README.md). Its arguments are ``Wide._launch_args``' with these placements:

- the peer buffer (``Wide.peers``, two parity halves sized as ``Wide._ensure`` sizes them, then the guard's
  flags) is the start of the framework's ``symmetric`` buffer;
- the launch's local hand-off scratch (``Wide.bufs["scratch"]``, an uncached one-rank buffer there too), the
  o_proj merge buffer ``acc_o`` and the tile counters ``cnt`` follow it in ``symmetric``. The framework zeroes
  the whole buffer before every call, which stands in for the runner's "zeroed once" plus its per-step epoch
  bump;
- ``ip_buf`` and ``acc_u`` (zeroed once upstream, left zeroed by every launch) are task inputs that the
  launch must leave zero;
- the KDA stage's pointers and strides are kernel arguments instead of the host-built ``kda_tab`` table
  (``layer._KdaKernArgs``);
- ``cap_out`` and ``sort_buf`` are not read by this build (null).
"""

import functools

import torch

from . import layer as lay
from .chain import Gemm
from .guard import FLAG_BYTES
from .wide_cfg import H, NF, SL, cfg_rows

L = 8  # tokens a request: DSpark's 7 draft tokens + 1
NVB = 4  # the bank snapshots both seams read (layer.prev_valid_blocks; not a block-write layer)
NBK = 8  # bank rows allocated a token (Wide: bufs["bank"] = [b, 8, SL])
EPS = 1e-5  # input_layernorm / mlp_res_norm / post_attention_layernorm variance_epsilon
EPS_N = 1e-5  # o_norm eps (and the latent RMSNorm's)
LAUNCHES = 52  # covered layers a step (Wide: len(self.layers)); the tag parity reads it
SLOT = 13  # this layer's launch slot in the step (Wide.slots); the mailbox tag reads it
TP = 8
_ALIGN = 4096


def _cfg(m):
    return lay.LayerCfg(*(Gemm(*(int(v) for v in c.split(":"))) for c in cfg_rows(m, L)))


@functools.lru_cache(maxsize=None)
def _layout(m):
    """(half_bytes, {region: byte offset in symmetric}, bytes used) of bucket m."""
    cfg = _cfg(m)
    # Wide._ensure: one peer layout for every variant of the bucket, sized for a capture launch too
    half = max(lay.plan(m, cfg, L, NVB, NVB, NF, cap)[0].peer_offsets[1] for cap in (False, True))
    pl = lay.plan(m, cfg, L, NVB, NVB, NF, False)[0]
    sizes = (
        ("peer", 2 * half + FLAG_BYTES),
        ("scratch", pl.scratch_bytes()),
        ("acc_o", m * H * 2),
        ("cnt", lay.counters(m, cfg, NF) * 4),
    )
    offs, end = {}, 0
    for name, nbytes in sizes:
        offs[name] = end
        end += -(-nbytes // _ALIGN) * _ALIGN
    return half, offs, end


@functools.lru_cache(maxsize=None)
def _launcher(m):
    """``Wide._build_launch`` of a middle covered layer of bucket m (L = 8)."""
    half, _, _ = _layout(m)
    return lay.kda_layer(
        m,
        _cfg(m),
        L,
        NF,
        nvb_a=NVB,
        nvb_m=NVB,
        write_a=False,
        nbk=NBK,
        launches=LAUNCHES,
        eps=EPS,
        eps_n=EPS_N,
        capture=False,
        t1=True,
        restart=False,
        half_bytes=half,
        front_rows=None,
        up_norm=True,
        sort=None,
    )[0]


def run(
    *,
    lat_p, sh_p, w_up, prefix, bank, cw_a, w_in, w_ip, w_fb, conv_w, a_log, dt_bias, on_w, conv_state, rstate,
    st_idx, num_acc, win_row, wo, cw_m, w_ln, wf, bias, w_sd, epoch,
    prefix_out, ip_buf, acc_u, ring_k, ring_v, ring_g, ring_b, conv_win,
    mid, core, lat_out, topk_ids, topk_w, logits_out, sh_out,
    rank, world_size, peers, symmetric,
) -> None:  # fmt: skip
    """One rank's layer launch on its shard; every rank of the group calls it."""
    m = lat_p.shape[0]
    rank = int(rank)
    assert int(world_size) == TP and m % L == 0, (world_size, m)
    _, offs, need = _layout(m)
    assert symmetric.numel() >= need, (symmetric.numel(), need)
    sym = symmetric.data_ptr()
    col0 = 2 * rank * SL  # bytes to this rank's first column of a full-width bf16 row
    conv_st = conv_state.mT  # [slot, channel, token]: the strides the KDA stage reads
    launch = _launcher(m)
    launch(
        # T1: the previous layer's MoE tail
        lat_p.data_ptr(), sh_p.data_ptr(), w_up.data_ptr(), prefix.data_ptr() + col0, mid.data_ptr(),
        prefix_out.data_ptr() + col0, bank.data_ptr(), cw_a.data_ptr(), w_in.data_ptr(), w_ip.data_ptr(),
        ip_buf.data_ptr(), acc_u.data_ptr(),
        # T4 .. T6
        core.data_ptr(), wo.data_ptr(), cw_m.data_ptr(), w_ln.data_ptr(), wf.data_ptr(), sym + offs["acc_o"],
        bias.data_ptr(), lat_out.data_ptr(), topk_ids.data_ptr(), topk_w.data_ptr(), w_sd.data_ptr(),
        sh_out.data_ptr(), 0, H, logits_out.data_ptr(),
        # the launch
        sym + offs["cnt"], sym + offs["scratch"], peers.data_ptr(), rank, epoch.data_ptr(), SLOT,
        m * H * 2 - col0, 0,
        # the KDA stage (pre_attn.KDA_PTRS, then pre_attn.KDA_INTS)
        w_fb.data_ptr(), conv_w.data_ptr(), conv_st.data_ptr(), a_log.data_ptr(), dt_bias.data_ptr(),
        on_w.data_ptr(), rstate.data_ptr(), st_idx.data_ptr(), num_acc.data_ptr(), ring_k.data_ptr(),
        ring_v.data_ptr(), ring_g.data_ptr(), ring_b.data_ptr(), core.data_ptr(), conv_win.data_ptr(),
        win_row.data_ptr(),
        conv_st.stride(0), conv_st.stride(1), conv_st.stride(2), rstate.stride(0), st_idx.stride(0),
        ring_k.stride(0), ring_v.stride(0), ring_g.stride(0), ring_b.stride(0), ring_k.shape[2],
        conv_win.stride(0), conv_win.stride(1), conv_win.stride(2),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
