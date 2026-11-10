# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""One Qwen3.8-Flash-Next decoder launch (three Gated DeltaNet layers and one QSA layer, each with its
hyper-connection mixers and its 512-expert MXFP4 MoE) in the arena run() interface.

The device code is sgl-amd-mono's FlyDSL launch kernel (``layers.py``, vendored at 52486703), built
for the production middle plan ``gdn,gdn,gdn,qsa`` of a NEXTN target verify: R requests x 4 tokens,
one persistent grid of 256 CTAs x 512 threads. Host work here mirrors sgl-amd-mono's
``VerifyRunner`` for one launch (``gfx950/runner.py``): it builds the int64 pointer table (one row a
layer plus the global row, ``layout.P_*`` / ``layout.G_*``) from the tensors' addresses, allocates
the tagged-pair mailbox scratch and the epoch / guard block, and launches on the current stream.

Differences from the runner, all host-side: the scratch and epoch are allocated zeroed per call (the
runner keeps one scratch and moves the epoch on after each step; a fresh zeroed scratch makes every
mailbox tag of this call current exactly once), the cache strides of the build key are those of the
task's fixed layouts, and the pointer table of a set of tensors is uploaded once (outside graph
capture, at the first eager call) and reused, as the runner caches its tables by address.
"""

import functools

import torch

from . import layout as LY
from .config import GDN_CONV_DIM, GDN_DK, GDN_HV, HC_DIM, ConvStrides, MonoBuild
from .layers import build_qwen38_flash_next_kernel

PLAN = "gdn,gdn,gdn,qsa"
L = 4  # verify tokens a request: NEXTN, 3 draft steps
GDN_LAYERS = (0, 1, 2)
QSA_LAYER = 3
N_LAYERS = 4
GI = N_LAYERS  # the global row of the pointer table
RING = 8  # index-key ring slots a request (two groups)
# guard block (guard.py): word 0 the epoch, words 1-2 the wait deadline in 100 MHz clock ticks
GUARD_WORDS = 8
EPOCH_PAD_WORDS = 64  # the mailboxes start 256 bytes after the guard block
DEADLINE_TICKS = 30 * 100_000_000  # 30 s: a wait that never completes stops the launch
# the deadline as the two int32 words of the guard block (low word as a signed int32)
DEADLINE_LO = (DEADLINE_TICKS & 0xFFFFFFFF) - ((DEADLINE_TICKS & 0x80000000) << 1)
DEADLINE_HI = DEADLINE_TICKS >> 32
_TABLES = {}  # pointer tuple -> device table
_MAX_TABLES = 64


@functools.lru_cache(maxsize=None)
def _launcher(nreq: int, rtt_stride: int):
    key = MonoBuild(
        plan=PLAN,
        nreq=nreq,
        ntok=L,
        rtt_stride=rtt_stride,
        pos_stride=0,  # 1-D positions: the three MRoPE axes read the same element
        map_i64=False,
        first_src="global",
        # intermediate conv window [R, 10240, L + 2]: draft step j's window is columns j .. j + 2
        iconv=ConvStrides(step=1, channel=L + 2, window=1, row=GDN_CONV_DIM * (L + 2)),
        issm_row=L * GDN_HV * GDN_DK * GDN_DK,
        ssm_bf16=False,
        seq_p0=True,
        ring=RING,
    )
    return build_qwen38_flash_next_kernel(key)


@functools.lru_cache(maxsize=None)
def _scratch_words(nreq: int) -> int:
    return EPOCH_PAD_WORDS + LY.scratch_layout(N_LAYERS, nreq * L, nreq)[1] // 4


def _layer_ptr(t: torch.Tensor, i: int) -> int:
    """Address of t[i] (a layer of a stacked weight)."""
    return t.data_ptr() + i * t.stride(0) * t.element_size()


def _rows(a: dict) -> list:
    rows = []
    for li in range(N_LAYERS):
        row = [0] * LY.N_PTR
        for slots, pre in (((LY.P_HCA_NORM, LY.P_HCA_DOWN, LY.P_HCA_UP), "hc_attn"),
                           ((LY.P_HCM_NORM, LY.P_HCM_DOWN, LY.P_HCM_UP), "hc_mlp")):
            for slot, part in zip(slots, ("norm", "down", "up")):
                row[slot] = _layer_ptr(a[f"{pre}_{part}"], li)
        row[LY.P_ROUTER] = _layer_ptr(a["router"], li)
        row[LY.P_GU] = _layer_ptr(a["moe_w13"], li)
        row[LY.P_GU_S] = _layer_ptr(a["moe_w13_scale"], li)
        row[LY.P_DN] = _layer_ptr(a["moe_w2"], li)
        row[LY.P_DN_S] = _layer_ptr(a["moe_w2_scale"], li)
        if li in GDN_LAYERS:
            g = GDN_LAYERS.index(li)
            row[LY.P_IP] = _layer_ptr(a["gdn_in_proj"], g)
            row[LY.P_IP_S] = _layer_ptr(a["gdn_in_proj_scale"], g)
            row[LY.P_BA] = _layer_ptr(a["gdn_ba_proj"], g)
            row[LY.P_BA_S] = _layer_ptr(a["gdn_ba_proj_scale"], g)
            row[LY.P_CONV_W] = _layer_ptr(a["gdn_conv_weight"], g)
            row[LY.P_NEG_A] = _layer_ptr(a["gdn_neg_a"], g)
            row[LY.P_DT_BIAS] = _layer_ptr(a["gdn_dt_bias"], g)
            row[LY.P_NORM_W] = _layer_ptr(a["gdn_norm_weight"], g)
            row[LY.P_OUT] = _layer_ptr(a["gdn_out_proj"], g)
            row[LY.P_OUT_S] = _layer_ptr(a["gdn_out_proj_scale"], g)
            row[LY.P_CONV_STATE] = _layer_ptr(a["gdn_conv_state"], g)
            row[LY.P_SSM_STATE] = _layer_ptr(a["gdn_ssm_state"], g)
            # request r owns row r of the verify scratch
            row[LY.P_ISSM] = _layer_ptr(a["gdn_intermediate_ssm"], g)
            row[LY.P_ICONV] = _layer_ptr(a["gdn_intermediate_conv"], g)
        else:
            row[LY.P_IP] = a["qsa_qkv_proj"].data_ptr()
            row[LY.P_IP_S] = a["qsa_qkv_proj_scale"].data_ptr()
            row[LY.P_IP2] = a["qsa_index_proj"].data_ptr()
            row[LY.P_IP2_S] = a["qsa_index_proj_scale"].data_ptr()
            row[LY.P_OUT] = a["qsa_o_proj"].data_ptr()
            row[LY.P_OUT_S] = a["qsa_o_proj_scale"].data_ptr()
            row[LY.P_QNORM] = a["qsa_q_norm"].data_ptr()
            row[LY.P_KNORM] = a["qsa_k_norm"].data_ptr()
            row[LY.P_IQNORM] = a["qsa_index_q_norm"].data_ptr()
            row[LY.P_IKNORM] = a["qsa_index_k_norm"].data_ptr()
            row[LY.P_KBUF] = a["k_cache"].data_ptr()
            row[LY.P_VBUF] = a["v_cache"].data_ptr()
            row[LY.P_RING] = a["index_ring"].data_ptr()
            row[LY.P_COMP] = a["index_keys"].data_ptr()
            row[LY.P_KNEW] = a["k_new"].data_ptr()
            row[LY.P_VNEW] = a["v_new"].data_ptr()
            row[LY.P_CNEW] = a["index_key_new"].data_ptr()
        rows.append(row)
    g = [0] * LY.N_PTR
    g[LY.G_POS] = a["positions"].data_ptr()
    g[LY.G_REQ] = a["req_pool_indices"].data_ptr()
    g[LY.G_SEQ] = a["seq_lens"].data_ptr()
    g[LY.G_RTT] = a["req_to_token"].data_ptr()
    g[LY.G_COS] = a["rope_cos_sin"].data_ptr()
    g[LY.G_ROPEPOS] = a["index_ring_positions"].data_ptr()
    g[LY.G_MAMBA_MAP] = a["mamba_slot_map"].data_ptr()
    g[LY.G_HIN_SSQ] = a["h_in_ssq"].data_ptr()
    g[LY.G_HOUT_SSQ] = a["h_out_ssq"].data_ptr()
    rows.append(g)
    return [v for row in rows for v in row]


def _table(a: dict, device) -> torch.Tensor:
    """The launch's pointer table on the device: built and uploaded at the first (eager) call with
    these tensors, then reused (a captured graph replays the same addresses)."""
    flat = _rows(a)
    key = tuple(flat)
    table = _TABLES.get(key)
    if table is None:
        if len(_TABLES) >= _MAX_TABLES:
            _TABLES.clear()
        host = torch.empty(len(flat), dtype=torch.int64)
        for i, v in enumerate(flat):
            host[i] = v
        table = _TABLES[key] = host.to(device)
    return table


def run(*, h_in, h_in_ssq, positions, seq_lens, req_pool_indices, req_to_token, mamba_slot_map,
        rope_cos_sin, hc_attn_norm, hc_attn_down, hc_attn_up, hc_mlp_norm, hc_mlp_down, hc_mlp_up,
        router, moe_w13, moe_w13_scale, moe_w2, moe_w2_scale, gdn_in_proj, gdn_in_proj_scale,
        gdn_ba_proj, gdn_ba_proj_scale, gdn_conv_weight, gdn_neg_a, gdn_dt_bias, gdn_norm_weight,
        gdn_out_proj, gdn_out_proj_scale, gdn_conv_state, gdn_ssm_state, qsa_qkv_proj,
        qsa_qkv_proj_scale, qsa_index_proj, qsa_index_proj_scale, qsa_o_proj, qsa_o_proj_scale,
        qsa_q_norm, qsa_k_norm, qsa_index_q_norm, qsa_index_k_norm, k_cache, v_cache, index_keys,
        index_ring, index_ring_positions, h_out, h_out_ssq, gdn_intermediate_ssm,
        gdn_intermediate_conv, k_new, v_new, index_key_new) -> None:
    tensors = dict(
        h_in_ssq=h_in_ssq, positions=positions, seq_lens=seq_lens, req_pool_indices=req_pool_indices,
        req_to_token=req_to_token, mamba_slot_map=mamba_slot_map, rope_cos_sin=rope_cos_sin,
        hc_attn_norm=hc_attn_norm, hc_attn_down=hc_attn_down, hc_attn_up=hc_attn_up,
        hc_mlp_norm=hc_mlp_norm, hc_mlp_down=hc_mlp_down, hc_mlp_up=hc_mlp_up, router=router,
        moe_w13=moe_w13, moe_w13_scale=moe_w13_scale, moe_w2=moe_w2, moe_w2_scale=moe_w2_scale,
        gdn_in_proj=gdn_in_proj, gdn_in_proj_scale=gdn_in_proj_scale, gdn_ba_proj=gdn_ba_proj,
        gdn_ba_proj_scale=gdn_ba_proj_scale, gdn_conv_weight=gdn_conv_weight, gdn_neg_a=gdn_neg_a,
        gdn_dt_bias=gdn_dt_bias, gdn_norm_weight=gdn_norm_weight, gdn_out_proj=gdn_out_proj,
        gdn_out_proj_scale=gdn_out_proj_scale, gdn_conv_state=gdn_conv_state,
        gdn_ssm_state=gdn_ssm_state, qsa_qkv_proj=qsa_qkv_proj, qsa_qkv_proj_scale=qsa_qkv_proj_scale,
        qsa_index_proj=qsa_index_proj, qsa_index_proj_scale=qsa_index_proj_scale,
        qsa_o_proj=qsa_o_proj, qsa_o_proj_scale=qsa_o_proj_scale, qsa_q_norm=qsa_q_norm,
        qsa_k_norm=qsa_k_norm, qsa_index_q_norm=qsa_index_q_norm, qsa_index_k_norm=qsa_index_k_norm,
        k_cache=k_cache, v_cache=v_cache, index_keys=index_keys, index_ring=index_ring,
        index_ring_positions=index_ring_positions, h_out_ssq=h_out_ssq,
        gdn_intermediate_ssm=gdn_intermediate_ssm, gdn_intermediate_conv=gdn_intermediate_conv,
        k_new=k_new, v_new=v_new, index_key_new=index_key_new)
    nreq = seq_lens.shape[0]
    assert h_in.shape == (nreq * L, HC_DIM) and h_in.dtype == torch.bfloat16
    device = h_in.device
    launch = _launcher(nreq, req_to_token.stride(0))
    table = _table(tensors, device)
    # guard block + mailboxes, zeroed: no pair of an earlier call carries this call's tag
    buf = torch.zeros(_scratch_words(nreq), dtype=torch.int32, device=device)
    buf[1:2].fill_(DEADLINE_LO)
    buf[2:3].fill_(DEADLINE_HI)
    launch(
        h_in.data_ptr(),
        h_out.data_ptr(),
        table.data_ptr(),
        buf.data_ptr() + EPOCH_PAD_WORDS * 4,
        buf.data_ptr(),
        0,
        stream=torch.cuda.current_stream(),
    )
