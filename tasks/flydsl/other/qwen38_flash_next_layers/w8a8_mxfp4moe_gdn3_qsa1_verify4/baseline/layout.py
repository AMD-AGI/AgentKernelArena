# SPDX-License-Identifier: Apache-2.0
"""Compile-time geometry and scratch (mailbox) layout of the Qwen3.8-Flash-Next
launch, kept free of FlyDSL so it is testable without a GPU.

A launch serves R requests x L verify tokens = S rows (S <= 32). Every activation
mailbox is row-major [S][n], row t = request t // L, draft step t % L."""

from .config import (
    GDN_HV,
    GDN_QKVZ,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    TOPK,
)
from .plan import BLOCKS, THREADS, WAVES, padded

assert (BLOCKS, THREADS, WAVES) == (256, 512, 8)  # one CTA per MI355X CU, co-resident
MOE_PAIR = TOPK + 1  # (row, expert) pairs per row: top-10 routed + the shared expert

# hyper-connection mix
MIX_KSPLIT = 10  # down GEMV split-K across CTAs (1024-wide K slices)
MIX_DOWN_RG = HC_LOWRANK // 16 + 1  # 20 row tiles + the combine-gate tile
MIX_DOWN_TASKS = MIX_DOWN_RG * MIX_KSPLIT  # 210
MIX_UP_TASKS = HIDDEN // 16  # 160 (each covers 16 outputs x 4 branches)
# sum-of-squares partials per (token, branch) published with a residual stream (one
# per 16-row group)
NPART = HIDDEN // 16

# GDN
IP_ROWS = GDN_QKVZ + 2 * GDN_HV  # 16480: [q k v | z | b | a]
IP_RG_MAIN = GDN_QKVZ // 16  # 1024 = 4 per CTA; the core tasks compute the b/a rows
MAX_CORE_SPLIT = 8  # the scratch's bound on core_split


def core_split(T):
    """V-row splits per value head of the GDN core: 4 for every T (8 would need 384
    CTAs)."""
    return 4


def core_geom(split):
    """(V rows per task, tasks, threads per V row, K elements per thread, xor offsets
    of a row reduce) of the GDN core at ``split``."""
    rows = 128 // split
    rpt = 512 // rows
    kpt = 128 // rpt  # state words per thread
    red_offs = tuple(rpt >> k for k in range(1, rpt.bit_length()))
    return rows, GDN_HV * split, rpt, kpt, red_offs


OUT_RG = HIDDEN // 16  # 160

# MoE
SHARED_EXPERT = N_EXPERTS  # index of the shared expert in the stacked arrays
ROUTER_ROWS = N_EXPERTS + 16  # 512 routed logits + shared gate + 15 zero rows
ROUTER_RG = ROUTER_ROWS // 16  # 33
# MXFP4 experts in SGLang's AITER layout. get_moe_weight_sizes(is_aiter_moe) pads the
# intermediate dim to the AITER alignment (w2 row 320 -> 384 bytes): FusedMoE w13
# [513, 2 * 768, 1280] (gate rows 0..639, zero rows to 767, up rows 768..1407, zero
# rows to 1535), w2 [513, 2560, 384] (K 640..767 zero); process_weights_after_loading
# applies shuffle_weight(w, (16, 16)) and e8m0_shuffles the scales over the flattened
# [E * rows, groups] (groups padded to 8). The kernel reads only the real rows / K
# (gate tiles 0..39, up tiles 48..87, down K chunks 0..4).
MOE_INTER_PAD = 768
# MXFP4 gate/up tasks per expert (40): gate tile n + up tile n + UP_TILE0
GU_PAIRS = MOE_INTER // 16
UP_TILE0 = MOE_INTER_PAD // 16  # 48: first up row tile of w13
DN_RG = HIDDEN // 16  # 160
GU_BYTES = 2 * MOE_INTER_PAD * HIDDEN // 2  # fp4 bytes per expert (w13)
GU_SGROUPS = HIDDEN // 32  # 80
GU_SBYTES = 2 * MOE_INTER_PAD * GU_SGROUPS
DN_BYTES = HIDDEN * MOE_INTER_PAD // 2  # w2: 384-byte rows
DN_SGROUPS = (MOE_INTER_PAD // 32 + 7) // 8 * 8  # 24
DN_SBYTES = HIDDEN * DN_SGROUPS


def max_slots(T):
    """Union of routed experts of T tokens + the shared expert."""
    return min(TOPK * T, N_EXPERTS) + 1


def max_groups(T, sg):
    """MoE down K-groups of ``sg`` expert slots (partials summed by the combine)."""
    return (max_slots(T) + sg - 1) // sg


def kphases(T):
    """K phases of the K=2560 activation GEMMs: [T][2560] bf16 rows fit the LDS for
    T <= 16; above, the CTA stages the activation in two K halves (accumulators of
    all its row tiles kept across the halves)."""
    return 1 if T <= 16 else 2


def gu_kphases(T):
    """K halves of the MoE gate/up: the staged [T][2560] bf16 rows + the route tables
    fit the LDS for T <= 28; above, a CTA stages one K half and publishes f32 gate/up
    partials (summed by the down staging)."""
    return 1 if T <= 28 else 2


def dn_sg(T):
    """Expert slots per MoE down K-group: the staged [T][SG * 640] bf16 mid of a
    group must fit the LDS."""
    return 4 if T <= 28 else 3


# QSA (full attention) layer
# [q | gate per head | k | v | index q (4 x 128) + index k (128)]
QP_ROWS = 24 * 512 + 512 + 512 + 640
QP_RG = QP_ROWS // 16  # 872
QP_K_OFF = 24 * 512
QP_V_OFF = QP_K_OFF + 512
QP_IDX_OFF = QP_V_OFF + 512
# top-512 selection: an exact top-512 per segment (16 blocks a thread), then a merge
SEG_BLOCKS = 8192
NSEG_MAX = 9
MAX_BLOCKS = SEG_BLOCKS * NSEG_MAX  # 73728 compressed blocks
MAX_CTX = MAX_BLOCKS * 4  # 294912 >= max_position_embeddings (262144) + verify tokens
ATT_CHUNK = 64  # tokens per attention split
ATT_CHUNKS = 2048 // ATT_CHUNK + 1  # 32 block chunks + 1 tail chunk
ATT_TASKS = ATT_CHUNKS  # per token; one task covers both kv heads
PART_STRIDE = 24 * 256 + 2 * 24  # per task: o[24][256], m[24], l[24]
MERGE_TASKS = 24  # one per (token, head)
PREP_Q_TASKS = 24  # CTAs [0, 24): one attention q head each (all tokens)
PREP_K_CTA = 24  # CTAs 24, 25: k head (norm+rope) + v -> KV cache
# CTAs [26, 26 + R): index q + index k ring / compress of request r (its tokens, in
# order)
PREP_IDX_CTA0 = 26


def sel2_cta0(T):
    """CTAs [256 - T, 256): merge of the segment top-512s of row t (long contexts)."""
    return 256 - T


SEL_STRIDE = 1026  # pairs per token: (block, loc) as 2 pairs each, count at pair 1024


def cnew_slots(L):
    """Compressed blocks completed inside one request's L verify tokens."""
    return L // 4 + 2


# per-layer pointer table slots
P_HCA_NORM, P_HCA_DOWN, P_HCA_UP, P_HCA_INJ = 0, 1, 2, 3
P_IP, P_CONV_W, P_NEG_A, P_DT_BIAS, P_NORM_W, P_OUT = 4, 5, 6, 7, 8, 9
P_CONV_STATE, P_SSM_STATE = 10, 11
P_HCM_NORM, P_HCM_DOWN, P_HCM_UP, P_HCM_INJ = 12, 13, 14, 15
P_ROUTER, P_GU, P_GU_S, P_DN, P_DN_S = 16, 17, 18, 19, 20
# QSA reuses P_IP (fused q/k/v/index projection) and P_OUT (o_proj)
P_QNORM, P_KNORM, P_IQNORM, P_IKNORM = 5, 6, 7, 8
P_KBUF, P_VBUF, P_RING, P_COMP = 10, 11, 21, 22
P_BA = 23  # GDN: row-major [b (48) ; a (48)] x HIDDEN bf16
# PLE (layer 1 only): fused into the start of the launch whose first layer carries it
P_PLE_W, P_PLE_NK, P_PLE_NQ, P_PLE_NC, P_PLE_CW, P_PLE_STATE = 24, 25, 26, 27, 28, 29
# GDN verify scratch: per-draft-step SSM states / conv windows (batch row 0)
P_ISSM, P_ICONV = 30, 31
# PLE verify scratch: per-draft-step short-conv windows (GDN rows leave P_RING free)
P_PLE_ISTATE = 21
PLE_ROWS = HC_DIM + HIDDEN  # [key_proj (10240) ; value_proj (2560)]
PLE_RG = PLE_ROWS // 16  # 800
PLE_STATE_LEN = 9
PLE_STATE_SLOT_BYTES = HC_DIM * PLE_STATE_LEN * 2
PLE_CH_PER_CTA = HC_DIM // 256  # 40
PLE_PARTS = HIDDEN // PLE_CH_PER_CTA  # 64 ple_h producers per branch
N_PTR = 40
# FP8 projections: F32 per-output-channel weight scales
P_IP_S, P_OUT_S, P_BA_S = 32, 33, 34
# QSA: SGLang keeps qkv_proj and indexer.index_qk_proj as separate FP8 params (rows
# [0, 832 * 16) / the rest of QP)
P_IP2, P_IP2_S = 35, 36
QP_RG_QKV = (24 * 512 + 512 + 512) // 16  # 832
BA_ROWS = 128  # SGLang pads in_proj_ba (96 rows) to 128 for AITER's CK GEMM
# global row (index n_layers of the table)
G_POS, G_LOC, G_REQ, G_SEQ = 0, 1, 2, 3
G_RTT, G_COS, G_ROPEPOS, G_MAMBA_MAP = 4, 5, 6, 7
G_PLE_EMB = 9  # (slot 8 is unused)
G_FIN_OUT = 10  # final mixer output x [T, HIDDEN] bf16 (final_mix launches)
G_HIN_SSQ, G_HOUT_SSQ = 11, 12  # f32 [T, 4, NPART] sum-of-squares of h_in / h_out
G_INPUT_IDS, G_EMBED_W = 13, 14
G_NGRAM_W, G_NGRAM_SCALE, G_NGRAM_CTX, G_NGRAM_META = 15, 16, 17, 18
# int64 [T, 2] per-draft-step n-gram history (batch row 0 of intermediate_context)
G_NGRAM_ICTX = 19
# MTP draft launches (first_src "mtp"): input fusion of the draft model
# bf16 [T, HC_DIM] target (or previous draft step) hyper-connection hidden states
G_MTP_HIN = 20
# pre_fc_norm_embedding [HIDDEN], pre_fc_norm_hidden [HC_DIM] (bf16)
G_MTP_PE, G_MTP_PH = 21, 22
G_MTP_FCE, G_MTP_FCH = 23, 24  # packed bf16 fc_embedding / fc_hidden [HIDDEN, HIDDEN]
NGRAM_HEADS, NGRAM_DIM, HEADS_PER_NGRAM = 16, 160, 8
PRELUDE_CTA0 = 224  # CTAs [224, 224 + T): PLE n-gram rows of verify row t
CONV_SLOT_BYTES = 10240 * 3 * 2  # bf16 [10240, 3] per mamba slot
SSM_SLOT_BYTES = 48 * 128 * 128 * 4  # f32 [48, 128, 128] per mamba slot
SSM_STEP_WORDS = 48 * 128 * 128
LAUNCH_SLOTS = 64  # tag = step * LAUNCH_SLOTS + launch_id + 1


# the guard block (``guard.py``) heads the scratch; the mailboxes follow it
GUARD_BYTES = 256
# the mailboxes a GDN / QSA layer of a middle verify launch uses (no PLE, MTP or
# model-level mixer)
GDN_REGIONS = frozenset((
    "a.mixd", "a.mixx", "a.injp", "m.mixd", "m.mixx", "m.injp", "ip", "ssq",
    "og_amax", "og", "h1", "h1ssq", "logit", "mid", "gupart", "dpart", "h2", "h2ssq",
))
QSA_REGIONS = frozenset((
    "qp", "q", "kvnew", "qi", "cnew", "ilog", "sel", "cand_k", "cand_bl", "part", "og",
    "a.mixd", "a.mixx", "a.injp", "m.mixd", "m.mixx", "m.injp", "og_amax", "h1",
    "h1ssq", "logit", "mid", "gupart", "dpart", "h2", "h2ssq",
))


def scratch_layout(plan, T: int, R: int = 1):
    """Byte offsets of the tagged-pair mailboxes of the layers of ``plan`` (kinds),
    one region per (layer, stage) after the guard block, and the total size.
    T = rows, R = requests."""
    regions = {}
    off = [GUARD_BYTES]

    for layer, kind in enumerate(plan):
        p = f"L{layer}."
        used = GDN_REGIONS if kind == "gdn" else QSA_REGIONS

        def add(name, pairs, p=p, used=used):
            if name in used:
                regions[p + name] = off[0]
                off[0] += padded(pairs * 8, 256)

        add("qp", T * QP_ROWS // 2)
        add("q", T * 24 * 256 // 2)
        add("kvnew", T * 1024 // 2)
        add("qi", T * 512 // 2)
        add("cnew", R * cnew_slots(T // R) * 64)  # [R][CN][64]
        add("ilog", T * MAX_BLOCKS)
        add("sel", T * SEL_STRIDE)
        # per (token, segment): key bits of the segment's top-512
        add("cand_k", T * NSEG_MAX * 512)
        add("cand_bl", T * NSEG_MAX * 512 * 2)  # (block, KV loc)
        add("part", T * ATT_TASKS * PART_STRIDE)
        add("og", T * 24 * 256 // 2)
        add("ple_kv", T * PLE_ROWS // 2)
        add("ple_h", T * HC_DIM // 2)
        add("ple_ssq", T * 4 * PLE_PARTS)
        for hc in ("a", "m"):
            add(hc + ".mixd", MIX_KSPLIT * T * HC_LOWRANK)
            add(hc + ".mixx", T * HIDDEN // 2)  # bf16 pairs
            add(hc + ".injp", MIX_KSPLIT * T * 4)
        add("ip", T * IP_ROWS // 2)  # bf16 pairs
        # per (token, head, split) sum of squares of the GDN output
        add("ssq", T * GDN_HV * MAX_CORE_SPLIT)
        # per (row, producer task) max |og| (the FP8 activation scale)
        add("og_amax", T * GDN_HV * MAX_CORE_SPLIT)
        add("h1", T * HC_DIM // 2)  # bf16 pairs
        add("h1ssq", T * 4 * NPART)
        add("logit", T * ROUTER_ROWS)  # f32
        # (row, expert) pairs p = t * 11 + rank: bf16 h = silu(g) * u [p][640], and
        # under gu_kphases 2 the f32 K-half-0 gate/up partials [p][1280]
        add("mid", MOE_PAIR * T * MOE_INTER // 2)
        if gu_kphases(T) == 2:
            add("gupart", MOE_PAIR * T * 2 * MOE_INTER)
        add("dpart", max_groups(T, dn_sg(T)) * T * HIDDEN)  # f32 [group][T][2560]
        add("h2", T * HC_DIM // 2)  # bf16 pairs
        add("h2ssq", T * 4 * NPART)
        # MTP input fusion output (layer-0 input of a draft launch)
        add("mtp_h", T * HC_DIM // 2)
        add("mtp_ssq", T * 4 * NPART)
    return regions, off[0]
