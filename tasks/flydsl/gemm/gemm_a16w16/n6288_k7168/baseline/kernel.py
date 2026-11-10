# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang contributors (sgl-project/sgl-amd-mono)
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Kimi-K3.1 decode/verify bf16 GEMM baseline: out[m, n] = a[m, k] @ b[n, k]^T.

The GEMM is sgl-amd-mono's LDS tile GEMM stage (``tile_mma.py``, the ``mma_tile`` that the
Kimi-K3.1 wide-row layer launch runs for its in_proj), launched standalone. Its per-row-bucket
configuration is the layer's own table (``CFGS`` / ``cfg_rows`` below, vendored from
``sgl_amd_mono/kimi_k31/wide.py``): a call with m token rows takes the smallest 8-token bucket
that holds m, pads the rows to it as the layer launch does, and runs that bucket's in_proj
configuration. Every in_proj configuration splits K across CTAs; the K slices' f32 partials
are summed in slice order and rounded to bfloat16 once by a second FlyDSL kernel
(``_sum_launcher``), which stands in for the layer's in-launch finish of the tile.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from . import buffer_ops as bo
from . import tile_mma
from .ops import THREADS, bf16_pair, rsrc

# ===========================================================================
# Production dispatch (sgl_amd_mono/kimi_k31/wide.py at 52486703). The tables and cfg_rows
# are unchanged; buckets() reads REQ_BUCKETS directly instead of req_buckets(), whose
# SGLANG_AMD_MONO_K31_WIDE_BUCKETS environment override is not vendored.
# ===========================================================================
REQ_BUCKETS = (
    2,
    4,
    8,
    16,
    32,
    48,
)  # requests a bucket holds; its rows are L x requests
SUPPORTED_L = (6, 8)
BUCKETS = tuple(8 * r for r in REQ_BUCKETS)
CFGS = {  # 8-token bucket -> (up, in_proj, o_proj, front, shared down): bm:bn:wm:wn:ks; an L-token
    # bucket of the same requests takes the same configs
    16: (
        "16:128:1:8:8",
        "16:128:1:8:4",
        "16:128:1:8:4",
        "16:128:1:8:8",
        "16:128:1:8:1",
    ),
    32: (
        "32:128:2:4:8",
        "32:128:2:4:4",
        "32:128:2:4:2",
        "32:128:2:4:8",
        "32:128:2:4:1",
    ),
    64: ("64:128:2:4:8", "64:128:2:4:4", "64:64:2:4:2", "64:128:2:4:8", "64:64:2:4:1"),
    128: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "64:64:2:4:1",
        "128:128:2:4:8",
        "64:64:2:4:1",
    ),
    256: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "128:64:2:4:1",
        "128:128:2:4:8",
        "128:64:2:4:1",
    ),
    384: (
        "128:32:4:2:4",
        "192:128:4:2:2",
        "192:64:4:2:1",
        "128:128:2:4:4",
        "128:64:2:4:1",
    ),
}
# bucket rows -> MoE front config, where the CFGS one is a poor fit: 17 column tiles x the row tiles x ks must stay
# within BLOCKS (one round), and a row tile past m is wasted work
FRONT_ROWS = {192: "64:128:2:4:4", 256: "96:128:2:4:4", 288: "96:128:2:4:4"}
# bucket rows -> in_proj config, by the same rule (50 column tiles x row tiles x ks within BLOCKS, no padded row tile)
INP_ROWS = {96: "96:128:2:4:4", 192: "192:128:4:2:4", 256: "128:128:2:4:2"}
# bucket rows -> latent up / shared down configs, by the same rule (o_proj: padded rows measured no cost, kept)
UP_ROWS = {
    192: "96:64:2:4:8",
    256: "128:64:4:2:8",
    288: "96:64:2:4:4",
    384: "96:128:2:4:8",
}
SDOWN_ROWS = {288: "96:128:2:4:1", 384: "192:64:4:2:1"}
NARROW = 8  # rows up to this run Kimi-K3's launches (the wide launch starts at 2 requests above 8 rows)


def buckets(L: int = 8) -> tuple:
    return tuple(L * r for r in REQ_BUCKETS)


def cfg_rows(b: int, L: int) -> str:
    """The CFGS entry of an L-token bucket of b rows."""
    assert b % L == 0 and b // L in REQ_BUCKETS, (b, L)
    up, inp, oproj, front, sdown = CFGS[8 * (b // L)]
    return (
        UP_ROWS.get(b, up),
        INP_ROWS.get(b, inp),
        oproj,
        FRONT_ROWS.get(b, front),
        SDOWN_ROWS.get(b, sdown),
    )


# ===========================================================================
# Selection for this task (the layer's in_proj GEMM, N=6288, K=7168)
# ===========================================================================
_N = 6288
_K = 7168
_L = 8  # tokens a request in a DSpark verify step (7 draft tokens + 1); the 8-token buckets
_GEMM = 1  # in_proj, the column of cfg_rows this GEMM takes


def _select(m):
    """(bucket rows, bm, bn, wm, wn, ks): the smallest bucket that holds m rows, as the layer
    launch picks it (rows up to NARROW, which production serves with Kimi-K3's narrow launch,
    take the smallest wide bucket)."""
    bucket = next(b for b in buckets(_L) if b >= m)
    bm, bn, wm, wn, ks = map(int, cfg_rows(bucket, _L)[_GEMM].split(":"))
    return bucket, bm, bn, wm, wn, ks


# ===========================================================================
# Consumer of the K-slice partials
# ===========================================================================
@functools.lru_cache(maxsize=None)
def _sum_launcher(ks, rows, n):
    """out[m, n] (bf16) = sum over s of part[s, :m, :] (f32 [ks, rows, n]), added in slice
    order and rounded to bfloat16 once; a thread takes 4 consecutive columns."""
    assert n % 4 == 0
    quads = n // 4  # 4-column groups a row
    plane = rows * n  # f32 elements a K slice

    @flyc.kernel(name=f"k31_mma_sum_n{n}_r{rows}_ks{ks}", known_block_size=[THREADS, 1, 1])
    def sum_kernel(part: Int64, out: Int64, m: Int32):
        g = fx.block_idx.x * THREADS + fx.thread_idx.x
        live = g < m * quads
        r_p = rsrc(part, ks * plane * 4)
        r_o = rsrc(out, m * (n * 2))  # sized: a masked store is an out-of-range offset
        v = fx.Vector.filled(4, 0.0, fx.Float32)
        for s in range_constexpr(ks):
            v = v + fx.Vector(
                bo.buffer_load(r_p, s * plane + 4 * g, vec_width=4, dtype=T.f32, mask=live)
            )
        words = fx.Vector.from_elements(
            [
                bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                bf16_pair(v[2], v[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        bo.buffer_store(words, r_o, 2 * g, mask=live)

    @flyc.jit
    def launch(part: Int64, out: Int64, m: Int32, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        sum_kernel(part, out, m).launch(
            grid=((m * quads + THREADS - 1) // THREADS, 1, 1), block=(THREADS,), stream=stream
        )

    return launch


# ===========================================================================
# Host entry point
# ===========================================================================
def run(*, a, b, out):
    """out[m, n] = sum_k a[m, k] * b[n, k], bfloat16 in and out, fp32 accumulation."""
    m = int(a.shape[0])
    bucket, bm, bn, wm, wn, ks = _select(m)
    assert ks > 1  # every in_proj configuration splits K; the partials need the consumer
    stream = torch.cuda.current_stream(a.device)
    x = a
    if m < bucket:
        # the layer launch runs a bucket's rows; padded rows are computed and never read
        x = torch.empty((bucket, _K), dtype=a.dtype, device=a.device)
        x[:m].copy_(a)
    part = torch.empty((ks, bucket, _N), dtype=torch.float32, device=a.device)
    gemm = tile_mma.standalone(bucket, _N, _K, bm, bn, wm, wn, ks)
    gemm(x.data_ptr(), b.data_ptr(), part.data_ptr(), stream=stream)
    _sum_launcher(ks, bucket, _N)(part.data_ptr(), out.data_ptr(), m, stream=stream)
