# SPDX-License-Identifier: BSD-3-Clause
"""LayerNorm over (C, H, W) per sample as a split reduction in Triton.

Triton rewrite of what TorchInductor generates for F.layer_norm over a 4M-element row: the row is too
long for one program, so the reduction is split. Kernel 1 computes, per chunk, its mean and the sum of
squared deviations from that mean; kernel 2 merges the chunks per sample (Chan's parallel variance
combine, as triton_helpers.welford_combine does) into mean and reciprocal standard deviation; kernel 3
normalizes and applies the affine transform.
"""
import torch
import triton
import triton.language as tl

CHUNK = 8192


@triton.jit
def _partial_stats(x_ptr, mean_part_ptr, m2_part_ptr, row_len, num_chunks, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_len
    x = tl.load(x_ptr + row * row_len + offs, mask=mask, other=0.0)
    count = tl.minimum(row_len - chunk * BLOCK, BLOCK).to(tl.float32)
    mean = tl.sum(x, axis=0) / count
    dev = tl.where(mask, x - mean, 0.0)
    m2 = tl.sum(dev * dev, axis=0)
    out = row * num_chunks + chunk
    tl.store(mean_part_ptr + out, mean)
    tl.store(m2_part_ptr + out, m2)


@triton.jit
def _combine_stats(mean_part_ptr, m2_part_ptr, mean_ptr, rstd_ptr, row_len, num_chunks, eps,
                   BLOCK: tl.constexpr, CHUNK_LEN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < num_chunks
    means = tl.load(mean_part_ptr + row * num_chunks + offs, mask=mask, other=0.0)
    m2s = tl.load(m2_part_ptr + row * num_chunks + offs, mask=mask, other=0.0)
    # Every chunk holds CHUNK_LEN elements (row_len is a multiple of CHUNK_LEN), so the combine is unweighted.
    mean = tl.sum(means, axis=0) / num_chunks
    between = tl.where(mask, means - mean, 0.0)
    m2 = tl.sum(m2s, axis=0) + CHUNK_LEN * tl.sum(between * between, axis=0)
    var = m2 / row_len
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, 1.0 / tl.sqrt(var + eps))


@triton.jit
def _normalize(x_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, out_ptr, row_len, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_len
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x = tl.load(x_ptr + row * row_len + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * row_len + offs, (x - mean) * rstd * w + b, mask=mask)


def run(*, x, weight, bias, epsilon, out):
    batch = x.shape[0]
    row_len = x.numel() // batch
    assert row_len % CHUNK == 0
    num_chunks = row_len // CHUNK
    mean_part = torch.empty((batch, num_chunks), dtype=torch.float32, device=x.device)
    m2_part = torch.empty((batch, num_chunks), dtype=torch.float32, device=x.device)
    mean = torch.empty((batch,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((batch,), dtype=torch.float32, device=x.device)
    _partial_stats[(batch, num_chunks)](x, mean_part, m2_part, row_len, num_chunks, BLOCK=CHUNK, num_warps=8)
    _combine_stats[(batch,)](mean_part, m2_part, mean, rstd, row_len, num_chunks, epsilon,
                             BLOCK=triton.next_power_of_2(num_chunks), CHUNK_LEN=CHUNK, num_warps=4)
    _normalize[(batch, num_chunks)](x, weight, bias, mean, rstd, out, row_len, BLOCK=CHUNK, num_warps=8)
