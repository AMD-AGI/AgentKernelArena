# SPDX-License-Identifier: BSD-3-Clause
# Triton rewrite of the pointwise broadcast-add kernel that PyTorch (TorchInductor) generates
# for x + F.embedding(arange(seq_len), pos_embedding).
import triton
import triton.language as tl


@triton.jit
def _position_add_kernel(x_ptr, pos_ptr, out_ptr, seq_len, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    position = row % seq_len
    col = tl.arange(0, BLOCK)
    mask = col < hidden
    x = tl.load(x_ptr + row * hidden + col, mask=mask, other=0.0)
    p = tl.load(pos_ptr + position * hidden + col, mask=mask, other=0.0)
    tl.store(out_ptr + row * hidden + col, x + p, mask=mask)


def run(*, x, pos_embedding, out):
    batch, seq_len, hidden = x.shape
    block = triton.next_power_of_2(hidden)
    _position_add_kernel[(batch * seq_len,)](x, pos_embedding, out, seq_len, hidden, BLOCK=block)
