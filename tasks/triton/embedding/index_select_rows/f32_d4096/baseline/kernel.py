# SPDX-License-Identifier: BSD-3-Clause
# Triton rewrite of the pointwise indirect-load kernel that PyTorch (TorchInductor) generates
# for torch.index_select(table, 0, idx).
import triton
import triton.language as tl


@triton.jit
def _index_select_rows_kernel(table_ptr, idx_ptr, out_ptr, row_width, table_stride, out_stride,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < row_width
    src = tl.load(idx_ptr + row).to(tl.int64)
    vals = tl.load(table_ptr + src * table_stride + col, mask=mask, other=0.0)
    tl.store(out_ptr + row * out_stride + col, vals, mask=mask)


def run(*, table, idx, out):
    n_idx = idx.shape[0]
    row_width = table.shape[1]
    block = 1024
    grid = (n_idx, triton.cdiv(row_width, block))
    _index_select_rows_kernel[grid](table, idx, out, row_width, table.stride(0), out.stride(0), BLOCK=block)
