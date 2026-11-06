# Triton rewrite of the reduction kernel that torch.compile (Inductor) generates for
# torch.sum(x, dim=1, keepdim=True) on a contiguous [B, N, M] float32 tensor.
# Original source: pytorch/pytorch, torch/_inductor (BSD-3-Clause).
"""Sum over the middle axis of a [B, N, M] tensor, in Triton."""
import triton
import triton.language as tl


@triton.jit
def _sum_dim1_kernel(in_ptr, out_ptr, n_rows, n_cols, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    xindex = pid_x * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < n_cols
    base = pid_b * n_rows * n_cols
    acc = tl.zeros([RBLOCK, XBLOCK], dtype=tl.float32)
    for r_off in range(0, n_rows, RBLOCK):
        rindex = r_off + tl.arange(0, RBLOCK)
        rmask = rindex < n_rows
        offs = base + rindex[:, None].to(tl.int64) * n_cols + xindex[None, :]
        vals = tl.load(in_ptr + offs, mask=rmask[:, None] & xmask[None, :], other=0.0)
        acc += vals
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_b * n_cols + xindex, total, mask=xmask)


def run(*, x, out):
    batch, n_rows, n_cols = x.shape
    XBLOCK, RBLOCK = 64, 32
    grid = (triton.cdiv(n_cols, XBLOCK), batch)
    _sum_dim1_kernel[grid](x, out, n_rows, n_cols, XBLOCK=XBLOCK, RBLOCK=RBLOCK)
