# Triton rewrite of the kernels torch.compile (Inductor) generates for
# torch.nn.functional.cross_entropy(logits, targets) with mean reduction:
# log-softmax over the class row, gather of the target log-probability and a mean.
# Original source: pytorch/pytorch, torch/_inductor (BSD-3-Clause).
"""Per-row and mean cross entropy of [batch, num_classes] float32 logits, in Triton."""
import torch
import triton
import triton.language as tl

_PARTIALS = 256


@triton.jit
def _row_loss_kernel(logits_ptr, targets_ptr, loss_ptr, num_classes, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * num_classes
    target = tl.load(targets_ptr + row)
    # Online log-sum-exp over BLOCK-wide column tiles.
    running_max = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, num_classes, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < num_classes, other=float("-inf"))
        new_max = tl.maximum(running_max, x)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(x - new_max)
        running_max = new_max
    row_max = tl.max(running_max, axis=0)
    total = tl.sum(running_sum * tl.exp(running_max - row_max), axis=0)
    picked = tl.load(base + target)
    tl.store(loss_ptr + row, row_max + tl.log(total) - picked)


@triton.jit
def _partial_sum_kernel(loss_ptr, partial_ptr, rows, BLOCK: tl.constexpr, NUM_PARTIALS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range((pid * BLOCK), rows, NUM_PARTIALS * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(loss_ptr + offs, mask=offs < rows, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _finalize_kernel(partial_ptr, out_ptr, rows, NUM_PARTIALS: tl.constexpr):
    offs = tl.arange(0, NUM_PARTIALS)
    total = tl.sum(tl.load(partial_ptr + offs), axis=0)
    tl.store(out_ptr, total / rows)


def run(*, logits, targets, losses, out):
    rows, num_classes = logits.shape
    partials = torch.empty(_PARTIALS, dtype=torch.float32, device=logits.device)
    _row_loss_kernel[(rows,)](logits, targets, losses, num_classes, BLOCK=1024, num_warps=4)
    _partial_sum_kernel[(_PARTIALS,)](losses, partials, rows, BLOCK=1024, NUM_PARTIALS=_PARTIALS)
    _finalize_kernel[(1,)](partials, out, rows, NUM_PARTIALS=_PARTIALS)
