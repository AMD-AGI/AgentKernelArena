# Transpose of sequence and batch, float32 `[seq, 4, 32, 128]`

`out = x.transpose(0, 1).contiguous()`: the sequence and batch dimensions of a float32
activation swap places and the result is materialized contiguously, `out[b, s, h, d] = x[s, b, h, d]`.
The sequence length is the varying axis (1 to 32768); batch 4, 32 heads and head size 128 are fixed.
This is the layout change between sequence-first `[seq, batch, heads, dim]` and batch-first
`[batch, seq, heads, dim]` activations. It is the dimension pair (0, 1) that the earlier task's
module swaps by default; the sibling scenario `f32_4d_swap12` covers the (1, 2) swap.

## Reference and comparison

The reference is the PyTorch transpose itself. A transpose only moves values, so the float32 result
is exact and the output is compared bitwise (`exact`, a justified departure from the default bound); no tolerance is involved.

## Baseline

A HIP kernel from this repository's earlier `hip2hip/Transpose` task: one thread per output element
decomposes its contiguous output index with the output strides and re-composes the input offset with
the permuted input strides, in a grid-stride loop. Changes for the v3 interface: specialized to
4-D float32 and dimensions 0 and 1, registered as the `run(x, out)` op writing into the
preallocated output, and the stride arrays are passed to the kernel by value. The earlier version
copied them to device tensors on every call, which cannot be captured in a CUDA graph. The kernel
is deliberately a plain generic gather, not a tiled transpose, so there is room to improve on it.

## Workloads

The earlier task covered five fp32 shapes from 4x8x16x32 up to 64x128x256x512 (4 GB), scaling every
dimension together. This scenario fixes the batch, head and head-size dimensions and varies the
sequence length. Performance cases span sequence
lengths 256 to 32768 (128x; 34 MB to 4.3 GB of traffic, each at or above the 10 us floor). Shorter
sequences, the minimum, and the unaligned lengths 17, 100 and 1000 are correctness-only. The largest
case holds a 2.1 GB input and a 2.1 GB output.
