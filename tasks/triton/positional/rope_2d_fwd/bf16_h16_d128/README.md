# 2-D image RoPE, bfloat16, 16 x 16 token grid, 16 heads, head size 128

Rotary embedding for image tokens laid out on a height x width grid (row-major, `x[b, H*W, heads, d]`).
The first `d/2` channels of each head are rotated with the cos/sin of the token's row, the other
`d/2` with the cos/sin of its column, in NeoX style (rotate-half inside each half). Each cos/sin table
holds `d/2` entries per position. The reference works in float64, written independently of the
baseline (it reshapes to the grid, splits the head and rotates each half).

Baseline: AITER's Triton `rope_fwd_2d` (MIT; `_rope_fwd_2d_kernel_neox`), one program per
(image, head) holding the whole 256-token image in a `[256, 128]` tile. The kernel is copied
unchanged except that the masked cos/sin loads gain `other=0.0`, so the half of the row that a table
does not cover is defined as zero rather than depending on the backend's masked-load value (they are
summed into the result). The old task validated against AITER's HIP `rope_2d_fwd` operator; that
operator is not vendorable, so Triton is the baseline. The kernel needs a power-of-two token count
per image (`tl.arange`), which is why the grid is fixed at 16 x 16; the old task's 24 x 24 grid is
not a power of two and cannot run on this kernel, and its 32 x 32 grid (a `[1024, 128]` tile per
program) is not carried over. The old head-size-64 case is the sibling scenario `bf16_h16_d64`.

Workloads: batch is the only varying axis. Batches below 16 images (about 2 MB of traffic per image
against the 10 us timing floor) are correctness-only, as are 100 and 1000 images (unaligned). Timed
cases run from 16 to 2048 images (128x). The reference processes images in chunks to bound its
float64 temporaries.

Compare: default `baseline_relative` bound.

Risk for the GPU audit: the baseline holds a whole `[256, 128]` tile per program and may spill
registers; that is acceptable for a baseline but would show as low bandwidth.
