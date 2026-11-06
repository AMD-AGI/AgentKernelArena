# 2-D image RoPE, bfloat16, 16 x 16 token grid, 16 heads, head size 64

Rotary embedding for image tokens laid out on a height x width grid (row-major, `x[b, H*W, heads, d]`).
The first `d/2` channels of each head are rotated with the cos/sin of the token's row, the other
`d/2` with the cos/sin of its column, in NeoX style (rotate-half inside each half). Each cos/sin table
holds `d/2` entries per position. The reference works in float64, written independently of the
baseline (it reshapes to the grid, splits the head and rotates each half).

Baseline: AITER's Triton `rope_fwd_2d` (MIT; `_rope_fwd_2d_kernel_neox`), one program per
(image, head) holding the whole 256-token image in a `[256, 64]` tile. The kernel is copied
unchanged except that the masked cos/sin loads gain `other=0.0`, so the half of the row that a table
does not cover is defined as zero rather than depending on the backend's masked-load value (they are
summed into the result). The old task validated against AITER's HIP `rope_2d_fwd` operator; that
operator is not vendorable, so Triton is the baseline. The kernel needs a power-of-two token count
per image (`tl.arange`), which is why the grid is fixed at 16 x 16; other grids are separate
scenarios.

Workloads: batch is the only varying axis. Batches below 32 images (about 1 MB of traffic per
image against the 10 us timing floor) are correctness-only, as are 100 and 1000 images (unaligned).
Timed cases run from 32 to 4096 images (128x). The reference processes images in chunks to bound its
float64 temporaries. The head-size-128 variant is the sibling scenario `bf16_h16_d128`.

Compare: default `baseline_relative` bound.

Risk for the GPU audit: the baseline holds a whole `[256, 64]` tile per program and may spill
registers; that is acceptable for a baseline but would show as low bandwidth.
