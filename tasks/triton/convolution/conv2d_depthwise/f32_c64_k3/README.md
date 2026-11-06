# Depthwise conv2d, float32 NCHW, 64 channels, 3x3

KernelBench level 1 problem 82 (`conv_depthwise_2D_square_input_square_kernel`): `nn.Conv2d(64, 64, 3,
groups=64, bias=False)`, stride 1, no padding, on a float32 `[batch, 64, height, width]` batch. Each output
`out[n, c, h, w]` is the 9-tap dot product of channel `c`'s 3x3 filter with the input window at `(h, w)`;
the output is `[batch, 64, height - 2, width - 2]`. Batch, height and width vary; channels, kernel size,
stride, padding and the NCHW layout are fixed.

- **Reference.** Float64 direct sum over the nine taps (`reference.py`), a few images at a time so the
  largest case keeps one float64 output-sized buffer (8.5 GB). It does not call a PyTorch convolution.
- **Inputs.** As in KernelBench, `x` is uniform on [0, 1). The weights follow `nn.Conv2d`'s default
  initialization for a fan-in of 9, uniform on ±1/3.
- **Baseline.** A Triton rewrite, not a vendored production kernel. `torch.compile(mode="max-autotune")` on
  ROCm times MIOpen (an external call that a task cannot vendor) against Inductor's Triton alternative, the
  grouped implicit-GEMM `convolution2d` template in `torch/_inductor/kernel/conv.py`. With one channel per
  group, that template uses only 1 of the 16 or more reduction lanes and 1 of the 16 or more output-channel
  lanes of each `tl.dot`. The baseline therefore follows Inductor's direct depthwise template
  (`torch/_inductor/kernel/templates/triton_depthwise_conv.py.jinja`, PyTorch commit `d0f2af8b`, BSD-3-Clause,
  `baseline/LICENSE`). It accumulates one multiply-add per tap into a float32 tile. That template is 1-D and
  channels-last. Here it is applied to 2-D NCHW: each program computes a 16x64 output tile of one
  (image, channel) plane, with the 3x3 taps unrolled and a fixed configuration (4 warps).
- **Workloads.** KernelBench's `get_inputs` shapes (batch 4 to 16 at 128x128, 256x256 and 512x512), plus
  batch 8 at 128x128, batch 32 at 224x224 and batch 32 and 64 at 512x512. Performance cases run from
  8x128x128 (66 MB moved, an estimated 19 us) to 64x512x512 (8.6 GB), a 128x span. KernelBench's smallest
  input, 4x128x128, moves 33 MB, close to the 10 us floor, so it is checked for correctness only. Single
  images, an odd non-square batch, the height minimum (one output row) and the width minimum (one output
  column) are correctness-only boundary cases. The largest case holds 4.3 GB of input, 4.3 GB of output
  and an 8.5 GB float64 reference per draw.
- **Comparison.** `sqnr` with `min_db: 130`, an override (see below).

## Why the bound departs from the default

The output is float32, and its error is the rounding of a 9-term float32 accumulation, not the output
rounding. Any other tap order, or the absence of fused multiply-adds, gives an independent error of the same
size. A CPU simulation against the float64 reference (2x64x130x130, three draws) compared exact float32 FMA
accumulations in other orders with a row-major-tap baseline:

| Implementation | default bound, out-of-bound elements | `k: 8` | SQNR |
| --- | --- | --- | --- |
| baseline order | 0 | 0 | 144.6-144.9 dB |
| reversed taps | 1.1-1.3% | 1e-6 to 3e-6 | 144.5-144.7 dB |
| column-major taps | 0.6-0.9% | 0 to 5e-7 | 144.7-145.0 dB |
| separate multiply and add | 1.0-1.3% | 0 | 143.5-143.9 dB |
| three row sums, then added | 0.4-0.5% | 0 | 145.5-145.9 dB |
| float64 accumulation | 0 | 0 | 151.9-152.0 dB |
| baseline x (1 + 16 ulp) (QA control) | 91-92% | 60-65% | 120.4 dB |

The default `baseline_relative` bound (`k = 2`, no violations) rejected every reordering. Without a
violation allowance, `k = 8` still rejected the reversed order on single elements. `k: 8` with
`max_violation_fraction: 0.001` would accept them, but it would also accept 0.1% of the elements being
arbitrarily wrong. A whole-tensor SQNR bounds the total error energy instead. Every correct float32 order
stays at about 143 dB or more; a second, independent simulation on the boundary shapes (3x64x3x70,
2x64x37x53, 1x64x128x128, 5x64x200x99; three draws each) found 142.8-144.7 dB for forward and reversed
FMA orders, separate multiply and add, and oneDNN's float32 depthwise convolution. The 16-ulp systematic
shift of the QA control falls to 120.4 dB. Reduced-precision inputs fall much lower: float16 inputs give
70-71 dB, TF32 63 dB and bfloat16 53 dB. With a minimum of 130 dB, correct implementations keep a
margin of about 13 dB (4.4x in error RMS). The shifted control misses the minimum by 10 dB. Unwritten (NaN-poisoned) elements always fail. An error concentrated in a few elements is
still detected: one wrong output pixel per channel plane, at the output's own magnitude, gives about 54 dB
on the largest case.

## Risk

Not run on a GPU. The CPU pre-check covers the contract, the inputs and the reference; it cannot run
Triton. The baseline's index math was checked by a CPU emulation of its tiles, but its compile, timing
and the control suite are unverified. The baseline is memory bound. Each performance case is estimated
from bytes moved; 8x128x128 is the one closest to the 10 us floor.
