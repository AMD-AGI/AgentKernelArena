# Conv2d, BatchNorm2d (inference) and scale, float32 NCHW, 8 to 64 channels, 3x3

KernelBench level 2 problem 73 (`Conv2d_BatchNorm_Scaling`): `nn.Conv2d(8, 64, 3)` (stride 1, no padding,
with bias), then `nn.BatchNorm2d(64)` in eval mode, then multiplication by `scaling_factor = 2.0`. All of it
runs on a float32 `[batch, 8, height, width]` batch:

    z   = conv2d(x, conv_weight, conv_bias)                            # [batch, 64, height - 2, width - 2]
    out = scaling_factor * ((z - bn_running_mean[c]) * rsqrt(bn_running_var[c] + bn_eps) * bn_weight[c] + bn_bias[c])

Batch, height and width vary. Channels, kernel size, stride, padding, `bn_eps = 1e-5` and the NCHW
layout are fixed. Batch norm uses its running statistics, as `model.eval()` does. They are inputs of the
call, not values folded into the weights ahead of time.

- **Reference.** Float64 `torch.nn.functional.conv2d` followed by the batch norm and scale in float64
  (`reference.py`), in place, so the largest case keeps one float64 output-sized buffer (8.5 GB).
- **Inputs.** As in KernelBench, `x` is uniform on [0, 1). The convolution weight and bias follow
  `nn.Conv2d`'s default initialization, uniform on ±1/sqrt(72). KernelBench's module runs batch norm at
  its initial state (mean 0, variance 1, weight 1, bias 0), which hides any mistake in those terms. Here
  they are non-identity: weight uniform on [0.5, 1.5], bias and running mean uniform on ±0.25, running variance
  uniform on [0.05, 0.25].
- **Baseline.** A Triton rewrite of what TorchInductor generates for this module, not a vendored production
  kernel. `torch.compile(mode="max-autotune")` on ROCm times MIOpen (an external call that cannot be
  vendored) against Inductor's Triton `convolution2d` template (`torch/_inductor/kernel/conv.py`, PyTorch
  commit `d0f2af8b`, BSD-3-Clause, `baseline/LICENSE`). Inductor peels the convolution bias off and fuses it
  into the template's epilogue, together with `torch._decomp`'s inference batch norm,
  `(z - mean) * rsqrt(var + eps) * weight + bias`, and the scale.
  `baseline/kernel.py` is that template rendered for this scenario, with the epilogue written out:
  - implicit GEMM over NCHW, with one 128 (n*h*w) x 64 (output channels) tile per program;
  - a float32 accumulator, IEEE float32 `tl.dot` (TF32 off), and one `tl.dot` per tap with the 8 input
    channels padded to the template's minimum `BLOCK_K = 16`;
  - the nine taps unrolled with `tl.static_range`, where Inductor's template unrolls only 1x1 kernels
    and loops over the taps otherwise;
  - one fixed configuration from Inductor's conv config list (`ConvConfig(128, 128, 32, 2, 8)` scaled to
    64 output channels, with Inductor's 4-warp cap for non-1x1 kernels), launched with the
    `matrix_instr_nonkdim=16` that Inductor's ROCm config finalization adds (its `kpack=2`, which Triton
    resets to 1 on gfx950, is omitted). No autotuning runs.
- **Workloads.** KernelBench's `get_inputs` shapes (batch 32 to 128 at 64x64 and 128x128), plus batch 32 and
  64 at 224x224 and batch 16 and 64 at 512x512. Performance cases run from 32x64x64 (36 MB and 1.1 GFLOP,
  at least 12 us by bandwidth alone; the baseline's K padding to 16 makes it 2.3 GFLOP of float32 MFMA
  work, at least 14 us at the MI355X float32 matrix peak) to 64x512x512 (4.8 GB, 153 GFLOP), a 128x span. Single images,
  including one 224x224 inference image below the timing floor, odd non-square batches, the height
  minimum (one output row) and the width minimum (one output column) are correctness-only boundary
  cases. The largest case holds 0.5 GB of input, 4.3 GB of output and an 8.5 GB float64 reference per
  draw.
- **Comparison.** `sqnr` with `min_db: 126`, an override (see below).

## Why the bound departs from the default

The output is float32, and its error is dominated by the 72-term float32 convolution accumulation. The
batch norm amplifies it (by up to about 13x here), not the output rounding. Any other accumulation order
gives an independent error of the same size, and in some channels a different order's error is several
times the baseline's. A CPU simulation against the float64 reference compared exact float32 FMA
accumulations with a tap-major baseline followed by Inductor's epilogue (2x8x130x130, six draws):

| Implementation | `k: 4`, out-of-bound elements | `k: 8` | SQNR |
| --- | --- | --- | --- |
| baseline order | 0 | 0 | 135.5-137.6 dB |
| channel-major accumulation | 0.1-0.2% | 2e-6 to 2e-5 | 135.9-137.1 dB |
| reversed taps, batch norm folded into a scale and shift | 0.1-0.3% | 3e-6 to 3e-5 | 136.0-137.1 dB |
| batch norm folded into the weights | 0.03-0.05% | 0 to 1e-6 | 135.8-138.0 dB |
| float64 accumulation | 0 | 0 | 142.5-143.8 dB |
| baseline x (1 + 16 ulp) (QA control) | 47.6-57.3% | 11.0-17.6% | 120.2-120.4 dB |

The default bound (`k = 2`, no violations) is stricter still: on 4x8x66x66 it rejected even the
float64-accumulating implementation, which is more accurate than the baseline. Without a violation
allowance, `k = 8` still rejects the reorderings on a few elements in nearly every draw. `k: 8` with
`max_violation_fraction: 0.001` would accept them, but it would also accept 0.1% of the elements being
arbitrarily wrong. A whole-tensor SQNR bounds the total error energy instead. Correct float32
implementations stay at about 135 dB or more. The 16-ulp systematic shift of the QA control falls to
120.3 dB, and TF32 inputs to about 60 dB. A second, independent simulation on the boundary shapes
(2x8x3x70, 3x8x37x53, 1x8x64x64, 5x8x130x66; four draws each) found the same range: 135.0-140.4 dB for
per-tap, reversed-tap, fully sequential, oneDNN and weight-folded float32 implementations, and
120.2-120.4 dB for the shifted control. With a minimum of 126 dB, correct implementations keep about
9 dB (2.8x in error RMS) of margin. The shifted control misses the minimum by about 6 dB. Unwritten
(NaN-poisoned) elements always fail. A single gross error also fails: one output element per (image,
channel) plane wrong by its own magnitude gives about 54 dB on the largest case. Semantic slips fail by
tens of dB. A dropped `bn_eps` changes `rsqrt` by 2e-5 to 1e-4 and gives about 86 dB; a dropped convolution bias gives 15 dB.

## Risk

Not run on a GPU. The CPU pre-check covers the contract, the inputs and the reference; it cannot run
Triton. The baseline's index math and epilogue were checked by a CPU emulation of its tiles. Its compile,
fp32 MFMA numerics (expected to match the simulated 136 dB), timing and the control suite are unverified.
The smallest performance case, 32x64x64, is compute-heavy for its size and is expected to sit well above
the 10 us floor, but it is the one to watch.
