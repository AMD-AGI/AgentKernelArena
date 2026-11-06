# AWQ int4 dequantization, 4096 columns, group size 128

Unpacks AWQ-quantized weights: `qweight[K, 512]` (int32, eight 4-bit codes each, in AWQ's interleaved
order 0,4,1,5,2,6,3,7), `zeros[K/128, 512]` (packed the same way) and `scales[K/128, 4096]` (float16)
become `out[K, 4096] = (code - zero) * scale` in float16. Only `K` varies.

- **Production origin.** vLLM's Triton AWQ dequantization (`awq_triton.py`), used before GEMM when the
  fused AWQ kernels are unavailable.
- **Baseline.** vLLM's `awq_dequantize_kernel`, vendored unchanged (Apache-2.0) and launched with its
  default 32x32 block; the production wrapper became `run()` writing into the preallocated output.
- **Inputs.** Packed words are drawn over the full int32 range (including the sign bit), scales uniformly in
  [0.01, 0.2].
- **Comparison.** Default `baseline_relative`. The product of a 5-bit integer and a float16 scale is exact in
  float32, so the reference is the correctly rounded float16 result and the baseline error is zero; the
  bound reduces to one output rounding.
- **Workloads.** `K` is a multiple of 128. A single 4096-wide projection dequantizes in about 4 us of
  traffic, so the performance cases run from 3072 rows (hidden size of a 3B-class model) through the
  intermediate sizes of Llama-class models (14336, 28672, 53248) to stacked multi-layer weights up to
  307200 rows (100x span). The stacked cases are synthetic: they stand for dequantizing several layers'
  weights in one launch. Small `K` cases are correctness-only.
- **Risks.** Timing is tight at the smallest performance cases (about 31 MB of traffic); the largest case
  writes 2.5 GB and its comparison holds several float64 temporaries on the GPU.
- **Scope.** The scenario fixes group size 128, the standard AWQ configuration. The earlier task also ran
  tiny test shapes with group sizes 32 and 64 and bfloat16 scales; those are unit-test shapes of the vLLM
  kernel, not production sizes, and a different group size or scale dtype belongs in a sibling scenario.
