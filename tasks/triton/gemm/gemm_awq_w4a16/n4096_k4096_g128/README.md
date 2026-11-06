# AWQ int4 weight-only GEMM, N = 4096, K = 4096, group size 128

`out = x @ dequant(W)` with float16 activations `x` [m, 4096] and AWQ-quantized weights: `qweight` [4096, 512] int32
(eight 4-bit values per word along N), `qzeros` [32, 512] int32 (same packing; one zero point per 128-row group and
output column) and `scales` [32, 4096] float16. Output column `8*j + i` takes the nibble at bit offset
`4 * [0, 4, 1, 5, 2, 6, 3, 7][i]` of word `j` (AWQ's interleaved order); the dequantized weight is
`(nibble - zero) * scale`. The product is rounded to float16. 4096 x 4096 is a Llama-class hidden-to-hidden linear layer;
group size 128 is the standard AWQ setting.

**Inputs.** `inputs.py` draws standard-normal fp16 activations, packed words uniform over the whole int32 range (every
bit pattern is a valid weight or zero point), and scales `|N(0,1)| * 0.004 + 0.001` (the magnitude of AWQ group scales),
which keeps outputs well inside fp16 range. The old task's generator drew words from `[0, 2^31)` (top bit never set) with
toy shapes (K <= 256); both are replaced.

**Reference.** Unpacks nibbles with the AWQ order, forms `(nibble - zero) * scale` in float64 and multiplies in float64.
It replaces the old per-element Python loop reference with a vectorized form of the same math (checked against the loop
form on a small shape).

**Workloads.** Token counts for the layer. Weights are 8 MB packed, so the estimated baseline time (fp16 flops / 1e15 plus a
4 us floor) is below 10 us under about M = 192; decode-sized M <= 128 and the unaligned cases are correctness-only.
Performance cases run from 256 to 32768 tokens (128x). The old tasks' toy shapes (M 32 to 128, K <= 256) are far below the
timing floor and are dropped.

**Baseline.** vLLM's Triton `awq_gemm_kernel` (kernel body unchanged) with its default 32x32x32 tile launch, from
`vllm/model_executor/layers/quantization/awq_triton.py` (see `provenance`). Two deliberate points:
- vLLM launches with a split-K factor and then sums the `[split_k, M, N]` fp16 partials with `result.sum(0)`. That is
  PyTorch compute outside a kernel, which the candidate rules forbid (and the baseline must satisfy them when the target
  language is Triton). The baseline therefore uses split-K = 1 and the kernel stores directly into `out`.
- The kernel accumulates in the output dtype (float16), as upstream does: each `tl.dot` result is rounded to fp16.
  The baseline's error against the float64 reference is therefore larger than an fp32-accumulating kernel's; the default
  `baseline_relative` bound is scaled by that, and the baseline sanity check (SQNR >= 25 dB) is expected to pass with
  margin (about 40 dB or more at K = 4096).

**GPU audit risks.** The smallest performance case (M = 256) is estimated near 12 us; with the slow 32x32 tile baseline it
should measure well above 10 us, but if not, raise the smallest performance case. The fp16 accumulation makes the SQNR
margin the number to read first in the baseline correctness output. The kernel has no gfx950-specific feature; `required_arch`
is kept at gfx950 as the verified architecture.
