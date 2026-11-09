# Batched afp4wfp4 GEMM, 128 batch elements, N = 128, K = 512

`out[b] = dequant(x[b]) @ dequant(w[b])^T` for 128 batch elements. Activations `x` [batch, m, 256] and weights
`w` [batch, 128, 256] are MXFP4: two e2m1 codes per byte (the even K index is the low nibble) with uint8 e8m0
scales (`2^(s - 127)`), one per 32 values of K (`x_scales` [batch, m, 16], `w_scales` [batch, 128, 16]). The block
products are accumulated in float32 and the result is rounded to bfloat16. `m` is the only varying axis.

## Scenario and config citations

The scenario is the DeepSeek multi-head latent attention value up-projection (the absorbed `W_UV` batched
GEMM): each attention head multiplies its latent attention output by its slice of `W_UV`. DeepSeek-V3
`config.json` (Hugging Face, `deepseek-ai/DeepSeek-V3`): `num_attention_heads` 128 (batch), `kv_lora_rank` 512 (K),
`v_head_dim` 128 (N). `qk_nope_head_dim` is also 128 and `qk_rope_head_dim` 64; the absorbed W_UK projection is the
transposed shape (N = 512, K = 128). 128 heads is the single-rank (no tensor parallelism) head count; with tensor
parallelism the batch shrinks to 128 / TP.

AITER's gfx950 tuning has a dedicated table for this shape, `gfx950-BATCHED_GEMM-AFP4WFP4-N=128-K=512.json`
(and one for the transposed `N=512-K=128`), which is what the baseline uses. At the pinned SGLang and vLLM
commits the MXFP4 MLA path calls the sibling `batched_gemm_afp4wfp4_pre_quant` / `batched_gemm_a16wfp4` (the
activation quantized inside the call, see the `batched_gemm_a16wfp4` task); this operator is the same GEMM with the
activations already quantized by an upstream kernel (as in `gemm_afp4wfp4`), so the quantization is part of the
input and not timed. No caller of `batched_gemm_afp4wfp4` itself was found in those two projects.

## Reference and inputs

`reference.py` dequantizes both operands from codes and scales and multiplies per batch element in float64
(row-chunked). It is independent of the baseline and of the quantizer in `inputs.py`.

`inputs.py` draws normal activations (std 1) and weights (std 0.05), independently per batch element, and
quantizes them like AITER's dynamic MXFP4 quantization: the scale of a 1x32 group is the group maximum rounded to a
power of two minus the exponent of the largest e2m1 value, and codes round to nearest with ties to the even code,
saturating at 6. Weight std 0.05 gives outputs of std about 1.1, so the bfloat16 output rounding is exercised at
realistic magnitudes.

## Workloads

`m` is the number of tokens of a step: decode batches of a few hundred tokens up to batched prefill chunks.
The performance cases are memory-bound (about 68 KB moved per token across the 128 heads: 272 B of fp4 activations
and scales plus 256 B of bfloat16 output per head) and start at 384 tokens, estimated at 10.8 us (bytes / 4.5e12 B/s
plus 4 us; 256 tokens would be about 8.8 us and is correctness-only). 38400 tokens (estimated about 580 us) give the
required 100x span. The largest case holds 1.3 GB of inputs and a 5.0 GB float64 reference output. Boundary
cases: 1 token (axis minimum), 3 (non power of two), 17, 100, 1000 and 3000 (not multiples of 16, above 16).

## Baseline

AITER's Triton `_batched_gemm_afp4_wfp4_kernel` (`tl.dot_scaled` e2m1 x e2m1) with the gfx950 tuned table for
N = 128, K = 512 (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The kernel is
unchanged. Every row of the table has `NUM_KSPLIT` 1, so the split-K reduce kernel, `get_splitk` and the float32
partial-sum buffer are dropped. The lookup follows `get_gemm_config` with the custom bounds of `_get_config`
(M_LEQ rows up to 256, then `any`; the table has no M_LEQ_320 row); tile heights range from 16 (M <= 32) to
256 (M > 256), widths from 32 to 128. Upstream calls `remap_xcd(pid, GRID_MN)` and discards the result, so no XCD remapping takes
place; this is kept as is. The vendored files carry the upstream SPDX/copyright header; `kernel_repr.py` has none
upstream, so the repository MIT license applies to the inlined `make_kernel_repr`. No third-party origin is
recorded in any of the vendored files or their upstream directories.

## Comparison and memory

`tl.dot_scaled` needs gfx950. M = 384 is the floor of the performance cases (10.8 us by the bandwidth
estimate, which is optimistic for this tile shape). Output comparison keeps several float64 temporaries of the whole
output (5 GB each at 38400 tokens). The default `baseline_relative` bound is used; accumulation order inside a tile
differs between implementations only by float32 rounding, far below the bfloat16 output rounding.
