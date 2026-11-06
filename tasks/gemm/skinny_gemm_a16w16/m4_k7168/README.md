# Skinny GEMM (GEMV), bfloat16, 4 tokens, K = 7168

`out[4, N] = x[4, 7168] @ weight[N, 7168]^T`, bfloat16 in and out, no bias. This is the decode-time
projection (QKV, MLP, lm_head) where 4 tokens multiplies a large weight matrix, so the kernel is bound
by streaming the weight. The variable axis is `n_out`. It is one of a family of sibling scenarios
(`m1_k4096`, `m1_k2048`, `m1_k7168`, `m2_k7168`, `m3_k7168`, `m4_k7168`, `fp16_m1_k4096`) that
together cover the old task's token counts 1 to 4, hidden sizes 2048, 4096 and 7168, and float16.
The bias variants (hidden 2880 and 4096) are not covered because the schema has no optional tensors.

**Baseline.** vLLM's ROCm `wvSplitK` (`csrc/rocm/skinny_gemms.cu`), identical to `m1_k4096/baseline`
(see that task's README for what was removed); the dispatch handles 1 to 4 tokens and any dtype.

**Reference.** float64 matmul of the logical inputs. Bounds are the default `baseline_relative`.

**Workloads.** `n_out` 16 to 256000. Output widths come from the old task's captured cases and the
vocabulary sizes of its lm_head cases, plus intermediate and larger sizes that keep the 100x span needed
for scoring; sizes above 201088 are stress sizes, not captured shapes. Performance cases start at
N = 2560 (about 37 MB of weight, estimated 12 us); narrower
outputs are correctness-only because the weight read is below the 10 us floor.

## Baseline adaptation

The vendored `baseline/kernel.hip` was adapted to build against the torch 2.11 ROCm headers: the dtype checks use `at::kHalf`/`at::kBFloat16` instead of `torch::kFloat16`/`torch::kBFloat16`, and a stray `return out_c;` fragment (left over from removing the other skinny-GEMM ops) that broke the build was deleted. The kernel bodies and algorithm are unchanged.
