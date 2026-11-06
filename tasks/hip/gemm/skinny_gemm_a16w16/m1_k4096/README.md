# Skinny GEMM (GEMV), bfloat16, one token, K = 4096

`out[1, N] = x[1, 4096] @ weight[N, 4096]^T`, bfloat16 in and out, no bias. This is the decode-time
projection (QKV, MLP, lm_head) where one token multiplies a large weight matrix, so the kernel is
bound by streaming the weight. The variable axis is `n_out`. Sibling scenarios cover 2 to 4 tokens, hidden sizes 2048 and 7168 and float16
(`m1_k2048`, `m1_k7168`, `m2_k7168`, `m3_k7168`, `m4_k7168`, `fp16_m1_k4096`). The old task's bias
variants are not covered because the schema has no optional tensors.

**Baseline.** vLLM's ROCm `wvSplitK` (`csrc/rocm/skinny_gemms.cu`), a wave64 split-K GEMV with an
LDS-resident-activation path, vendored as `baseline/kernel.hip`. Kernel bodies are unchanged. The
repository's previous copy of this operator was taken from an older vLLM; its `wvSplitK` kernels are
identical to the recorded revision. Removed: the bias argument (the schema has no optional tensors),
the wave32 (gfx11/gfx12) dispatch, `LLMM1`, `wvSplitKrc`, `wvSplitKQ` and the helper headers. The
host launcher writes the preallocated output and takes the CU count from the device properties.
The dispatch keeps the 1 to 4 token cases of `wvSplitK`; this scenario uses one token.

**Reference.** float64 matmul of the logical inputs. Bounds are the default `baseline_relative`; the
previous task's fixed `atol=rtol=0.05` gate is not carried over.

**Workloads.** `n_out` 16 to 512000. Output widths come from the previous task's captured cases
(128, 37984, hidden-4096 projections) and the vocabulary sizes of its lm_head cases (151936, 201088),
plus intermediate and larger sizes that keep the 100x span needed for scoring; 320000 and 512000 are
stress sizes (a weight of up to 4.2 GB, about 17 GB with the float64 reference), not captured shapes.
Performance cases start at N = 5120 (about 42 MB of weight, estimated 13 us); narrower outputs are
correctness-only because the weight read is below the 10 us floor.

## Baseline adaptation

The vendored `baseline/kernel.hip` was adapted to build against the torch 2.11 ROCm headers: the dtype checks use `at::kHalf`/`at::kBFloat16` instead of `torch::kFloat16`/`torch::kBFloat16`, and a stray `return out_c;` fragment (left over from removing the other skinny-GEMM ops) that broke the build was deleted. The kernel bodies and algorithm are unchanged.
