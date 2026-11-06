# LLaMA feed-forward gate with fused RMSNorm, float16, hidden 4096, intermediate 14336

For each token row, applies RMSNorm with weight `rms_w`, projects it with two weight matrices and combines them with a
SiLU gate: `out = silu(h @ w1.T) * (h @ w3.T)` where `h = rms_w * x * rsqrt(mean(x^2) + epsilon)`. The kernel folds
the normalization into the GEMM (the weight is applied to `x` before the dot products and the row scale after).

**Production origin.** The gate and up projections of a Llama-3-8B MLP (hidden 4096, intermediate 14336); the
previous task package used the TritonBench `llama_ff_triton` kernel at toy dimensions (hidden 64). Weights are in the
usual `[out, in]` layout.

**Scenario choice.** The previous kernel had an fp8e5 bit-pattern weight mode, but its harness (and production
Llama) use float16 weights; this task covers float16 only.

**Baseline.** The TritonBench `ff_llama` kernel, with a launcher that picks the row tile from the batch size (16 rows up
to 16 tokens, then 32, 64, 128; 64 x 64 tiles over N and K). The previous fused single-block-per-row variant required
K to equal one tile and does not scale to hidden 4096, so it is dropped. The N and K tails are not masked, which is
valid because 14336 and 4096 are multiples of 64. The tile choice is untuned: the baseline is a faithful but plain
kernel and leaves room for a candidate.

**Runtime adaptation.** The MI355X audit (Triton 3.8.0 ROCm build) failed to lower the original loop-carried `[1, BLOCK_K]` pointer tensor for `rms_w` (`LLVM Translation failed ... unrealized_conversion_cast` at `rms_w_ptr`). The kernel now loads the RMSNorm weight as a 1-D vector addressed from the loop index (`rms_w_ptr + (kb * BLOCK_K + offs_k) * stride`) and broadcasts it; the algorithm, tiling and numerics are unchanged. Not re-run on a GPU here.

**Numerics.** The kernel rounds `rms_w * x` to float16 before the dot products (as upstream does), which the default
baseline-relative bound accounts for. Weights use a standard deviation of 0.02 so that gate values stay far from
float16 overflow.

**Workloads.** Token counts through the block. The two weight matrices (235 MB) are read at every batch size, so even
one token takes the baseline over 50 us and decode batches are performance cases too; performance spans 1 to 4096
tokens. The largest case is bound by compute (about 1 PFLOP/s at best).
