# Batched bf16 GEMM, 128 batch elements, N = 128, K = 512

`out[b] = x[b] @ w[b]^T` for 128 batch elements: bf16 inputs, float32 accumulation, bf16 output, no bias.
The reference is the same product in float64.

**Production origin.** The scenario is the DeepSeek multi-head latent attention value up-projection: after
attention, each of the 128 heads multiplies its latent output (kv_lora_rank = 512) by that head's W_UV slice
into a 128-wide value head, which AITER serves with this batched kernel. Weights are stored as
`w[b, n, k]` as the AITER op takes them. The old tasks (an AITER extraction, a PyTorch-oracle task and a
vLLM fp16 NN `bmm`) used synthetic shapes (B = 16, 1280 x 8192, 8192 x 1024); the vLLM task is a different
(NN, fp16) layout and is dropped. The planned scenario `b128_n512_k128` (the absorbed W_UK projection) was
replaced by this one: its output is four times larger, and the framework's comparison keeps several
float64 temporaries of the whole output, which at the 100x token span needed here would not fit.

**Workloads.** Token counts from decode (correctness-only, below the 10 us timing floor) to batched
prefill. The performance cases are memory-bound (about 163 KB moved per token) and start at 256 tokens,
estimated at 13 us; 25600 tokens give the required 100x span. 3000 tokens exercises the config row
used between the two tuned entries.

**Baseline.** AITER's Triton `_batched_gemm_bf16_kernel` with its gfx950 tuned configuration (256 x 256 x 64
tiles for M >= 4096, otherwise 128 x 128 x 32). See `provenance` in `task.yaml`; the kernel is unmodified.
`baseline/LICENSE` is the AITER MIT license.
