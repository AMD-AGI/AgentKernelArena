# GDN chunk output pass, bfloat16, 16 query/key heads, 32 value heads, K = V = 128, chunk 64

For each chunk of 64 tokens and value head (query/key head `h // 2`) the operator combines the state at
the start of the chunk with causal attention inside the chunk (Gated DeltaNet prefill, `chunk_fwd_o`):

```text
o = scale * ( exp(g) * (q h^T)  +  tril( (q k^T) * exp(g_i - g_j) ) v )
```

`h` is the start-of-chunk state produced by the chunk state pass, `g` the chunk-local cumulative
log-decay (restarting every 64 tokens) and `scale` = 128^-1/2. The last chunk is partial when
`n_tokens` is not a multiple of 64. Inputs and output are bfloat16, `g` float32.

- **Origin.** The chunk output kernel of SGLang's FLA package
  (`python/sglang/kernels/ops/attention/fla/chunk_o.py`, adapted from flash-linear-attention), the
  second GDN prefill kernel after the state pass, with Qwen3-Next dimensions. It merges the earlier
  SGLang task of this operator (shapes: 8/16 heads tensor-parallel splits, V = 64, batches) and the vLLM
  copy of the kernel; those variants are separate scenarios and are not part of this one.
- **Baseline.** The earlier task's Triton extraction (fixed-length, scalar-gate path, tile BK 128 x
  BV 64, 4 warps, 2 stages), wrapped in `run()` to write into the preallocated `o`. The chunk length is
  the kernel's `min(64, next_power_of_2(T))` as in the original wrapper.
- **Inputs.** Queries and keys have element standard deviation 128^-1/2 (the model L2-normalizes them),
  states 0.3, values 1.0; the earlier task used unit-variance queries and keys and states of 0.1. The
  gates are `0.03 * logsigmoid(x)` per token, accumulated within each chunk, so both the state term and
  the intra-chunk term contribute to every output.
- **Reference.** Float64, chunks processed in groups of 64 with no intermediate rounding; checked on CPU
  against an independent per-token loop. The kernel rounds the decayed attention matrix to bfloat16
  before multiplying by `v`, which belongs to the baseline's own error under the default
  `baseline_relative` bound (a CPU emulation has about 53 dB SQNR against the reference).
- **Workloads.** 1024 to 131072 tokens (128x span; about 40 KB of traffic per token, so the smallest
  case is expected near 14 us). Shorter prompts are correctness-only: 1 token (axis minimum), 100 and
  200 (partial chunk) and 1000 (unaligned). The largest case's float64 reference and comparison
  temporaries need tens of GB.
