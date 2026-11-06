# Context-parallel LSE correction, bfloat16, 32 heads, head_dim 128, cp = 4

Under context parallelism (decode or prefill) every rank attends over its own slice of the KV
cache, producing a partial output and a log-sum-exp (LSE) per token and head. Before the partial
outputs are combined, each rank rescales its output by `exp(lse_rank - global_lse)`, where
`global_lse` is the stable log-sum-exp of all ranks' LSEs. This task is that rescale
(vLLM `correct_attn_out`) after the all-gather: it produces the corrected `[tokens, 32, 128]`
output in bfloat16 and the `[tokens, 32]` float32 global LSE.

- **Baseline:** vLLM's Triton kernel `_correct_attn_cp_out_kernel` (one program per token and
  head), vendored unchanged from `vllm/v1/attention/ops/common.py` (revision edbc4969); the allocating wrapper became
  `run()`, specialized to the natural-base, 3-D layout.
- **Scenario:** `lses` is float32 `[cp, tokens, heads]`; `lse_idx` (a scalar) selects this
  rank. About 2% of LSEs are `-inf` (a rank with no KV for that token), except rank 0, which keeps
  the global LSE finite; non-finite LSEs count as `-inf`, as in the kernel.
- **Workloads:** token counts from decode batches (1-48, correctness only) through prefill chunks
  of 2048-204800 tokens. The smallest performance case (2048 tokens, about 34 MB read and written)
  is estimated near the 10 us floor and must be confirmed on MI355X.
- **Bounds:** both outputs use the default `baseline_relative` bound.
