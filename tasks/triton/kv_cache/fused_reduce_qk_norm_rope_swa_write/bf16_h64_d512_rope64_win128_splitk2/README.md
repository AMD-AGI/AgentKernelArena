# Fused split-K reduce + QK norm + RoPE + sliding-window write, 64 heads, head size 512, window 128

The post-GEMM chain of a DeepSeek-V4-style sliding-window attention layer, as one launch. The query
projection `wq_b` returns two split-K partial sums `q [2, tokens, 64 * 512]` (bfloat16); the kv projection
returns `kv [tokens, 512]`.

- **Query.** The partials are added in float32; each of the 64 heads is RMS-normalized over its 512 values
  **without weight** (`x * rsqrt(mean(x^2) + q_eps)`); the interleaved (GPT-J style) rotary embedding rotates
  the last 64 values with the cos/sin row of `positions[t]`; the result is `q_out [tokens, 64, 512]`.
- **Key/value.** Each row named by `write_indices` is RMS-normalized with `kv_norm_weight` (`kv_eps`), its
  last 64 values are rotated, and the row is stored back into `kv` (the tensor is `mutable`).
- **Sliding-window write.** The same normalized, rotated row goes to
  `swa_kv[state_slot_mapping[batch_id_per_token[row]], positions[row] % 128]`, the ring buffer of one sequence's
  last 128 positions. `swa_kv` is `mutable`; cells nobody writes keep their zero initial value.

`kv` and `swa_kv` are compared in full, so rows that are not written stay unchanged.

## Baseline

AITER's Triton `_fused_reduce_qk_norm_rope_swa_write_kernel` (MIT, `baseline/LICENSE`): grid
`(cdiv(M, BLOCK_SIZE_M), heads + 1)`, where the extra program column handles the kv row. The kernel, its
two helpers and `_pick_block_size_m` are copied; the two rotation helpers (`rope.py`) and `make_kernel_repr`
(`utils/_triton/kernel_repr.py`) are inlined; the launch code is reduced to `run()`. The kernel file carries the
AITER SPDX/copyright header, kept verbatim; `rope.py` has the same header; `kernel_repr.py` has none, so the
repository MIT license applies to it. No third-party origin note was found; `git log --follow` ends at the
AITER commit "[Triton] [ATOM] DSV4 fusions phase 1". Another variant of this fusion with an FP8 SWA store is
in SGLang (`fused_qk_norm_rope_store.py`); it was not vendored because it writes the packed FP8 pool and uses
no ring buffer, a different operator.

## Scenario and config citations

DeepSeek-V4-Flash `config.json` (Hugging Face), the fields vLLM's `DeepseekV4Attention` reads:
`num_attention_heads` 64 (all local at TP 1), `head_dim` 512, `qk_rope_head_dim` 64, `sliding_window` 128,
`rms_norm_eps` 1e-6 (both epsilons). Q is normalized per head without weight (the AITER test and vLLM's fused
Q-norm test use the same weight-free query norm) and K/V with a learned weight, which is drawn uniformly in
(0.5, 1.5). cos/sin tables hold `pos * 10000^(-2i/64)` rows for 131072 positions in bfloat16. The fields above are those of
the DeepSeek-V4-Flash `config.json` on Hugging Face (`deepseek-ai/DeepSeek-V4-Flash`): `num_attention_heads` 64,
`head_dim` 512, `qk_rope_head_dim` 64, `sliding_window` 128, `rms_norm_eps` 1e-06, `rope_theta` 10000. Its
`rope_scaling` is YaRN (`factor` 16, `original_max_position_embeddings` 65536). The cos/sin tables are task inputs,
so the YaRN scaling of the real checkpoint changes only their values, not the kernel's work.

## Workloads and inputs

Decode batches of a V4-style serving engine with three speculative tokens: each request contributes up to four
consecutive positions (`max_position - 4` random starts, so most positions exceed the window and the ring index
`position % 128` wraps), owns a distinct state slot, and `write_indices` is a random permutation of the rows.
This is the AITER test's routing (a permutation, unique slots). It is also what keeps the task deterministic:
the kernel writes ring cells from unordered programs, so two written tokens of one sequence that were 128 apart
would race; a sequence here spans at most 4 positions. Prefill chunks with long sequences are therefore not part
of this scenario (the production caller must filter them with `write_indices`, which this task does not model).

Per token the operator moves about 195 KB (two partials 131 KB read, `q_out` 65 KB written, about 3 KB for kv,
the ring-buffer cell and the cos/sin rows), so 256 tokens take about 15 us (50 MB / 4.5e12 B/s + 4 us) and
32768 tokens about 1.5 ms (a 128x span). Smaller batches are correctness-only (1, 3, 17 and 100 are the boundary
cases; 3 and 1000 also leave a partial last sequence).

## Comparison

Default `baseline_relative` for `q_out`, `kv` and `swa_kv`. The reference sums the partials in float64 and keeps
everything in float64; the baseline's own error against it (a CPU emulation of the kernel's float32 arithmetic
differs from the reference by at most one bfloat16 rounding step, 0.0156 on values up to 5.6) is only the output
rounding, so a candidate must also be that accurate.
