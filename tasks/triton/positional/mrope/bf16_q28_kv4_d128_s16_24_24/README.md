# M-RoPE (multimodal rotary embedding), bfloat16, 28 query / 4 KV heads, head size 128, sections 16/24/24

Qwen2-VL / Qwen2.5-VL style rotary embedding for `q[T, 28 * 128]` and `k[T, 4 * 128]` with NeoX
(rotate-half) rotation over the full head. Every token carries three positions (temporal, height, width),
`positions[3, T]`. Rotary index `i` in `[0, 64)` takes its cos/sin from the temporal position for `i < 16`,
the height position for `16 <= i < 40` and the width position for `40 <= i < 64`. The cache is
`[max_position, 128]` float32 with `[cos(64) | sin(64)]` per row (base 1e6). The shape is the Qwen2.5-VL-7B
attention layout; the same operator is exposed by SGLang (`triton_mrope_fused`) and vLLM (`triton_mrope`).

Baseline: SGLang `_triton_mrope_forward_fused` (Apache-2.0), Triton. Upstream rotates q and k in place;
the task interface requires an output, so the vendored kernel reads `q`/`k` and stores to `q_out`/`k_out`
(only the store pointers change; traffic is identical). `run()` fixes the scenario: non-interleaved
sections, NeoX, `rotary_dim == head_dim`. The interleaved and GPT-J branches stay in the kernel as
compile-time-dead code. The old tasks' other head layouts and interleaved/GLM variants are dropped.

Workloads: about 16 KB of traffic per token, so decode batches and short prompts (below ~1500 tokens)
are correctness-only; timed cases are prefill chunks from 2048 to 262144 tokens (128x span).
`positions` are uniformly random in the 32768-row cache so the three-way gather is exercised.

Compare: both outputs use the default `baseline_relative` bound. The reference computes in float64 from the
same float32 cache.
