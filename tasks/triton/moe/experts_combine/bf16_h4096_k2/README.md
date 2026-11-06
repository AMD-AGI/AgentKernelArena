# Experts combine, bfloat16, hidden size 4096, top-2

Combines the routed-expert branch of a MoE layer with its shared-expert branch for each token:
`out = (sum_k moe_hidden_states[:, k] + mlp_hidden_states) / sqrt(2)`. The routed input carries the
`combine_k = 2` expert outputs of every token still to be summed (shape `[num_tokens, 2, 4096]`); the factor
`1/sqrt(2)` balances the variance of the two added branches. This is SGLang's `experts_combine_triton` used by models
with a dense MLP running next to the routed experts (for example Grok-style top-2 layers).

Scenario: hidden size 4096, top-2, bfloat16 (the 4096 / top-2 variant of the original task's case list). The operator is
memory bound (`4 * hidden * 2` bytes per token), so batches below about 1500 tokens run faster than the 10 us timing
floor on MI355X and are checked for correctness only; the performance cases start at 2048 tokens. The largest case is
limited to 204800 tokens because the baseline (like production) addresses the routed tensor with 32-bit offsets.

Baseline: SGLang's Triton kernel, one program per token row with the whole row in one block. The kernel text is the
unchanged function from `python/sglang/srt/layers/elementwise.py`. The pinned SGLang checkout used for other tasks no
longer contains the file, so the provenance records the AgentKernelArena revision whose task carried the extracted
kernel; the original upstream revision was not recorded by that task.

Bounds: the default `baseline_relative` bound. The reference (float64 sum and scale, chunked over tokens to bound
memory) is independent of the baseline.
