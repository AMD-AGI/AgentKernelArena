# LoRA expand (punica SGMV), bfloat16, rank 16, output width 4096, four adapters

Expands the rank-16 intermediate of each token with the B matrix of the adapter it is routed to:
`out[t] = x[t] @ lora_b[a].T`, accumulated in float32 and rounded to bfloat16. The kernel writes 8 KB per token, so it
is bound by output bandwidth.

**Production origin.** vLLM's `lora_expand` (`vllm/lora/ops/triton_ops/lora_expand_op.py`) following `lora_shrink`.
Rank 16 and width 4096 are Llama-3-8B-class values.

**Scenario choice.** vLLM normally calls expand with `add_inputs=True`, accumulating into the base layer's output.
The task contract requires at least one declared output, so this scenario uses the overwrite mode (`add_inputs=False`,
the path the previous task package scored): `out` is written completely. Every token belongs to an active adapter,
so no output row is left unwritten. The accumulate mode is not covered.

**Routing metadata.** `inputs.py` produces vLLM's `LoRAKernelMeta` tensors: token indices sorted by adapter group,
tokens per group, group start offsets and the adapter slot of each group (a random permutation of the four slots).
Group sizes are skewed.

**Baseline.** The vLLM Triton kernels, launched with vLLM's default expand configuration for one slice (block 64 x 128, 4 warps, 2 stages), except that BLOCK_K is the rank (16) rather than the default 32, so the K loop needs no masking (the previous task package used the same value).
Only the single-slice path is kept; the multi-slice pointer-table preparation, which allocates tensors on the host, is
dropped. The kernel code is the version of the previous task package, which matches upstream apart from comments.

**Workloads.** Prefill token counts. Batches under about 4096 tokens (including decode) write less than 34 MB and run
below the 10 us timing floor, so they are correctness-only; performance cases span 4096 to 409600 tokens (100x). The
largest case writes 3.4 GB. The timing floor is tight at the smallest performance case.
