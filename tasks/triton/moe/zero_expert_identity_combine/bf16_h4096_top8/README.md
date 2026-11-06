# Identity (zero) expert combine, bfloat16, hidden size 4096, 8 expert scales

For MoE layers with identity ("zero") experts, a token routed to such an expert returns its own hidden
state scaled by the router weight. This operator computes
`out[i] = sum_k hidden_states[i] * expert_scales[i, k]`, accumulating in float32 and rounding to
bfloat16. Scales are float32 router weights in [0, 0.5) (non-negative, so the sum does not cancel).

The baseline is vLLM's `compute_identity_kernel` (`vllm/model_executor/layers/fused_moe/fused_moe.py`),
one program per token and 256-wide column block, wrapped in `run()`. The host-side mask that zeroes the
scales of real experts and the output allocation are not part of the timed operator. The original task
used float16 activations, 2-4 scales and hidden sizes up to 4096; this scenario fixes bfloat16, hidden
4096 and top-8, the production configuration of models with identity experts.

The work is memory-bound (about 16 KB moved per token); decode batches and small prefill chunks run
below the 10 us timing floor and are correctness-only. If the GPU audit finds the smallest timed case
below 10 us, raise the smallest performance case.
