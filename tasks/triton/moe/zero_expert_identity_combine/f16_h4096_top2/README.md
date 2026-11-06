# Identity (zero) expert combine, float16, hidden size 4096, 2 expert scales

Same operator as `bf16_h4096_top8` with the configuration of the original task: float16 activations and
2 expert scales per token. `out[i] = sum_k hidden_states[i] * expert_scales[i, k]`, accumulated in float32
and rounded to float16. Scales are float32 router weights in [0, 0.5).

The baseline is vLLM's `compute_identity_kernel` (`vllm/model_executor/layers/fused_moe/fused_moe.py`),
one program per token and 256-wide column block, wrapped in `run()`. The host-side mask that zeroes the
scales of real experts and the output allocation are not part of the timed operator.

The work is memory-bound (about 16 KB moved per token); decode batches and small prefill chunks run
below the 10 us timing floor and are correctness-only. The smallest timed case (2560 tokens, about 42 MB
moved) is estimated at about 13 us and may need raising after the GPU audit. The largest case's float64
reference needs about 20 to 25 GB of memory.
