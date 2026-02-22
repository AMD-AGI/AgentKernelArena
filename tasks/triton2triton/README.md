# Recommended Subset Tasks (~4h)

This is a recommended subset of 25 `triton2triton` tasks for a roughly 4-hour run, selected to be representative of open-source LLM inference workloads (especially vLLM / DeepSeek-style paths) while mixing easier kernels and harder kernels across logits/sampling, KV-cache + attention, speculative decoding (EAGLE), and MoE/EP routing.

## Full Subset List

```
- triton2triton/triton_temperature
- triton2triton/triton_log_softmax
- triton2triton/triton_penalties
- triton2triton/triton_apply_grammar_bitmask
- triton2triton/triton_logit_bias
- triton2triton/triton_topk_log_softmax
- triton2triton/triton_silu_mul_fp8_quant_dg
- triton2triton/triton_compute_slot_mappings
- triton2triton/triton_gather_block_tables
- triton2triton/triton_reshape_and_cache_flash_diffkv
- triton2triton/triton_decode_attn_stage1
- triton2triton/triton_decode_attn_stage2
- triton2triton/triton_flash_prefill_attention
- triton2triton/triton_unified_attention_3d
- triton2triton/triton_prepare_eagle_inputs
- triton2triton/triton_prepare_eagle_docode
- triton2triton/triton_eagle_prepare_inputs_padded
- triton2triton/triton_update_eagle_inputs
- triton2triton/triton_copy_and_expand_eagle_inputs
- triton2triton/triton_rejection_sample
- triton2triton/triton_count_expert_tokens
- triton2triton/triton_ep_scatter_1
- triton2triton/triton_ep_scatter_2
- triton2triton/triton_ep_gather
- triton2triton/triton_fused_moe

```