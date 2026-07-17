# Recommended vLLM Subset Tasks (~4h)

This is a recommended subset of 25 `triton2triton` tasks for a roughly 4-hour run, selected to be representative of open-source LLM inference workloads (especially vLLM / DeepSeek-style paths) while mixing easier kernels and harder kernels across logits/sampling, KV-cache + attention, speculative decoding (EAGLE), and MoE/EP routing.

## Full Subset List

```
- triton2triton/vllm/triton_temperature
- triton2triton/vllm/triton_log_softmax
- triton2triton/vllm/triton_penalties
- triton2triton/vllm/triton_apply_grammar_bitmask
- triton2triton/vllm/triton_logit_bias
- triton2triton/vllm/triton_topk_log_softmax
- triton2triton/vllm/triton_silu_mul_fp8_quant_dg
- triton2triton/vllm/triton_compute_slot_mappings
- triton2triton/vllm/triton_gather_block_tables
- triton2triton/vllm/triton_reshape_and_cache_flash_diffkv
- triton2triton/vllm/triton_decode_attn_stage1
- triton2triton/vllm/triton_decode_attn_stage2
- triton2triton/vllm/triton_flash_prefill_attention
- triton2triton/vllm/triton_unified_attention_3d
- triton2triton/vllm/triton_prepare_eagle_inputs
- triton2triton/vllm/triton_prepare_eagle_docode
- triton2triton/vllm/triton_eagle_prepare_inputs_padded
- triton2triton/vllm/triton_update_eagle_inputs
- triton2triton/vllm/triton_copy_and_expand_eagle_inputs
- triton2triton/vllm/triton_rejection_sample
- triton2triton/vllm/triton_count_expert_tokens
- triton2triton/vllm/triton_ep_scatter_1
- triton2triton/vllm/triton_ep_scatter_2
- triton2triton/vllm/triton_ep_gather
- triton2triton/vllm/triton_fused_moe

```
### ROCmBench Subset Structure

`triton2triton/rocmbench` tasks are and grouped by difficulty:

```
easy:
- test_add_kernel
- test_batched_vecmat
- test_block_copy
- test_kernel_dot
- test_kernel_sub
- test_load_reduce
- test_randn
- test_random_int
- test_reverse_range

medium:
- layernorm
- naive_softmax
- rmsnorm_fwd
- softmax
- test_cast_matmul
- test_chained_matmul
- test_gemm_no_scf
- test_iv_dependent_matmul
- test_triton_sort
- test_triton_swizzle2d

hard:
- gemm
- moe_gemm
- multreduce_matmul_dot_kernel
- rmsnorm_bwd
- test_block_pointer_matmul
- test_chained_dot_fp8
- test_flashattention_fwd
- test_gemm_fusion
- test_matmul_MXFP
- test_tma_store_gemm
- triton_multreduce_matmul_kernel
```
