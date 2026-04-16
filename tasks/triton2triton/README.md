# Standard Triton-to-Triton Task Set (18 kernels)

This is the standard set of 18 `triton2triton` kernel optimization tasks, covering MoE routing, attention (MLA decode, lean paged attention), feed-forward layers, quantized GEMM, normalization, and fused ops representative of production LLM inference workloads.

## Standard Kernel List

```
triton2triton/geak_eval/L1/fused_append_shared_experts
triton2triton/geak_eval/L1/llama_ff_triton
triton2triton/geak_eval/L1/mla_decode
triton2triton/geak_eval/L1/moe_routing_sigmoid_top1
triton2triton/geak_eval/L1/refk_fp8_blockwise_mm
triton2triton/geak_eval/L1/refk_identity
triton2triton/geak_eval/L2/fast_rms_layernorm
triton2triton/geak_eval/L2/ff_backward
triton2triton/geak_eval/L2/lean_atten_paged
triton2triton/geak_eval/L2/topk
triton2triton/geak_eval/L3/fused_moe_mxfp4
triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort
triton2triton/geak_eval/L3/fused_qk_rope_cache_mla
triton2triton/geak_eval/L3/fused_qkv_rope
triton2triton/geak_eval/L3/fused_rms_fp8
triton2triton/geak_eval/L3/gemm
triton2triton/geak_eval/L3/gemm_a16w16_atomic
triton2triton/geak_eval/L3/gemm_a16wfp4
```
