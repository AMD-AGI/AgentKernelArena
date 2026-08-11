# Head Kernels

20 kernels extracted from production vLLM / SGLang inference paths, each with the
captured launch signatures and harness from the originating engine run.

Run the whole suite with the `head_kernels` selector:

```yaml
tasks:
  - head_kernels
```

## Tasks

`triton2triton` (13):

```
_fwd_grouped_kernel_stage1
_gemm_a16_w16_kernel
_gemm_a8w8_blockscale_kernel
_per_token_group_quant_fp8
_topk_forward
_w8a8_triton_block_scaled_mm
chunk_scaled_dot_kkt_fwd_kernel
fused_moe_int4_w4a16
fused_moe_kernel
fused_moe_kernel_gptq_awq
gemm_a8w8_blockscale
kernel_unified_attention_2d
write_req_to_token_pool_triton
```

`hip2hip` (7):

```
moe_gemm_fp8_blockscale
moe_stage1
moe_stage2
paged_attention_decode
paged_attention_large
paged_attention_ragged
wvSplitK
```

## Setup: paged_attention_large

`paged_attention_large/test_cases.json` ships with the 8 captured cases only. The 8
memory-bound `perf_only` cases are omitted because their baked index tensors make the
file 234 MB, over GitHub's 100 MB file limit. Regenerate them before benchmarking:

```bash
cd tasks/head_kernels/paged_attention_large
python3 scripts/gen_perf_cases.py
```

This needs no GPU and reproduces the omitted cases byte-for-byte. Correctness runs skip
`perf_only` cases either way, so only the performance sweep is affected.
