# Fused block-scaled fp8 GEMM + KV split + rope concat, DeepSeek-V3 kv_b_proj (N = 32768, K = 512, 128 heads)

The MLA key/value up-projection with the split and concatenation fused into the GEMM epilogue. `c = dequant(x) @ dequant(w)^T`
(float8_e4m3fn `x` [m, 512] with one float32 scale per 1x128 block, `w` [32768, 512] with one float32 scale per 128x128 block,
float32 accumulation) is viewed as [m, 128, 256]. The outputs are

- `k [m, 128, 192] = cat(c[..., :128], k_pe)` where the rope key `k_pe` [m, 64] is shared by all 128 heads, and
- `v [m, 128, 128] = c[..., 128:]`,

both rounded to bfloat16 (the wrapper's `dtype` argument; bfloat16 is its default and what AITER's tests use). This is AITER's
`fused_gemm_a8w8_blockscale_split_cat`. `k_pe` is the kernel's `y` argument, passed upstream as a [m, 128, 64] view that is expanded
over the heads without a copy; the interface here takes `k_pe` [m, 64] and builds that view inside `run`.

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`kv_lora_rank` 512 (K), `num_attention_heads` 128 (D), `qk_nope_head_dim` 128 (S1), `v_head_dim` 128 (S2) and `qk_rope_head_dim` 64 (S3), so
N = D x (S1 + S2) = 32768 and the weight is the `kv_b_proj` of the model; `quantization_config.weight_block_size` [128, 128] gives the scale
layout. 128 heads is the full-model (tensor parallelism 1 or data-parallel attention) count; AITER ships a tuned file for exactly
this (N, K) = (32768, 512) (`configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=32768-K=512.json`). The consumer in the pinned SGLang tree is
the MHA prefill path of `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py`, which up-projects the
compressed KV (`kv_a`) of the prefix and the current tokens through `kv_b_proj` and concatenates `k_pe` to the keys; SGLang calls the
MXFP4 sibling `fused_gemm_afp4wfp4_split_cat` there, and no pinned vLLM or SGLang call site of the block-scaled fp8 variant exists
(AITER's test `op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py` exercises it with the same layout).

**Reference.** `reference.py` dequantizes `x` and `w` from codes and scales, multiplies in float64 row-chunked (the unchunked
[m, 32768] product would need 8.6 GB at m = 32768), and writes `k` and `v` in float64 including the broadcast `k_pe`.

**Inputs.** `x` is a standard normal [m, 512] activation (a layer-normalized compressed KV is of this scale) quantized in 1x128
blocks (scale = block amax / 448); `w` is a normal weight with std 0.05 (outputs have std about 1.1) quantized in 128x128 blocks with the
same rule; `k_pe` is standard normal bfloat16.

**Workloads.** M is the number of tokens whose compressed KV is up-projected. SGLang bounds a one-shot or chunked prefix KV by
`SGLANG_MAX_KV_CHUNK_CAPACITY` (default 128 K tokens); the largest case here is 32768 because the float64 `k` of one case is
`m x 24576` doubles (6.4 GB at 32768) and the comparison makes several copies. Counts from 1 to 1000 are correctness-only; performance cases run from
128 to 32768 tokens (256x). The kernel reads a 16.8 MB weight and writes 82 KB per token. Estimated baseline time (bytes / 4.5 TB/s + fp8
flops at 2 PFLOP/s + 4 us): M = 128 about 12 us (27 MB), M = 32768 about 0.7 ms (2.7 GB).

**Baseline.** AITER's Triton `_fused_gemm_a8w8_blockscale_split_cat` kernel with its reduce kernel, the launcher logic and the N = 32768,
K = 512 gfx950 table inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The kernels are unchanged. Every table
row has NUM_KSPLIT 1, so the reduce kernel is never launched here and there is no split-K nondeterminism. The upstream kernel and wrapper
files carry the SPDX header `MIT` and `Copyright (C) 2024-2025, Advanced Micro Devices, Inc.` (kept verbatim); the helper `pid_grid` file
carries 2024-2026. No third-party origin is noted in any vendored file or in its history.

**Table change for M <= 8.** With the upstream `M_LEQ_8` row (BLOCK_SIZE_M 8, BLOCK_SIZE_N 16, one warp) a byte copy of the baseline
used as a candidate was not reproducible at M = 1 on MI355X: it passed on one seed (74 dB) and failed on another (about 400 wrong elements
of k and v, 482x the bound). The same code gave different results, so the result is not a function of the inputs. Static analysis of the
kernel and launcher for this shape rules out the usual causes: every row has `NUM_KSPLIT` 1, so there is no split-K workspace, no reduce
kernel, no atomics and no uninitialized partial buffer; every element of `k` (GEMM columns, and the `k_pe` columns written by
`BLOCK_SIZE_S3` = 4 slots per program) and of `v` is written by exactly one program, and the 2048 programs cover N = 32768 and
D x S3 = 8192 exactly. About 400 wrong elements is about 25 programs of 16 columns out of 2048, i.e. a few whole tiles wrong at random,
which points at the compiled 8-row, 16-column, one-warp tile (the only tile in the table below the 16-row MFMA height with 1-warp
scheduling) rather than at the data. The failure could not be reproduced on a CPU, so the cause is inferred. The baseline therefore
drops that row, so M <= 8 selects the next row of the same upstream table (`M_LEQ_16`: 16x32 tiles, one warp), the smallest tile with a
full MFMA height; all other fields and rows are unchanged. Decode batches of 1 to 8 tokens remain correctness-only.

**Launcher constraint.** `GROUP_K` must equal `BLOCK_SIZE_K` (128 in every row), which the launcher asserts. M = 128 is about 1.2x above the 10 us floor.
