# Fused MXFP4 GEMM + KV split + rope concat, DeepSeek-R1 MXFP4 kv_b_proj (N = 32768, K = 512, 128 heads)

The MLA key/value up-projection of an MXFP4-quantized model with the split, the rope concatenation and the fp8 cast fused into the GEMM
epilogue. `c = dequant(x) @ dequant(w)^T` (MXFP4 `x` [m, 256] bytes and `w` [32768, 256] bytes, e2m1 codes two per byte with the even K
index in the low nibble, e8m0 scales per 32 values of K, float32 accumulation) is viewed as [m, 128, 256]. The outputs are

- `k [m, 128, 192] = cat(c[..., :128], k_pe)` where the rope key `k_pe` [m, 64] is shared by all 128 heads, and
- `v [m, 128, 128] = c[..., 128:]`,

both rounded to float8_e4m3fn by the kernel (the wrapper's `dtype` argument). This is AITER's `fused_gemm_afp4wfp4_split_cat`. `k_pe` is
the kernel's `y` argument, passed upstream as a [m, 128, 64] view expanded over the heads without a copy; the interface here takes `k_pe`
[m, 64] and builds that view inside `run`.

**Production use.** SGLang's MXFP4 path calls exactly this operator: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py`
(pinned commit) says "MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat into a single kernel (fused_gemm_afp4wfp4_split_cat)
that writes k and v directly in FP8", calling `self.kv_b_proj((kv_a, k_pe.expand(-1, self.num_local_heads, -1), self.qk_nope_head_dim,
self.v_head_dim, fp8_dtype))`; the wrapper is `fused_gemm_afp4wfp4_split_cat` in `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`.
The fp8 output dtype follows that call (`float8_e4m3fn`).

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`kv_lora_rank` 512 (K), `num_attention_heads` 128 (D = `num_local_heads` with tensor parallelism 1 or data-parallel attention), `qk_nope_head_dim` 128
(S1), `v_head_dim` 128 (S2) and `qk_rope_head_dim` 64 (S3), so N = D x (S1 + S2) = 32768, the `kv_b_proj` weight; AITER ships a tuned file for exactly this
(N, K) = (32768, 512) (`configs/gfx950/triton/gemm/GEMM-AFP4WFP4-N=32768-K=512.json`, K is named by the unpacked count). The MXFP4 weight format is that of an
MXFP4-quantized DeepSeek-R1 checkpoint (AMD Quark) served through AITER.

**Reference.** `reference.py` dequantizes both operands from codes and scales (the e2m1 grid 0, 0.5, 1, 1.5, 2, 3, 4, 6 and the sign bit; scale 2^(s - 127)),
multiplies in float64 row-chunked, and writes `k` and `v` in float64 including the broadcast `k_pe`; the evaluator rounds them to float8_e4m3fn for the
comparison. It is independent of the baseline and of the quantizer in `inputs.py`.

**Inputs.** `inputs.py` quantizes a standard normal activation and a normal weight (std 0.05) with the rule of AITER's dynamic MXFP4 quantization (scale of a
1x32 group = group maximum rounded to a power of two minus the exponent of the largest e2m1 value, codes round to nearest with ties to the even code, saturating
at 6). The outputs have std about 1.1, far from the float8_e4m3fn saturation at 448. `k_pe` is standard normal bfloat16.

**Comparison.** The default `baseline_relative` bound with float8_e4m3fn outputs: the unit roundoff of the output dtype is 2^-4, so one fp8 rounding step
is always allowed in addition to twice the baseline's own error. The `k_pe` columns of `k` pass through one bfloat16-to-fp8 rounding.

**Workloads.** M is the number of tokens whose compressed KV is up-projected. SGLang bounds a one-shot or chunked prefix KV by `SGLANG_MAX_KV_CHUNK_CAPACITY`
(default 128 K tokens); the largest case here is 32768 because the float64 `k` of one case is `m x 24576` doubles (6.4 GB at 32768) and the comparison makes
several copies. Counts from 1 to 1000 are correctness-only; performance cases run from 256 to 32768 tokens (128x). The kernel reads an 8.4 MB weight and writes
41 KB per token. Estimated baseline time (bytes / 4.5 TB/s + fp4 flops at 4 PFLOP/s + 4 us): M = 256 about 10.4 us (20 MB, 8.6 GFLOP), M = 32768 about 0.37 ms (1.4 GB).

**Baseline.** AITER's Triton `_fused_gemm_afp4wfp4_split_cat` kernel (`tl.dot_scaled` e2m1 x e2m1) with its reduce kernel, the launcher logic, `get_splitk` and the
N = 32768, K = 512 gfx950 table (the basic afp4wfp4 family, which the wrapper's `_get_config` imports) inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is
the AITER MIT license). The kernels are unchanged. The table row for M <= 8 asks for split-K 4, but `get_splitk` (K = 256 packed bytes against BLOCK_SIZE_K = 512) collapses it to a single
partition, so no case splits K and the reduce kernel is never launched here; there is no split-K nondeterminism.
The upstream files carry the SPDX header `MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, kept verbatim; no third-party origin is noted in any
vendored file or in its history.

**Kernel notes.** `tl.dot_scaled` needs gfx950. M = 256 is estimated at 10.4 us by the bandwidth formula, only barely above the 10 us floor;
the real kernel at that M uses a 256x128 tile and is slower than that estimate. fp8 rounding of the outputs happens in the kernel (`.to(float8_e4m3fn)`).
