# Batched a16wfp4 GEMM, 128 batch elements, N = 128, K = 512

`out[m, b] = mxfp4(x[m, b]) @ dequant(w[b])^T` for 128 batch elements. The activations `x` [m, 128, 512] arrive in
bfloat16, token-major, and are quantized to MXFP4 **inside the operator** (that work is part of the timed call):
one e8m0 scale per 32 values of K, taken from the group maximum rounded to a power of two minus the exponent of the
largest e2m1 value, then round to nearest even onto the e2m1 grid, saturating at 6. The weights are MXFP4: `w`
[128, 128, 256] holds two e2m1 codes per byte (low nibble is the even K index) and `w_scales` [128, 128, 16] holds
uint8 e8m0 scales (`2^(s - 127)`) per 32 values of K. The block products are accumulated in float32 and the result
is written token-major as bfloat16 `out` [m, 128, 128]. `m` is the only varying axis.

## Scenario and config citations

The scenario is the DeepSeek multi-head latent attention value up-projection (the absorbed `W_UV` batched GEMM):
each attention head multiplies its latent attention output by its slice of `W_UV`. DeepSeek-V3 `config.json`
(Hugging Face, `deepseek-ai/DeepSeek-V3`): `num_attention_heads` 128 (batch), `kv_lora_rank` 512 (K),
`v_head_dim` 128 (N). 128 heads is the single-rank head count; with tensor parallelism the batch shrinks to
128 / TP.

**Layout.** vLLM (`vllm/model_executor/layers/attention/mla_attention.py`, `_v_up_proj`, at the pinned commit)
views the attention output as `[tokens, heads, kv_lora_rank]`, transposes it to `[heads, tokens, kv_lora_rank]`
without a copy, and calls `batched_gemm_a16wfp4(x, W_V, W_V_scale, out, transpose_bm=True, prequant=True)`, where `W_V`
is the MXFP4 quantization of `W_UV.permute(1, 2, 0)` (`[heads, v_head_dim, kv_lora_rank]`) and `out` is
`[tokens, heads, v_head_dim]`. The task keeps exactly that memory layout: `x` is `[m, batch, k]` and `out` is
`[m, batch, n]`, and the baseline's `run()` takes the same `transpose(0, 1)` view. A candidate may read
the strided activations however it likes. The reference computes the same math on those logical tensors.

## Reference and inputs

`reference.py` reproduces the activation quantization (power-of-two rounding of the group maximum, value-based
round-to-nearest-even onto the e2m1 grid) and multiplies dequantized operands in float64 per batch element, so a
quantization difference between an implementation and the operator is caught as an error rather than hidden in a
tolerance. It reproduces the dequantized activations of the sibling `gemm_a16wfp4` reference exactly (checked on
CPU) and is row-chunked to bound memory.

`inputs.py` quantizes normal weights (std 0.05) to MXFP4 with e8m0 scales per batch element (the scale exponent
is `round(log2(amax / 6))`, codes round to nearest with ties to the even code); activations are standard normal.
The outputs have std about 1.1.

## Workloads

`m` is the number of tokens of a step. vLLM pre-compiles the sibling fp8 bmm for 1 to 1024 tokens as the decode
range; larger values stand for batched prefill chunks. The performance cases are memory-bound (about 164 KB moved
per token across the 128 heads, 1 KiB of bfloat16 activations and 256 B of output per head) and start at 256 tokens
(estimated 14 us by bytes / 4.5e12 B/s plus 4 us; 100 tokens would be about 8 us and is correctness-only). 25600
tokens (estimated about 0.9 ms) give the required 100x span. The largest case holds 3.4 GB of activations and a
3.4 GB float64 reference output. Boundary cases: 1 token (axis minimum), 3 (non power of two), 17, 100, 1000 and
3000 (not multiples of 16, above 16).

## Baseline

AITER's Triton `_batched_gemm_a16wfp4_kernel` with its in-kernel `_mxfp4_quant_op` and the gfx950 tuned table for
N = 128, K = 512 (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). Kernel and quantizer
are unchanged. Every row of the table has `NUM_KSPLIT` 1, so the split-K reduce kernel, `get_splitk` and the
partial-sum buffer are dropped; the launcher is the `prequant=True, transpose_bm=True` path. The tuned rows use
64x128x256 tiles for M > 32 (16x64 for M <= 16, 32x128 for M <= 32). Upstream calls `remap_xcd(pid, GRID_MN)` and
discards the result, so no XCD remapping takes place; this is kept as is. The vendored files carry the upstream
SPDX/copyright header; `kernel_repr.py` has none upstream, so the repository MIT license applies to the inlined
`make_kernel_repr`. No third-party origin is recorded in any of the vendored files or their upstream directories.

## Comparison and memory

`tl.dot_scaled` needs gfx950. M = 256 is the floor of the performance cases. The comparison keeps several float64
temporaries of the whole output (3.4 GB each at 25600 tokens). The in-kernel quantizer uses float32 bit tricks; the
reference was written from its documented semantics and cross-checked against the sibling task's reference. The default
`baseline_relative` bound is used.
