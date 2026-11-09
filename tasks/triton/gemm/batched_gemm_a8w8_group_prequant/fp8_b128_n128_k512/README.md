# Batched a8w8 GEMM with per-token group activation quantization, 128 batch elements, N = 128, K = 512

`out[m, b] = (fp8(x[m, b]) @ wq[b]^T) * w_scale` for 128 batch elements. The activations `x` [m, 128, 512] arrive in
bfloat16, token-major, and are quantized to fp8 (e4m3fn) **inside the operator** (that work is part of the timed
call) per token, batch element and group of 128 values of K: the group scale is the group maximum (floored at 1e-10)
divided by 448 and the codes are `fp8(clamp(x / scale, -448, 448))`. The weights `wq` [128, 128, 512] are fp8 e4m3fn
with a single float32 per-tensor scale `w_scale` [1] shared by all batch elements. Each group's integer-valued fp8
products are accumulated in float32, multiplied by that group's activation scale and summed; the sum is multiplied by
`w_scale` and written token-major as bfloat16 `out` [m, 128, 128]. `m` is the only varying axis. K = 512 is four
quantization groups, so a candidate that uses one scale per row instead of per group fails.

## Scenario and config citations

The scenario is the DeepSeek multi-head latent attention value up-projection (the absorbed `W_UV` batched GEMM):
each attention head multiplies its latent attention output by its slice of `W_UV`. DeepSeek-V3 `config.json`
(Hugging Face, `deepseek-ai/DeepSeek-V3`): `num_attention_heads` 128 (batch), `kv_lora_rank` 512 (K),
`v_head_dim` 128 (N). 128 heads is the single-rank head count; with tensor parallelism the batch shrinks to
128 / TP.

**Production callers (pinned commits).** SGLang
(`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`, `rocm_absorb_v_bmm`)
and vLLM (`vllm/model_executor/layers/attention/mla_attention.py`, `_v_up_proj` through
`rocm_aiter_ops.triton_fp8_bmm` in `vllm/_aiter_ops.py`) call
`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` on gfx950 with `group_size=128`, the
attention output viewed as `[tokens, heads, kv_lora_rank]` and the fp8 weight `W_UV.permute(1, 2, 0)`
(`[heads, v_head_dim, kv_lora_rank]`), writing `[tokens, heads, v_head_dim]` (`transpose_bm=True`). The task keeps
that memory layout: `x` is `[m, batch, k]` and `out` is `[m, batch, n]`. The weight is quantized per whole batched
tensor, `scale = amax / 448` over all heads (vLLM `dynamic_per_batched_tensor_quant`), which is the single
`w_scale` element. The transposed absorbed W_UK projection (N = 512, K = 128, SGLang `rocm_absorb_q_bmm`) has its own
AITER table and is not part of this scenario.

## Reference and inputs

`reference.py` quantizes the activations with the operator's arithmetic (float32 group scale
`amax * (1 / 448)`, `x * (1 / scale)` in float32, round to nearest even into fp8) and accumulates the group products
of fp8 codes exactly in float64, so a quantization error in an implementation is caught instead of hidden in a
tolerance. It is independent of the baseline.

`inputs.py` draws normal weights (std 0.05) per head and quantizes them with one scale for the whole tensor
(`w_scale` is about 6e-4); activations are standard normal. Outputs have std about 1.1; the fp8 quantization noise
against unquantized math is about 2.6% of the output, which is why the reference reproduces the quantizer.

## Workloads

`m` is the number of tokens of a step. vLLM pre-compiles the fp8 bmm for 1 to 1024 tokens as its decode range;
larger values stand for batched prefill chunks. The performance cases are memory-bound (about 164 KB moved per token
across the 128 heads, 1 KiB of bfloat16 activations and 256 B of output per head, plus an 8.4 MB weight) and start at
256 tokens (estimated 15 us by bytes / 4.5e12 B/s plus 4 us; 100 tokens would be about 10 us and is
correctness-only). 25600 tokens (estimated about 0.94 ms) give the required 100x span. The largest case holds 3.4 GB of
activations and a 3.4 GB float64 reference output. Boundary cases: 1 token (axis minimum), 3 (non power of two), 17, 100,
1000 and 3000 (not multiples of 16, above 16).

## Baseline

AITER's Triton `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel` with the gfx950
tuned table for N = 128, K = 512 (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The
kernel is unchanged. The launcher is the no-bias path with `transpose_bm_in` and `transpose_bm`; `BLOCK_SIZE_K` is
the group size 128 and `kpack` 1 as upstream sets them. Tile heights are 16 to 64 rows for M <= 256 and 32 rows
above. The vendored files carry the upstream SPDX/copyright header; `kernel_repr.py` has none upstream, so the
repository MIT license applies to the inlined `make_kernel_repr`. No third-party origin is recorded in any of the
vendored files or their upstream directories. The AITER wrapper documents the weights as INT8; the kernel takes the
quantization range from the weight dtype, and the production callers (and this task) pass fp8 e4m3fn.

## Comparison and memory

The activation quantizer divides in float32 and the hardware reciprocal may differ from the exactly rounded value in
the last bit; a candidate whose arithmetic differs from the baseline's can then flip a rare fp8 code against the
reference (and, where the baseline did not flip, exceed the baseline-relative bound by up to about one
quantization step of a single product). The default `baseline_relative` bound is kept. `kpack` is passed as a launch
option as upstream does. M = 256 is the floor of the performance cases. The comparison keeps several float64
temporaries of the whole output (3.4 GB each at 25600 tokens).
