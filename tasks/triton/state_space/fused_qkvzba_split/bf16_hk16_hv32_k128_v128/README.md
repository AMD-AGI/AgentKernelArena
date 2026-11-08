# Fused QKVZBA split / reshape / concat (Gated DeltaNet prefill), bfloat16, 16 key / 32 value heads, K = V = 128

The Qwen3-Next Gated DeltaNet layer projects each token with two fused linear layers whose outputs
are interleaved per key-head group: `in_proj_qkvz` gives, for each of the 16 key heads,
`[q (128) | k (128) | v (2 x 128) | z (2 x 128)]` (12288 values per token) and `in_proj_ba` gives
`[b (2) | a (2)]` per key head (64 values). Before the convolution and the delta rule they are split
and regrouped into `mixed_qkv` (`n_tokens x 8192`: all q, then all k, then all v, the layout the
causal convolution consumes), `z` (`n_tokens x 32 x 128`), `b` and `a` (`n_tokens x 32` each). This is
a pure data movement (about 49 KB per token): every output element is one input element, so the
comparison is exact.

The layout is the one vLLM's `fix_query_key_value_ordering`
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` at
`edbc4969a76b25c531f7b9bb16b79984e65ef023`) splits with `torch.split` and SGLang documents as the
checkpoint's "per-head-group interleaved layout" (`python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py`
at `887c401e151e2d1b082d74ef8769a23eef5c465c`).

## Baseline

AITER's prefill kernels of `aiter/ops/triton/_triton_kernels/gated_delta_rule/fused_qkvzba_split.py`
(ROCm/aiter at `652f2d4b768239aefd515c929e84817681ff598d`), vendored unchanged: one program per
token (and per key head in the generic kernel), 1 warp, 3 stages. The wrapper's dispatch is kept: key-head
counts 1, 2, 4 and 8 use specialised kernels with one program per token, any other count (16 here, so the
scenario runs the generic 2D-grid kernel `_fused_qkvzba_split_reshape_cat_prefill_kernel`) uses one
program per `(token, key head)`. Sharding this layer over tensor-parallel 2 or 4 would give 8 or 4 key
heads and select a specialised kernel; this scenario is the unsharded layer, consistent with the other
Qwen3-Next Gated DeltaNet tasks. The decode kernel and its wrapper are not used and removed. `run()`
writes preallocated outputs instead of allocating them; the key-head count and size are module constants.

The kernels compute offsets in int32 (`i_seq * 16 * 768`): `n_tokens` is bounded by 131072 (offsets up
to 1.6e9 of 2.1e9), which is why the axis stops there.

**Origin: SGLang.** The AITER file has no SPDX or copyright header, and the only commit in its
history is "[Triton] Support GDR prefill and decode (#1594)". Its decode kernel
`_fused_qkvzba_split_reshape_cat_decode_kernel` is line for line SGLang's `fused_qkvzba_split_reshape_cat_kernel`
(`python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py` of sgl-project/sglang at
`887c401e151e2d1b082d74ef8769a23eef5c465c`, Apache-2.0; that file has no header either, and the earliest commit of
the path in the pinned checkout is a code migration, "[Kernel] Phase 4 batch-3: migrate tangled JIT subsystems ...
(#32045)"), and the five prefill kernels used here are variants of it (same pointer arithmetic, with the per-head
loop and specialisations added). The direction of the copy is unknown. `baseline/LICENSE` is AITER's MIT text, and
`baseline/LICENSE-sglang` is SGLang's Apache-2.0 LICENSE at that commit (the SGLang file has no header, so the
repository license is carried), so the baseline is MIT (AITER) and Apache-2.0 (SGLang).

## Scenario and config citations

Qwen3-Next-80B-A3B-Instruct, tensor-parallel 1. Source: the public `config.json` of Qwen/Qwen3-Next-80B-A3B-Instruct on Hugging Face (`https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json`): `linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128. The same values are the defaults of `Qwen3NextConfig` (`vllm/transformers_utils/configs/qwen3_next.py` in vLLM at `edbc4969a76b25c531f7b9bb16b79984e65ef023` and `python/sglang/srt/configs/qwen3_next.py` in SGLang), which corroborates them.

- `linear_num_key_heads` 16, `linear_key_head_dim` 128: q and k of a group are 128 wide, 16 groups.
- `linear_num_value_heads` 32, `linear_value_head_dim` 128: each key head owns 2 value heads, so v and z
  are 2 x 128 wide per group.
- Derived: `mixed_qkvz` 16 x (2 x 128 + 2 x 2 x 128) = 12288, `mixed_ba` 2 x 32 = 64, `mixed_qkv`
  2 x 16 x 128 + 32 x 128 = 8192.

## Inputs and reference

Both projections are standard normal bfloat16, so every element is distinct enough that a wrong
group offset, a swapped q / k or z / v block, or swapped b / a halves changes the result.
`reference.py` slices the interleaved rows with views and concatenates; the outputs are returned in
float32 (exact for bfloat16 values) because the CPU pre-check requires a float32 or float64 reference,
and `exact` compares them bitwise after rounding back to bfloat16. A float64 reference would add nothing
for a copy and would need 4x the memory of the output.

## Comparison

All four outputs use `exact`: a data movement has no rounding, and any tolerance would only hide
misplaced elements. No bound is relaxed.

## Workloads

Token counts from 1024 to 131072 (128x span). The kernel reads 24.7 KB and writes 24.7 KB per token, so
the smallest performance case moves about 51 MB, estimated at 15 us on MI355X (51 MB / 4.5e12 B/s + 4 us),
and the largest 6.5 GB (about 1.4 ms). Prefill steps of 2048 to 16384 tokens are typical;
the larger counts stand for long-context prefill. The boundary correctness cases are 1 (minimum), 3
(not a power of two), 100 and 513 (not multiples of 16).
