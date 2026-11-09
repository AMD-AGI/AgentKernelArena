# Fused triangular solve + recompute of w, u (Gated DeltaNet prefill), bfloat16, 16 key / 32 value heads, K = V = 128, chunk 64

The second step of AITER's Gated DeltaNet chunked prefill. Tokens are split into chunks of 64 (the last
may be partial). For every value head `h` (key head `h // 2`) and chunk, `A_raw[:, h]` is the 64 x 64
block `L = tril(beta * K K^T * decay, -1)` that the cumsum / K K^T kernel (see the `chunk_cumsum_kkt`
task) writes (zeros on and above the diagonal and in padding). The operator inverts `I + L` and applies
the inverse to the scaled keys and values:

```text
Ai = (I + L)^-1                              # unit lower triangular, per chunk and head
w  = Ai @ (beta * 2^g_cumsum * k)            # [H, T, 128]
u  = Ai @ (beta * v)                         # [H, T, 128]
```

`k` is `[T, 16, 128]` and `v` `[T, 32, 128]` bfloat16, `beta` `[T, 32]` and `A_raw` `[T, 32, 64]` float32,
`g_cumsum` `[32, T]` float32 (chunk-local cumulative log-decay in log2 space), `w` and `u` bfloat16,
head-major. One sequence batch (no variable-length metadata), as in the sibling Gated DeltaNet tasks:
with `cu_seqlens` AITER builds the chunk schedule on the host, which a graph-captured call cannot do.

## Baseline

AITER's `fused_solve_tril_recompute_w_u` of
`aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_solve_tril_recompute.py` and the
`merge_16x16_to_64x64_inverse_kernel` of `.../utils/solve_tril.py` (ROCm/aiter at
`652f2d4b768239aefd515c929e84817681ff598d`), kernels unchanged. The production dispatch is kept:

- up to 32 chunks (2048 tokens), the single fused kernel inverts the four diagonal 16 x 16 blocks by
  forward substitution, builds the six off-diagonal blocks with `ieee` dots, rounds `Ai` to bfloat16 in
  registers and multiplies it with the bfloat16-rounded `beta * v` and `beta * 2^g * k`;
- above 32 chunks, the split path stores `Ai` in bfloat16 with `merge_16x16_to_64x64_inverse_kernel` and
  a second kernel multiplies it the same way (`recompute_w_u_head_major_kernel`).

Both paths therefore round `Ai`, `beta * v` and `beta * 2^g * k` to bfloat16 before the products, which
accumulate in float32; the baseline's error against an exact evaluation is bfloat16-level (SQNR about
52.6 dB on `w` and `u`). The upstream `@triton.autotune` decorators have one entry each unless the
environment variable `GATED_DELTA_RULE_TRITON_AUTOTUNE` is set (fused and recompute kernels: block 64 x 64,
2 warps, 2 stages; solve kernel: 4 warps, 3 stages); those entries are passed at launch, which also keeps
a first call from tuning inside a graph capture. The threshold, the dot precision and `exp` are fixed to
their upstream defaults instead of read from environment variables. Unused kernels are removed (see
`task.yaml`).

**Provenance note.** Both vendored files carry the upstream header `SPDX-License-Identifier: MIT`,
`Copyright (C) 2024-2026, Advanced Micro Devices, Inc.` and `Adapted from flash-linear-attention:
Copyright (c) 2023-2025, Songlin Yang, Yu Zhang`, kept verbatim. flash-linear-attention is MIT licensed,
the same license; `baseline/LICENSE` holds the AITER MIT text followed by the flash-linear-attention MIT notice. `exp` and the non-TMA `make_tensor_descriptor` stub are the code
of `utils/op.py` (same header). The same family of kernels is vendored in the sibling `solve_tril` and
`recompute_w_u` tasks from vLLM.

## Scenario and config citations

Qwen3-Next-80B-A3B-Instruct. Source: the public `config.json` of Qwen/Qwen3-Next-80B-A3B-Instruct on Hugging Face (`https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json`): `linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128. The same values are the defaults of `Qwen3NextConfig` (`vllm/transformers_utils/configs/qwen3_next.py` in vLLM at `edbc4969a76b25c531f7b9bb16b79984e65ef023` and `python/sglang/srt/configs/qwen3_next.py` in SGLang), which corroborates them. Chunk size 64 as in AITER's `chunk_gated_delta_rule_fwd_opt_vk` (`prefill/chunk.py`).

## Inputs and reference

`k ~ N(0, 1/128)` (L2-normalized keys), `v ~ N(0, 1)`, `beta ~ U(0, 1)` float32. `inputs.py` produces
`g_cumsum` and `A_raw` as the upstream K K^T kernel would from these `k` and `beta` and a raw per-token
log-decay `-exp(A_log) * softplus(a + dt_bias)` (non-positive) with `a + dt_bias ~ N(0, 1)` and a
per-head rate `exp(A_log)` log-uniform in `[0.02, 0.4]` (a design choice of this task; trained values
were not available): `g_cumsum` reaches about -35, and `A_raw` follows the producer's layout: strictly lower
triangular, zeros on and above the diagonal inside the 16 x 16 diagonal blocks, because the kernels read
those entries unmasked in their forward substitution (`A is strictly lower triangular` in `solve_tril`'s
docstring). The six 16 x 16 blocks strictly above the diagonal blocks are never read by the baseline
(only the four diagonal blocks and the six blocks below them are loaded) and are not part of the
operator, so the generator fills them with random values in [-0.3, 0.3) that must be ignored and the reference
takes `tril(-1)` itself; this also keeps the input from being an all-zero or bfloat16-exact region (an
input-quality check flags constant or mostly bfloat16-exact float32 inputs). Its entries have an RMS of 0.017 and
reach 0.28, so `I + L` is well conditioned.

`reference.py` is float64: it takes the strict lower triangle of every chunk block, inverts `I + L` with
`torch.linalg.solve_triangular`, and applies it to `beta * 2^g_cumsum * k` and `beta * v`. It agrees to 1e-16 with an explicit `torch.linalg.inv` of single chunk blocks (a full chunk and a
22-token partial chunk).

## Comparison

`w` and `u` use `baseline_relative` with `k: 4` and `max_violation_fraction: 5.0e-4` (the default is
`k: 2` and no violations). The baseline rounds three operands to bfloat16, so its error is
bfloat16-level and set by those roundings. Designs that are equally sensible but round differently
land close to, and sometimes beyond, twice the baseline's error. CPU float32 emulations of the baseline
(`Ai` and both operands rounded to bfloat16, float32 accumulation, bfloat16 output) against designs that
differ in where they round, with the float64 reference (1000 tokens, two draws, default bound):

- everything in float32 then rounded to bfloat16: passes (peak 0.3x the bound);
- `Ai` bfloat16, operands float32: passes (peak 0.6x to 0.8x);
- `Ai` float32, operands bfloat16: fails on 2.3e-5 to 5.7e-5 of the elements, peak 1.3x;
- `beta * k` rounded to bfloat16 first, then scaled by `2^g` and rounded again: fails on 4.7e-3 of the
  `w` elements, peak 2.2x to 2.4x (`u` passes).

At `k: 4`, over 65, 100, 1000, 2100 and 4096 tokens (four draws for 65, 100 and 2100 tokens, two for 1000, one for 4096), the worst violation fractions are 1.5e-4 for the double-rounded design (peaks 1.2x to 1.6x, far
inside the comparator's 16x admission limit) and at most 1.0e-6 for the float32-`Ai` design (peaks up
to 1.2x); the other two pass with no violation. `max_violation_fraction: 5.0e-4` is 3x above the worst
correct design. The evaluator's 16-ulp shift control (a 6.25 % scale error for bfloat16) violates on 0.86
of the elements at `k: 4` with peaks of 10x to 17x, and rounding the outputs to fp8 violates on 0.64
to 0.93 of the elements with peaks of about 200x already at `k: 2`, so both controls stay rejected by a
wide margin. The emulation does not model the hardware's accumulation order inside the bfloat16 dots.

## Workloads

Token counts from 2048 to 204800 (100x span): 2048 tokens is the largest batch of the fused kernel,
larger ones take the split path, so both code paths are timed. Per token the fused kernel reads 20.7 KB
(`A_raw` 8 KB, `k` 4 KB, `v` 8 KB, `beta` and `g_cumsum` 0.25 KB) and writes 16.4 KB (`w`, `u`); the split
path additionally writes and reads `Ai` (4 KB each way). The smallest case moves about 76 MB, estimated at
about 21 us on MI355X (76 MB / 4.5e12 B/s + 4 us; the triangular solve may make it slower), the largest
about 9.3 GB (about 2 ms). Production prefill chunks are typically 2048 to 16384 tokens; larger counts stand
for batched or long-context prefill. Correctness-only cases are 1 (axis minimum), 65 (a full chunk plus
one token), 100 and 1000 (fused path, not multiples of 16), and 2100 (33 chunks: the first split-path
batch, partial last chunk). The largest case needs about 4.3 GB of inputs and 6.7 GB of
float64 outputs.
