# Gated bf16 GEMM (SwiGLU gate/up projection), N=28672, K=4096

`out = silu(x @ w[:14336]^T) * (x @ w[14336:]^T)` with `x[m, 4096]`, one stacked weight `w[28672, 4096]` (gate rows
first, up rows second, `nn.Linear` layout) and `out[m, 14336]`, all bfloat16. Both products are accumulated in float32;
the SiLU and the product are applied to the float32 accumulators and the result is rounded to bfloat16 once. The var
axis is `m`. This is the fused gate/up projection of a SwiGLU MLP, the first half of the feed-forward block.

## Production origin of the baseline

AITER's `gemm_a16w16_gated` (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT): the Triton kernel
`_gemm_a16_w16_gated_kernel` of `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py`, launched by the
wrapper `aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py` with `activation="silu"`. Each program computes one
`BLOCK_SIZE_M x BLOCK_SIZE_N/2` tile of the gate half and the matching tile of the up half, so the gating costs no
extra pass over memory. The tile configuration is AITER's `GEMM-A16W16-gated` table for gfx950
(`configs/gemm/gfx950-GEMM-A16W16-gated.json`; no N=28672, K=4096 file exists), selected with the upstream bounds
`(64, 128, 256, 512, 2048)`. All of it is kept in `baseline/kernel.py` (table and lookup) and the files beside it.

- The kernel is byte-identical to upstream except for its three import lines and the module docstring placed after the header. `baseline/pid_preprocessing.py` and
  `baseline/kernel_repr.py` are verbatim copies of `aiter/ops/triton/utils/_triton/`. `baseline/activation.py` holds
  `_silu_exp2` and `_silu` copied from `aiter/ops/triton/_triton_kernels/activation.py`.
- `kernel_repr.py`, `activation.py` and `gemm_config_utils.py` carry no SPDX or copyright header upstream; the
  repository-wide MIT license of ROCm/aiter applies to them (`baseline/LICENSE`). The other two vendored files keep
  the header `SPDX-License-Identifier: MIT` / `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`.
- No third-party origin is recorded for these files: no "adapted from"/"based on" text in them or beside them, and
  their first commits are AMD's own AITER Triton work (`60abc99c2`, `f9ee4168b` for the kernel).
- The upstream wrapper accepts other activations, a dtype and an optional output; `run()` fixes SiLU and writes into
  `out`. The SiLU of the kernel is `x / (1 + exp2(-1.4427 x))`, the same function as `torch.nn.functional.silu`.

## Scenario and config citations

Llama-3.1-8B (`config.json` on Hugging Face): `hidden_size` 4096 (so `k` = 4096), `intermediate_size` 14336 (so the
stacked gate+up weight has `n` = 2 x 14336 = 28672 rows and the output 14336 columns), `hidden_act` "silu". The
gate/up concatenation is the layout of fused `gate_up_proj` projections in serving engines.

## Workloads

`m` is the number of tokens in the step: decode batches of 1 to 64 sequences and prefill chunks of 256 to 8192 tokens.
The 235 MB weight alone takes about 52 us to stream at 4.5 TB/s, so even `m = 1` is above the 10 us floor (estimated
about 56 us); `m = 8192` is 3.85 TFLOP, estimated about 3.9 ms. Seven performance cases span 8192x. Correctness-only
cases: `m = 3`, `17`, `65`, `100`, `129`, `1000` and `2049`. They include the minimum, a non-power-of-two size,
sizes that are not multiples of 16, and the first size above each of the baseline's `M_LEQ_64`, `M_LEQ_128` and
`M_LEQ_2048` configurations. Memory: the float64 reference of the largest case needs 0.9 GB for the weight, 0.9 GB for
the output and a few hundred MB of chunk temporaries.

## Inputs

`x` is standard normal (a normalized hidden state); `w` is normal with standard deviation 1/64 = 1/sqrt(4096), the
scaling of AITER's test generator, which keeps the gate and up projections at unit variance. The product therefore has
a non-trivial SiLU argument (both signs, values around 0 and beyond 3), so a wrong gate or a swapped half is
detected.

## Comparison

Default `baseline_relative`, no override. The kernel is a single pass with float32 accumulation and no split-K or
atomics, so it is deterministic. The reference computes everything in float64 (rows in chunks of 1024).

## Tuned configurations

The tuned configurations come straight from AITER's gfx950 table (the `M_LEQ_8/16/32` rows are
unreachable because of the upstream `bounds`).
