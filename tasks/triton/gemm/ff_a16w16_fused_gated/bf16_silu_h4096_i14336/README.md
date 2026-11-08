# Fused gated bf16 feed-forward block (SwiGLU MLP), hidden 4096, intermediate 14336

`out = (silu(x @ w_up[:14336]^T) * (x @ w_up[14336:]^T)) @ w_down` with `x[n_tokens, 4096]`, the stacked up-projection
`w_up[28672, 4096]` (gate rows first, up rows second), the down-projection `w_down[14336, 4096]` and
`out[n_tokens, 4096]`, all bfloat16. Matrix products accumulate in float32; the gated intermediate is rounded to
bfloat16 before it feeds the down-projection (the float64 reference does not round it); the output is bfloat16. The
var axis is `n_tokens`. `w_down` is stored `[intermediate, hidden]`, the layout AITER's kernel reads (the transpose of
the `nn.Linear` `down_proj.weight`), unlike the `[hidden, intermediate]` weight of the sibling
`ff_a16w16_ungated` tasks.

## Production origin of the baseline

AITER's `ff_a16w16_fused_gated` (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT): the single Triton
kernel `_ff_a16w16_fused_gated` of
`aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_gated.py`, launched by
`aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_gated.py` with `activation="silu"`. One program computes a
`BLOCK_SIZE_M x BLOCK_SIZE_N/2` tile of the gate and of the up half, applies the gate, rounds the tile to bfloat16 and
multiplies it by the matching rows of `w_down`, adding the partial result into `out` with `tl.atomic_add` (so the
intermediate never reaches memory). The tile configuration is AITER's `FF-A16W16-fused` table for gfx950
(`configs/gemm/gfx950-FF-A16W16-fused.json`; no N/K-specific file exists) selected with the upstream bounds
`(4, 8, 64, 4096)`; it has rows for `M <= 4`, `M <= 8`, `M <= 64` and a default for everything larger, all with
`BLOCK_SIZE_N` 256.

- The kernel is byte-identical to upstream except for its two import lines and the module docstring placed after the
  header. `baseline/pid_preprocessing.py` is a verbatim copy of `aiter/ops/triton/utils/_triton/pid_preprocessing.py`;
  `baseline/activation.py` holds `_silu_exp2` and `_silu` from `aiter/ops/triton/_triton_kernels/activation.py`, which
  has no header upstream, like `gemm_config_utils.py`, whose lookup order `run()` reproduces. The repository MIT
  license of ROCm/aiter applies to those (`baseline/LICENSE`). The kernel and pid helper keep the header
  `SPDX-License-Identifier: MIT` / `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`.
- No third-party origin is recorded for these files (no attribution text in or beside them; first commits `67c5d61d2`
  "End-to-end fused feed-forward kernel" and `922891a9e` are AMD's AITER work).
- `run()` zero-fills `out` with `out.zero_()` before the launch, because the kernel only adds into it. That fill is
  part of every timed call, exactly as `torch.zeros` is part of the upstream wrapper.

## Upstream defect: the `M_LEQ_8` row cannot be launched on gfx950

Shared memory per pipeline stage is the x tile plus the gate and up weight tiles that feed `tl.dot`:
`(BLOCK_M x BLOCK_K + 2 x BLOCK_K x BLOCK_N/2) x 2 bytes`, times `num_stages` buffers.

| Row | Tiles (M x N x K), stages | Estimate | vs 163840 B limit |
| --- | --- | --- | --- |
| `M_LEQ_4` | 4 x 256 x 256, 1 | 133120 B | fits |
| `M_LEQ_8` | 8 x 256 x 256, 2 | (4096 + 2 x 65536) B per stage x 2 = 270336 B | over (compiler reported 278464 B) |
| `M_LEQ_64` | 64 x 256 x 256, 1 | 163840 B (+ epilogue scratch) | at the limit; compiles on gfx950 |
| default | 64 x 256 x 256, 1 | same | same |

The estimate matches the reported 278464 B within 3% (the rest, 8128 B, is scratch such as layout conversion that the
estimate does not model). The pinned AITER has no fallback for this shape: its `get_gemm_config` returns the
`M_LEQ_8` row unchanged. `run()` therefore maps the `M_LEQ_8` row to the `M_LEQ_4` row of the same table (the nearest
valid row: same tile widths and warps, `BLOCK_SIZE_M` 4, one stage), so batches of 5 to 8 rows run two 4-row tiles per
N block. The `M_LEQ_4` and `M_LEQ_64` rows are used as upstream has them. The number of N blocks, and so of atomic
adds per element, is 112 for every row (`BLOCK_SIZE_N` 256 throughout), so the comparison justification below is unchanged.

## Scenario and config citations

Llama-3.1-8B (`config.json` on Hugging Face): `hidden_size` 4096, `intermediate_size` 14336, `hidden_act` "silu"; the
stacked gate+up weight has 2 x 14336 = 28672 rows.

## Workloads

`n_tokens` is the number of tokens of a decode step: 1 to 256 concurrent sequences. AITER documents this kernel as
meant for small batches (the wrapper warns above 64 rows because the unfused pair of GEMMs is faster there), so the
axis stops at 256; the cases above 64 exercise the default configuration. The three matrices (352 MB) take about 78 us
to stream at 4.5 TB/s, so even `n_tokens = 1` is above the 10 us floor (estimated about 82 us); `n_tokens = 256` is
90 GFLOP, about 90 us of compute and about the same of streaming (estimated 100 us). Seven performance cases (1, 4, 8, 32, 64,
128 and 256 tokens) span 256x. Correctness-only cases: `n_tokens` = 3, 17, 65 and 100, covering the minimum, a non-power-of-two size, sizes that are not multiples of 16, and the
first size of the default configuration.

## Inputs

`x` is standard normal. `w_up` has standard deviation 1/64 = 1/sqrt(4096), so gate and up projections have unit
variance (AITER's test generator scales weights this way); `w_down` has standard deviation 0.008352 =
1/sqrt(14336), so the output keeps a standard deviation of about 0.6. SiLU sees both signs and values beyond 3.

## Comparison override: `k = 8`, `max_violation_fraction = 1e-4`

Every output element is the sum of 112 partial products (28672 / 256 programs along the up dimension) that the programs
add into the bfloat16 `out` with atomic adds in an arbitrary order, rounding to bfloat16 at every add. Two runs of the
baseline therefore differ, and a correct candidate or a second baseline run must not be rejected for that.

A CPU simulation (64 tokens, the real dimensions, the distributions of `task.yaml`, 112 blocks of 128 gated columns as
the `M <= 64` configuration with `BLOCK_SIZE_N` = 256 uses them; fp32 partials, bfloat16 rounding of each partial and
after every add; two random block orders; the bound of `src/tools/arena_eval/compare.py`) gave:

- SQNR of the emulated baseline against the float64 reference: 37.8 dB (both orders), above the 25 dB sanity bound.
- Peak of |e2| / max(|e1|, row RMS of e1) over the 262144 elements: 9.5, so a second run needs `k` of about 9.5 for
  every element without any admitted violations.
- With the full bound (which adds one output rounding): the second order fails 2.2% of the elements at `k` = 2 (peak
  3.8x), 0.11% at `k` = 4, and 7.6e-6 at `k` = 8 (2 elements, peak 1.12x the bound).
- The framework controls at `k` = 8 without a violation fraction: `fp8_rounded` violates 0.31% of the elements (peak
  2.1x) and `shifted_16_ulp` 4.7% (peak 2.6x), so both are rejected.

`k` = 8 alone would reject the baseline's own second run (a few tail elements), so the task also admits
`max_violation_fraction` = 1e-4: it passes the second run (7.6e-6, 13x margin) and still rejects both controls (31x and
470x over the admitted fraction). Admitted elements stay within 16x the bound by the framework rule, and the simulated peak is
1.12x. Choosing a looser `k` was not needed. The simulation is a model of the kernel's accumulation, one seed, one
batch size (the script is not part of the task).

## Atomics

bfloat16 `tl.atomic_add` with `sem="relaxed"` is used as in AITER's own tests; the M_LEQ_4 and M_LEQ_8 tiles use
`BLOCK_SIZE_M` 4 and 8 for `tl.dot`. Atomics make the baseline nondeterministic, so a control that perturbs by a few ulps
may behave differently from the deterministic tasks.
