# Ragged grouped bf16 GEMM (MoE expert down-projection), 64 groups, K=1408, N=2048

`out[rows of g] = lhs[rows of g] @ rhs[g]` for the 64 groups `g`. `lhs[m, 1408]` holds the routed rows sorted by expert,
`group_sizes[64]` (int32, summing to `m`) says how many consecutive rows belong to each group, `rhs[64, 1408, 2048]`
holds one `[K, N]` matrix per group (the non-transposed forward layout of the operator) and `out[m, 2048]` is
bfloat16. Accumulation is in float32 and each output is rounded to bfloat16 once. Groups may be empty. This is the
forward grouped GEMM of a mixture-of-experts layer (the "GMM" of MoE training stacks); the var axis is `m`, the total
number of routed rows.

## Production origin of the baseline

AITER's Triton `gmm` (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT): the persistent kernel
`gmm_kernel` and its helpers of `aiter/ops/triton/_triton_kernels/gmm.py`, launched by `gmm` in
`aiter/ops/triton/gmm.py`. Each of `GRID_DIM` = 256 programs walks the (group, row tile, column tile) space of all
groups with a stride of 256 tiles, so the ragged group sizes need no host synchronization and the launch is
graph-capturable. The tile configuration is the `gmm` / `default` entry of `aiter/ops/triton/configs/gfx950-GMM.json`
(256x256 output tiles, `BLOCK_SIZE_K` 64, `GROUP_SIZE` 4, 4 warps, 2 stages), which upstream's `get_config` returns for
every shape ("TODO: Fine tune GMM kernels" in the source).

- `gmm_kernel`, `_gmm`, `_process_gmm_tile`, the tile helpers and the unused work-stealing path are copied unchanged
  from the header `SPDX-License-Identifier: MIT` / `Copyright (C) 2025-2026, Advanced Micro Devices, Inc.` file; the
  persistent tgmm kernels (the weight-gradient operator) and the file-based `get_config` are left out. Only the module
  docstring and the Python constant for the configuration are new. `baseline/pid_preprocessing.py` is a verbatim copy
  of `aiter/ops/triton/utils/_triton/pid_preprocessing.py`.
- `run()` fixes `TRANS_RHS=False`, no bias and no work stealing (the default of the upstream wrapper), so the tile
  counter is not used and no atomics are involved: the kernel is deterministic.
- Third-party origin: the wrapper docstring of the sibling `ptgmm` says the "t" of "tgmm" derives from the MaxText
  `megablox` implementation (a JAX/Pallas kernel). Only the naming is borrowed; no code from that project is in the
  Triton kernel or its first commit (`cb0a08849`, "Add Group Matrix Multiplication (GMM) Triton kernels to AITER").
  No third-party license text is therefore added.

## Scenario and config citations

DeepSeek-V2-Lite (`config.json` on Hugging Face): `hidden_size` 2048 (the output width `n`), `moe_intermediate_size`
1408 (the reduction length `k` of the down-projection), `n_routed_experts` 64 (`num_groups`) and `num_experts_per_tok`
6. The shape `M=49152, K=1408, N=2048, G=64` is the one AITER's own GMM tests name "deepseekv2-16B"
(`op_tests/triton_tests/test_gmm.py`); 49152 = 8192 tokens x 6 experts.

## Workloads

`m` is tokens x 6: 64 to 16384 tokens per layer call (training micro-batches and prefill chunks). Performance cases
are `m` = 384, 1536, 6144, 24576, 49152 and 98304 (a 256x span). At the smallest, nearly all experts hold a few rows,
so the whole 369 MB of expert weights is read: about 80 us. The largest is 0.57 TFLOP, about 0.6 ms. Correctness-only
cases: `m` = 1 (one expert holds the only row, 63 groups are empty), 6, 17 (not a multiple of 16), 100 (unaligned, empty
experts) and 1000 (not a multiple of the 256-row tile). Memory for the largest case: inputs and bf16 output are about
1 GB; the float64 reference holds a 1.6 GB output plus per-group conversions of at most a few hundred MB.

## Inputs

`lhs` is standard normal; `rhs` is normal with standard deviation 0.02665 = 1/sqrt(1408), so every output element has
unit variance. `group_sizes` is drawn by `inputs.py` as the per-expert histogram of `m` independent rows routed with
lognormal-skewed expert popularity (spread 0.7): at `m` = 49152 the busiest expert holds about 5x the mean load, at small
`m` several experts are empty, and no case puts all rows on one expert (except the one-row case, where that is
unavoidable). The group boundaries are not tile multiples, so partial 256-row tiles are exercised throughout.

## Comparison

Default `baseline_relative`, no override. One K loop in float32, no split-K, no atomics, fixed tile order.

## Kernel notes

`tl.device_assert` calls in the kernel compile out unless Triton debug is enabled. The kernel is persistent with 256 programs, so
the smallest cases (a handful of tiles) leave most CUs idle; they are correctness-only except the 384-row case, which is
weight-bandwidth bound.
