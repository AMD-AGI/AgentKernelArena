# RMSNorm backward, bfloat16, hidden size 4096 (FlyDSL baseline)

Backward of `y = x * rsigma * gamma` (rsigma = `rsqrt(mean(x^2) + epsilon)` per token), the training-time companion of
the forward RMSNorm. Inputs are the output gradient `dy`, the saved activation `x`, `gamma` and the per-token `rsigma`
the forward saved. Outputs are `dx` (bfloat16, same shape as `x`) and `dgamma` (bfloat16, the sum over all tokens of
`dy * x * rsigma`, accumulated in float32 and rounded once). The interface and the reference are those of the Triton
task `tasks/triton/normalization/rms_norm_backward/bf16_h4096`, except for the `dgamma` dtype and the token minimum
described below.

`rsigma` is an input, not recomputed: the task-owned generator (`inputs.py`, the same as the Triton sibling's) derives
it from `x` in float32 with epsilon 1e-5, so it is consistent with `x`. The reference (float64, chunked over tokens)
uses the given `rsigma`.

## Baseline

The FlyDSL project's staged RMSNorm backward at `ROCm/FlyDSL@dd8373349e33638c8d158a9ea009b7b28598984a` (tag v0.3.2,
Apache-2.0, `# Copyright (c) 2025 FlyDSL Project Contributors` kept in every vendored file): a persistent
512-thread kernel (`rmsnorm_bwd_partial_kernel`, vec8 I/O for 16-bit rows of at least 4096 elements) that writes `dx`
and one float32 partial-`dgamma` row per program, and `rmsnorm_bwd_dweight_reduce_kernel`, which reduces the partial rows
and stores `dgamma` in the gamma dtype. Source files: `kernels/norm/rmsnorm_bwd_kernel.py`,
`kernels/norm/rmsnorm_common.py`, `kernels/common/kernels_common.py` (only `dtype_to_elem_type` and the
`get_warp_size` re-export); the program count of `run()` follows `_select_rmsnorm_bwd_config` and `rmsnorm_bwd` in
`kernels/norm/rmsnorm_kernel.py` (CU count for fewer than 2048 tokens, 1.5x CU count above, at most one program per
token). Third-party origin: none of the vendored files names one; their git history starts with FlyDSL's own commits.

Modifications (also in `task.yaml`): the two atomic one-row-per-block builders of `rmsnorm_bwd_kernel.py` are removed
(upstream selects them only below 512 tokens), and the absolute `kernels.*` imports are relative. No FlyDSL API port
is needed: the vendored FlyDSL is the version of the evaluation image.

### Why the token axis starts at 512 and `dgamma` is bfloat16

Upstream's `rmsnorm_bwd` runs the staged path for 512 or more tokens and an atomic path below that. The atomic path
returns its float32 `dweight` through a host-side `.to(weight.dtype)`, which the candidate rules forbid in `run()`,
and its float32 atomics are order-dependent. The token axis therefore starts at 512, where upstream always takes the
staged, deterministic path. That path stores `dgamma` in the gamma dtype, so the output is bfloat16 (the Triton
sibling returns float32). Training batches below 512 tokens per rank are not a bandwidth-bound regime anyway.

## Comparison

Both outputs use the default `baseline_relative` bound. For `dgamma` the bound is `2 * max(|baseline error|, row RMS of
the baseline error)` plus one bfloat16 rounding of the reference; a candidate that accumulates the token sum in a
different float32 order can differ from the baseline by one bfloat16 step, which the rounding term and the RMS floor
admit. No bound is loosened.

## Scenario and workloads

Hidden size 4096 is the `hidden_size` of Llama-3 8B and Mistral 7B (public Hugging Face `config.json`), corroborated by the `llama3` / `8B` entry of
`op_tests/op_benchmarks/triton/utils/model_configs.json` in ROCm/aiter at `652f2d4b768239aefd515c929e84817681ff598d`
(`hidden_size` 4096, `num_attention_heads` 32, `intermediate_size` 14336). bfloat16 activations and gamma are the usual mixed-precision training layout. FlyDSL's own test
(`tests/kernels/test_rmsnorm.py`, `_RMSNORM_BACKWARD_CONFIGS`) exercises `(4096, 4096, "bf16")` as its "model-relevant
large shape".

Per token the kernel moves about 24 KB (`x`, `dy` in, `dx` out, bfloat16), so 2048 tokens (about 49 MB, about 15 us
estimated) is the smallest performance case. Performance cases span 2048 to 262144 tokens (128x; the largest moves
6.3 GB, about 1.4 ms). Correctness-only cases: 512 (the minimum), 513, 1000, 1024 and 1537 (not a multiple of the
program count or of 16). The float64 reference is computed in chunks of 32768 tokens; the largest case needs about
13 GB for inputs, outputs and the float64 `dx`.

## Kernel notes

Scratch allocation of `num_programs * 4096` float32 per call (about 1.5 MB) is inside `run()` and capturable. The
launcher is built by the first call of each shape class (two classes: fewer and at least 2048 tokens).
