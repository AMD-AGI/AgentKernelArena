# bf16 GEMM, N=2048, K=4096

Dense bfloat16 matrix multiplication in the NT layout, `out = a @ b.T`, with `a[M, 4096]` activations,
a `[2048, 4096]` row-major weight `b` and a `[M, 2048]` bfloat16 output. Accumulation is in float32 or
better and the result is rounded to bfloat16 once. `M` is the only varying axis.

## Production origin

A 4096-to-2048 linear projection as it appears in the DeepSeek-V4 shape list of AITER's tuned bf16 GEMM catalogue (the `dsv4` model configuration).

## Baseline

The baseline is the FlyDSL split-K half-precision GEMM kernel (`ROCm/FlyDSL`, `kernels/hgemm_splitk.py`,
Apache-2.0) with the per-M tile and split-K dispatch of AITER (`ROCm/aiter` at commit
652f2d4b768239aefd515c929e84817681ff598d, MIT), whose `aiter/ops/flydsl/gemm_kernels.py` drives the same kernel
family from `aiter/configs/bf16_tuned_gemm.csv` and `aiter/configs/model_configs/*bf16_tuned_gemm.csv`
(gfx950, 256 CUs, bf16 in and out). The table in `baseline/kernel.py` lists, for this (N, K), the tuned FlyDSL
configuration of each M bucket (stages, tile sizes, split-K, warp layout); `run()` picks the first bucket whose bound
covers M and launches it on the current stream. The kernel file carries the bundled `buffer_ops.py`/`vector.py`
adapters the installed FlyDSL release needs (it has no `flydsl.expr.buffer_ops`); the tensor shim of AITER is
inlined, numpy is removed and the split-K semaphores are allocated once per device. Licenses are in `baseline/LICENSE`
(AITER, MIT) and `baseline/LICENSE-FlyDSL` (Apache-2.0).

The AITER dispatch is not reproduced exactly. Between M=144 and M=192 AITER selects an assembly split-K kernel, and above M=2048 Triton, an OPUS kernel or hipBLASLt. The baseline uses the nearest tuned FlyDSL entry below M=144..192 (the M=128 entry) and the M=2048 entry (128x128x64 tiles, no split-K) above 2048. The checkout at the recorded commit has no tuned row for
this shape in its main catalogue, so every row comes from the model configurations. Split-K configurations are
limited to 256 output tiles (the semaphore capacity of the kernel); a bucket that would exceed it falls back to
split-K 1. Split-K kernels add the partial sums into the bfloat16 output with atomic adds, so the baseline carries more
rounding error than a single-pass kernel.

## Workloads

M from 256 (prefill chunks) to 65536 (batched prefill), a 256x span. Decode batches (M up to 128, 8-12 us on the baseline) are correctness-only. Boundary correctness cases cover M=1, a non-power-of-two M and non-multiples of 16 (including a ragged
prefill chunk of 1000 rows). Inputs are standard-normal for both operands, as in the previous task.

## Correctness bound

See the split-K section below (`k` = 8, no violation fraction). This replaces the earlier `max_violation_fraction`
= 0.01 setting.

## Risks for the GPU audit

The baseline above M=2048 is a FlyDSL kernel, not the production hipBLASLt/Triton/OPUS path, so it can be slower than production at 16384 and 65536 rows. The baseline kernel is adapted from a FlyDSL release older than the installed 0.3.2 and has not run on
GPU hardware in this migration; tile combinations other than those exercised by the previous task come straight from
AITER's tuned catalogue.

## Split-K nondeterminism and the bound (k = 8)

`compare.out` is `baseline_relative` with `k` = 8 and no violation fraction. The baseline dispatches split-K for part of
the M range and combines the partial sums into the bfloat16 output with atomic adds, so its result depends on the
order in which the partial blocks arrive: an exact copy of the baseline differs from the baseline's own run on a few
elements. At the default `k` = 2 that copy fails; the violating elements are dominated by the baseline-error term of
the bound, so their multiple scales as `peak(k=2) * 2 / k`, and `k` = 4 still fails any element measured above 2x.
`k` = 8, the contract maximum, brings the measured peaks of all split-K gemm_a16w16 scenarios (1.65x to 2.36x at
`k` = 2) to about 0.4x to 0.6x. One uniform rule is used for every split-K scenario of this operator. A 16-ulp shift
of a bfloat16 output is about 6% relative error and fp8 rounding is larger, so both controls stay far outside the
`k` = 8 bound of a baseline whose error is one bfloat16 rounding plus the split-K roundings.

Measured for this scenario (MI355X QA audit r1b, initial candidate = exact copy of the baseline): 5 case(s) failed the default bound, up to 0.49% of the elements over it, peak 1.79x at `k` = 2, about 0.45x at `k` = 8.
