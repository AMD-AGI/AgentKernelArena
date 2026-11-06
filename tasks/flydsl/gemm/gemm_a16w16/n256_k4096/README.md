# bf16 GEMM, N=256, K=4096

Dense bfloat16 matrix multiplication in the NT layout, `out = a @ b.T`, with `a[M, 4096]` activations,
a `[256, 4096]` row-major weight `b` and a `[M, 256]` bfloat16 output. Accumulation is in float32 or
better and the result is rounded to bfloat16 once. `M` is the only varying axis.

## Production origin

A narrow 4096-to-256 linear projection (for example a router or gate projection), listed in AITER's tuned bf16 GEMM catalogue for several model configurations (`dsv4`, `qwen3_5_397b`).

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

The AITER dispatch is not reproduced exactly. AITER selects a skinny kernel for M=1 and 2, an assembly split-K kernel for M=832..2048 (apart from some FlyDSL rows) and hipBLASLt at 16384. The baseline uses the M=4 entry for the first two, the M=800 entry (32x64x256 tiles, split-K 2) for the assembly range, and the M=14336 entry (128x128x64 tiles) above the table; M between 2048 and 4096 uses the M=4096 entry. The checkout at the recorded commit has no tuned row for
this shape in its main catalogue, so every row comes from the model configurations. Split-K configurations are
limited to 256 output tiles (the semaphore capacity of the kernel); a bucket that would exceed it falls back to
split-K 1. Split-K kernels add the partial sums into the bfloat16 output with atomic adds, so the baseline carries more
rounding error than a single-pass kernel.

## Workloads

M from 1024 to 262144 (a 256x span of prefill chunks and batched prefill tokens). AITER's tuned catalogue records about 12.5 us (M=1024) and 17.4 us (M=2048) for this shape, so M of 1024 and above exceeds the 10 us floor; smaller batches are correctness-only. Boundary correctness cases cover M=1, a non-power-of-two M and non-multiples of 16 (including a ragged
prefill chunk of 1000 rows). Inputs are standard-normal for both operands, as in the previous task.

## Correctness bound

The `baseline_relative` bound is used with `k` = 8 (see below). The previous fixed bf16 gate (absolute or relative 1e-2 on every element)
rejected the AITER split-K dispatches of the production baseline itself on most M; the baseline's own error now sets
the scale. The baseline must reach 25 dB against the float64 reference, and a candidate must stay within eight times
the baseline's per-element error (plus one output rounding) with an unbiased mean.

## Risks for the GPU audit

Timing is tight at the smallest performance case (M=1024, about 12.5 us in AITER's catalogue, with the FlyDSL stand-in untimed); if the QA timing check rejects it, make m1024 correctness-only (M=2048 still leaves a 128x span). M up to 262144 needs a 2 GiB activation and an 8.6 GiB float64 reference copy. The baseline kernel is adapted from a FlyDSL release older than the installed 0.3.2 and has not run on
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

Measured for this scenario (MI355X QA audit r1b, initial candidate = exact copy of the baseline): 8 case(s) failed the default bound, up to 1.17% of the elements over it, peak 1.93x at `k` = 2, about 0.48x at `k` = 8.
