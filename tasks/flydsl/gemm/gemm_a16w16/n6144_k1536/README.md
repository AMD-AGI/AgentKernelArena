# BF16 GEMM, N=6144, K=1536 (NT)

`out = a @ b.T` with `a` of shape `[m, 1536]` and the row-major `[6144, 1536]` weight `b` (the `nn.Linear`
layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias. The var axis is `m`.
Both inputs are standard normal.

## Production origin

A 1536-to-6144 projection (a 4x expansion, for example an attention or MLP up-projection of a model with hidden
size 1536). The weight is 18.9 MB and each token costs 18.9 MFLOP, so decode is launch- and weight-read bound and
prefill is compute bound.

## Workloads

Performance cases are `m = 512 .. 65536` (128x span), prefill chunks and batched prefill; at `m = 512` the GEMM
is 9.7 GFLOP, about 10 us of MFMA time plus launch overhead. `m = 1 .. 300` (the old task's 1..4096 sweep plus
unaligned sizes) are correctness-only. The boundary correctness cases are `m = 1`, `m = 33` and `m = 100`.

## Baseline

AITER's checkout has no tuned bf16 row for N=6144, K=1536, so its `gemm_a16w16` sends every M to the
hipBLASLt/torch default, which cannot be vendored. `baseline/` therefore contains:

- `splitk_hgemm.py`, `tensor_shim.py`, `buffer_ops.py`, `vector.py`: AITER's FlyDSL HGEMM kernel and the
  helper modules it imports (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT,
  `baseline/LICENSE`), with relative imports. Environment-variable switches were replaced by their defaults,
  the numpy stride computation by a Python one, and the star import of the vector dialect by the single op the
  kernel uses (`broadcast`), so that the package meets the candidate rules. The kernel code itself is unchanged.
- `kernel.py`: the `run()` wrapper. It replaces AITER's launcher (`aiter/ops/flydsl/gemm_kernels.py`): the kernel
  is compiled once per tile configuration, the split-K semaphores are allocated once, and the launch uses the
  current stream. `run()` takes the first table row whose bound covers M (rows above the last use the last row).
- The table is not AITER's dispatch for this shape (there is none). It reuses the FlyDSL rows that AITER's model
  tuning files list for the same reduction width with half the output columns (N=3072, K=1536, M = 1 .. 1024),
  with the M=1024 row (128x128x64) extended to M=2048; M values that file sends to Triton (M=16) use the next
  row. Above M=2048 the 256x128x64 row that AITER tunes for N=6144, K=2048 at M=1024 is used. The rows are valid
  for this N (6144 divides by every tile width) and satisfy the kernel's assertions, but none was tuned for this
  shape and none was measured on hardware. The baseline is likely slower than hipBLASLt for large M; it is
  still this task's baseline.

## Comparison

`baseline_relative` with `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative
against an FP32 reference, which AITER's own split-K dispatches fail for some elements (the old README records
failures on 11 of 13 sizes for N=512, K=4096); here each element is judged against the baseline's error. The
reference is float64 `a.double() @ b.double().T`. Split-K kernels add partial sums into the bf16 output with
atomic adds, so the baseline carries more rounding error than a single-pass kernel; that error sets the scale.

## Risks for the GPU audit

- The table is borrowed, not tuned, and `m = 512` (about 10 us of ideal compute time) may fall under the 10 us
  floor or stay slow; if the audit shows under 10 us, move the smallest cases to correctness-only and extend the
  span with `m = 131072`.
- No FlyDSL toolchain or GPU was available when this package was written: only the CPU checks ran. Every
  table row was checked on the CPU against the kernel's own assertions (tiling, warp layout, stage count versus
  K loops, shared-memory size, tile-counter limit of the split-K semaphore), but compilation and numerics are
  unverified on gfx950.
- `compile_hgemm_kernel` import paths follow AITER's current FlyDSL API (`ptrtoint`, `fx.Stream`); the runtime
  image's FlyDSL version must provide them.
- The split-K kernel zeroes its output and uses spin-waiting semaphores; behavior under CUDA-graph replay with
  the cold-cache flush should be confirmed.
- The kernel addresses `a` and the output with 32-bit byte offsets, so `a` and `out` stay below 2 GiB.

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm.

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

No candidate run exists for this scenario yet: its last audit stopped at the FlyDSL 0.3.2 `aux` TypeError before the
baseline ran, so the bound follows the measured sibling scenarios of the same kernel family.
