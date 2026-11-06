# BF16 GEMM, N=6144, K=16384 (NT)

`out = a @ b.T` with `a` of shape `[m, 16384]` and the row-major `[6144, 16384]` weight `b` (the `nn.Linear`
layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias. The var axis is `m`.
Both inputs are standard normal.

## Production origin

A tensor-parallel slice of a very deep projection (reduction width 16384, 6144 output columns, as in the output
projection of a Llama-405B-class model). The 201 MB weight makes every `m` up to a few hundred weight-read
bound (about 45 us at 4.5 TB/s), and large `m` compute bound (201 MFLOP per row).

## Workloads

Because even `m = 1` needs at least about 45 us (the weight is read once), decode sizes are performance cases
here: `m = 1, 8, 32, 128, 512, 2048, 8192, 32768` (32768x span, 8 cases). Correctness-only cases are
`m = 2, 4, 16, 33, 64, 100, 256, 300, 1024`. The boundary correctness cases are `m = 1`, `m = 33` and `m = 100`.
`m` stops at 32768 because the activation of 65536 rows is exactly 2 GiB, which the kernel's 32-bit byte offsets
cannot address. The largest case holds a 1 GiB activation, a 4.3 GB float64 reference copy of it and a 1.6 GB
float64 output.

## Baseline

AITER's checkout has no tuned bf16 row for N=6144, K=16384, so its `gemm_a16w16` sends every M to the
hipBLASLt/torch default, which cannot be vendored. `baseline/` therefore contains:

- `splitk_hgemm.py`, `tensor_shim.py`, `buffer_ops.py`, `vector.py`: AITER's FlyDSL HGEMM kernel and the
  helper modules it imports (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT,
  `baseline/LICENSE`), with relative imports. Environment-variable switches were replaced by their defaults,
  the numpy stride computation by a Python one, and the star import of the vector dialect by the single op the
  kernel uses (`broadcast`), so that the package meets the candidate rules. The kernel code itself is unchanged.
- `kernel.py`: the `run()` wrapper. It replaces AITER's launcher (`aiter/ops/flydsl/gemm_kernels.py`): the kernel
  is compiled once per tile configuration, the split-K semaphores are allocated once, and the launch uses the
  current stream. `run()` takes the first table row whose bound covers M (rows above the last use the last row).
- The table is not AITER's dispatch for this shape (there is none). Its rows adapt the FlyDSL configurations that
  AITER's model tuning files list for N=6144 with K = 2048, 3072, 4096 and 6144, to the 4x deeper reduction of
  K=4096: tile shapes are kept, split-K is raised (up to 8) for small `m` so that 96 or 48 output-column tiles
  times the split fill the 256 CUs, and the split-K tile counter limit (at most 256 tiles of M x N per launch)
  is respected. Large `m` uses 128x128x64 and 256x128x64 single-pass tiles. None of the rows was tuned or
  measured for this shape; the baseline is likely slower than hipBLASLt for large `m`; it is still this task's
  baseline.

## Comparison

`baseline_relative` with `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative
against an FP32 reference, which AITER's own split-K dispatches fail for some elements (the old README records
failures on 11 of 13 sizes for N=512, K=4096); here each element is judged against the baseline's error. The
reference is float64 `a.double() @ b.double().T`. Split-K kernels add partial sums into the bf16 output with
atomic adds, so the baseline carries more rounding error than a single-pass kernel; that error sets the scale.

## Risks for the GPU audit

- The table is an untuned adaptation; K=16384 with split-K up to 8 means long accumulation chains, so the
  baseline's bf16 atomic-add error may be large. The baseline must reach 25 dB SQNR against the float64
  reference, otherwise the table needs fewer split-K partitions.
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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 6 case(s) failed the default bound, up to 0.40% of the elements over it, peak 2.06x at `k` = 2, about 0.51x at `k` = 8.
