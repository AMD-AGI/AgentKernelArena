# bfloat16 GEMM (NT), N=28672, K=512

`out = a @ b.T` with `a[m, 512]`, `b[28672, 512]` and `out[m, 28672]`, all bfloat16 row-major: a wide
up-projection from a short reduction. `b` is 29 MB and each output row is 57 KB, so small `m` is bound
by streaming `b` and large `m` by writing `out` and by the matrix units.

- **Baseline:** AITER's FlyDSL split-K HGEMM kernel (`aiter/ops/flydsl/kernels/splitk_hgemm.py`, ROCm/aiter
  at the recorded revision, MIT) with its `tensor_shim`, `buffer_ops` and `vector` helpers, launched
  through a per-`m` table of tile, stage, split-K and warp-layout choices in `baseline/kernel.py`. Split-K
  configurations reduce inside the kernel (zero-fill, then atomic bfloat16 adds guarded by a counter
  pair that the kernel resets), so their bfloat16 error is larger than a single-pass kernel's; that is
  part of the baseline's error budget under the `baseline_relative` bound (`k` = 8, see below), which replaces the
  earlier fixed 1e-2 element gate that production dispatches could not meet.
- **Baseline changes:** the kernel was a builder returning a launcher; `run()` selects the table row for
  `m`, compiles each configuration once (`functools.lru_cache`) and launches on the current stream.
  Absolute `aiter` imports became relative; environment switches, `numpy` and star imports were removed
  from the helpers so the sources pass the FlyDSL candidate rules. `buffer_ops.py` and `vector.py` are
  AITER's copies of ROCm/FlyDSL code (Apache-2.0, headers kept); `baseline/LICENSE` is AITER's MIT license.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task of the same shape, which called the installed
  `aiter.tuned_gemm.gemm_a16w16` (hipBLASLt, torch, assembly or FlyDSL depending on `m`). Sizes that AITER
  sends to a non-FlyDSL library cannot be vendored, so the FlyDSL table covers every `m`. `platform_support`
  is gfx950.
- **Not verified here:** the kernel needs FlyDSL and gfx950. The table rows were checked against the
  kernel's tiling, LDS and split-K counter constraints on the CPU only.
- **Dispatch table:** AITER has no tuned row for this (N, K), so `CONFIGS` is hand-derived from the tile shapes of
  AITER's FlyDSL rows for similar shapes. K = 512 leaves only 4 to 8 reduction steps, so every row is single-pass
  (`split_k = 1`); narrow tiles for small m spread the 28672 columns over the CUs and the 128x128x64 tile of AITER's
  large-m rows serves m above 512.
- **Workloads:** the original task's sizes (powers of two from 1 to 4096) are kept and extended up to 32768. The
  maximum is bounded by `out` (m x 57 KB), which must stay below 2^31 bytes for the kernel's 32-bit buffer offsets
  (m = 32768 gives 1.9 GB). Timed cases start at m = 64 (`m` spans 512x); at m = 1 to 32 the estimate is only about
  10 us (streaming `b`), so those stay correctness-only unless the GPU audit shows the baseline above 10 us there.

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
