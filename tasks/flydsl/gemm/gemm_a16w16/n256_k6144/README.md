# bfloat16 GEMM (NT), N=256, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[256, 6144]` and `out[m, 256]`, all bfloat16 row-major: a narrow
projection (256 outputs) of a 6144-wide hidden state. The problem streams `a` (about 12 KB per row), so
row tiles and split-K over `k` decide the speed for small `m`.

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
- **Dispatch table:** `CONFIGS` reproduces AITER's FlyDSL rows for this shape from the GLM-5 model tuning file
  (bfloat16 output, gfx950) for m = 4 to 128 and 16384; m = 1 and 2 (tuned to the skinny kernel) use the m <= 4
  row. m = 129 to 1024 reuse the m = 256 row's tile (the tuned m = 256 row is itself FlyDSL, with split-K 6 and
  counters within the kernel's limit; the table applies it up to 1024), and m = 1025 to 8192 use a single-pass
  64x64x128 tile with the shape of AITER's rows for N = 256, K = 4096. These two ranges are hand-derived.
- **Workloads:** decode batches and the tuned sizes up to 256 are correctness-only; timed cases are 512 to 131072
  tokens (`m` spans 256x; the largest case keeps `a` at 1.6 GB, below the 2^31 bytes that the kernel's 32-bit
  buffer offsets need). AITER measured 9.1 us at m = 256, so m = 512 is estimated at 10 to 12 us and is the case most
  likely to move to correctness-only if the GPU audit finds it under 10 us.

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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 15 case(s) failed the default bound, up to 0.98% of the elements over it, peak 2.33x at `k` = 2, about 0.58x at `k` = 8.
