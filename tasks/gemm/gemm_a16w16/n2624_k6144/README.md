# bfloat16 GEMM (NT), N=2624, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[2624, 6144]` and `out[m, 2624]`, all bfloat16 row-major: an
intermediate-width projection of a 6144-wide hidden state. 2624 = 41 x 64, so the kernel's output tile
is limited to 64 columns.

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
- **Dispatch table:** `CONFIGS` reproduces AITER's FlyDSL rows from the GLM-5 model tuning file (bfloat16 output,
  gfx950) for m = 1 to 64 (m = 1 and 2 share the m = 2 row). AITER tunes m = 128 and 256 to an assembly kernel and
  m = 16384 to torch/hipBLASLt, which cannot be vendored, so m = 65 to 128 use the m = 64 row's tile with split-K 3,
  m = 129 to 2048 a single-pass 64x64x128 tile and larger m a 128x64x64 tile (shapes of AITER's FlyDSL rows for
  other narrow-N shapes). Those ranges are hand-derived and unlikely to match the hipBLASLt speed at large m,
  which is acceptable for a baseline; the candidate is scored against this baseline.
- **Workloads:** decode batches and tuned sizes up to 256 are correctness-only; timed cases are 512 to 65536 tokens
  (`m` spans 128x; the largest case keeps `a` at 0.8 GB). m = 512 is estimated at about 16 us from the flop count.

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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 10 case(s) failed the default bound, up to 0.37% of the elements over it, peak 2.36x at `k` = 2, about 0.59x at `k` = 8.
