# bfloat16 GEMM (NT), N=6144, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[6144, 6144]` and `out[m, 6144]`, all bfloat16 row-major: a square hidden-to-hidden projection. `b` is 75 MB, so even `m = 1` is a weight-streaming problem of about 15 us; split-K over `k` and row tiles decide the speed for small `m`, MFMA throughput for large `m`.

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
- **Dispatch table:** `CONFIGS` reproduces AITER's FlyDSL rows for this shape from the GLM-5 and MiniMax-M3-Eagle model tuning files (bfloat16 output, gfx950) for m = 1, 2, 4, 8, 16, 32, 48, 64, 128, 196, 320 and 512. Those files send m = 256 and 1024 to 16384 to torch/hipBLASLt and m = 8192 to an assembly kernel; the table instead serves m = 129 to 256 with the m = 196 row and everything above 320 with the m = 512 row (128x128x64, four stages, single pass). These two choices are hand-derived.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task `gemm_a16w16_nt_n6144_k6144`, which called the installed
  `aiter.tuned_gemm.gemm_a16w16` (hipBLASLt, torch, assembly, skinny or FlyDSL depending on `m`) and gated every
  element at 1e-2 against an FP32 reference. Sizes that AITER sends to a non-FlyDSL library cannot be vendored, so
  the FlyDSL table covers every `m`. `platform_support` is gfx950 (the earlier gfx942 variant is not carried over).
- **Not verified here:** the kernel needs FlyDSL and gfx950; only the CPU contract check ran. Table rows were checked
  on the CPU against the kernel's static assertions (warp/tile divisibility, vector-load register counts, pipeline
  depth against k-steps, LDS size, split-K counter capacity for every `m` a row serves); hand-derived rows have
  not been compiled or timed.
- **Workloads:** m = 1 to 131072. Timed cases are m = 1, 8, 64, 256, 1024, 2048, 4096, 16384, 65536 and 131072. AITER measures 14.6 us at m = 1 (the 75 MB weight at about 5 TB/s) and 22.9 us at m = 64, so every timed case should exceed the 10 us floor; the problem is compute-bound above m of about 1000 (2 x 6144 x 6144 FLOP per row). The largest case keeps `a` and `out` at 1.6 GB each, below the 2^31 bytes that the kernel's 32-bit buffer offsets need. Remaining sizes are correctness-only (tuned sizes, odd batches and a non-power-of-two prefill).

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
