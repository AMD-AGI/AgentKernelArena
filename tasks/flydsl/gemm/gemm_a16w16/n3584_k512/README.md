# bfloat16 GEMM (NT), N=3584, K=512

`out = a @ b.T` with `a[m, 512]`, `b[3584, 512]` and `out[m, 3584]`, all bfloat16 row-major. A 3584-wide output projection of a 512-deep reduction (a low-rank / small-hidden up-projection): token counts of decode batches, prefill chunks and long batched prefills.
b is 3.7 MB; row tiles and split-K over `k` decide the speed for small `m`.

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
- **Dispatch table:** AITER's tuning files have no FlyDSL row for N=3584, K=512; the table is a hand-derived selection of the FlyDSL HGEMM kernel's tiles. The reduction is only 512 long, so split-K is used only for m <= 64 (two slices, the in-kernel counters limit it to 256 tiles) and the kernel's pipeline depth is limited to the 8 (tile_k=64) or 4 (tile_k=128) k-steps of one tile; tiles grow with m above, ending in the 256x256 tile of the kernel's eight-wave pipeline. The rows were checked on the CPU against the kernel's static
  assertions (tile and warp divisibility, vector-load register counts, pipeline depth against k-steps,
  LDS size, split-K counter capacity for every `m` a row serves); they have not been compiled or timed.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task `gemm_a16w16_nt_n3584_k512`, which called the installed `aiter.tuned_gemm.gemm_a16w16`
  (hipBLASLt, torch, assembly or FlyDSL depending on `m`) and gated every element at 1e-2 against an
  FP32 reference. `platform_support` is gfx950; the HGEMM kernel also has a gfx942 path, which is not
  verified for this table.
- **Workloads:** 3584 to 360448 tokens (100.6x); timed cases start at m = 3584, whose estimate (bytes / 4.5 TB/s plus a 4 us floor, about 11 us) clears 10 us; m = 2048 (about 8.5 us) and smaller are correctness-only. The problem is output-bound (7 KB of output per row against 1 KB of input). The kernel addresses `a` and `out` with 32-bit byte offsets, so `run()` launches at most 131072 rows per call (a 940 MB slice of `out`) and picks the table row for each chunk; the largest case (2.6 GB of output, 10.3 GB of float64 reference) therefore runs as three launches (131072, 131072 and 98304 rows). It is the 100x span that requires this large m.
- **Not verified here:** the kernel needs FlyDSL and gfx950; only the CPU contract check was run.

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
