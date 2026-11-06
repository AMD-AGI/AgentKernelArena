# bfloat16 GEMM (NT), N=32, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[32, 6144]` and `out[m, 32]`, all bfloat16 row-major. A 32-wide router/gate projection of a 6144-wide hidden state: token counts of decode batches, prefill chunks and long batched prefills.
b is only 393 KB; row tiles and split-K over `k` decide the speed for small `m`.

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
- **Dispatch table:** AITER's tuning files send N=32, K=6144 to the skinny kernel (m <= 4) or torch/hipBLASLt (the rest) and have no FlyDSL row for it; the table is therefore a hand-derived selection of the FlyDSL HGEMM kernel's tiles. With N = 32 the kernel needs `TILE_N = 32`, and its C-tile register constraint then limits a 16-row tile to a single warp, so small m uses one-warp tiles with split-K over the long reduction (the in-kernel counters limit split-K to 256 tiles), and large m uses 128x32 tiles that stream `a` once. The rows were checked on the CPU against the kernel's static
  assertions (tile and warp divisibility, vector-load register counts, pipeline depth against k-steps,
  LDS size, split-K counter capacity for every `m` a row serves); they have not been compiled or timed.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task `gemm_a16w16_nt_n32_k6144`, which called the installed `aiter.tuned_gemm.gemm_a16w16`
  (hipBLASLt, torch, assembly or FlyDSL depending on `m`) and gated every element at 1e-2 against an
  FP32 reference. `platform_support` is gfx950; the HGEMM kernel also has a gfx942 path, which is not
  verified for this table.
- **Workloads:** 2560 to 262144 tokens (102x); timed cases start at m = 2560, the first size whose estimate (bytes / 4.5 TB/s plus a 4 us floor, about 11 us) clears 10 us; smaller sizes (including 1600 and 2048, estimated 8 to 10 us) and decode batches are correctness-only. The problem streams `a` (12 KB per row) against a 393 KB weight, so timing grows only with m (AITER's tuned torch path measures 8 to 12 us at m <= 256 and 43 us at m = 16384). The kernel addresses `a` and `out` with 32-bit byte offsets, so `run()` launches at most 131072 rows per call (a 1.6 GB slice of `a`) and picks the table row for each chunk; m = 163840 is a correctness-only case whose second chunk has a different size. The 100x span needs m up to 262144, which is a very long batched prefill.
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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 10 case(s) failed the default bound, up to 3.12% of the elements over it, peak 1.88x at `k` = 2, about 0.47x at `k` = 8.
