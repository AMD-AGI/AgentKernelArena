# bfloat16 GEMM (NT), N=64, K=4096

`out = a @ b.T` with `a[m, 4096]`, `b[64, 4096]` and `out[m, 64]`, all bfloat16 row-major: a narrow projection (64 outputs, for example a router or low-rank down-projection) of a 4096-wide hidden state. The problem streams `a` (8 KB per row) against a 512 KB weight, so timing grows only with `m` and row tiles and split-K over `k` decide the speed.

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
- **Dispatch table:** `CONFIGS` reproduces AITER's FlyDSL rows for this shape from the DeepSeek-V4 and Qwen3.5 model tuning files (bfloat16 output, gfx950) for m = 8 to 2048 (m = 1 to 4 are tuned to the skinny kernel and use the m = 8 row) and m = 8192 to 16384. AITER sends m = 4096 and 6144 to an assembly split-K kernel, which cannot be vendored; m = 2049 to 4096 use a 32x64x64 split-K 4 tile and m = 4097 to 6144 a 48x64x64 split-K 2 tile (both hand-derived, picked so that a single 64-wide column of tiles still fills the GPU), and m above 16384 reuse the m = 16384 row (64x64x128, single pass).
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task `gemm_a16w16_nt_n64_k4096`, which called the installed
  `aiter.tuned_gemm.gemm_a16w16` (hipBLASLt, torch, assembly, skinny or FlyDSL depending on `m`) and gated every
  element at 1e-2 against an FP32 reference. Sizes that AITER sends to a non-FlyDSL library cannot be vendored, so
  the FlyDSL table covers every `m`. `platform_support` is gfx950 (the earlier gfx942 variant is not carried over).
- **Not verified here:** the kernel needs FlyDSL and gfx950; only the CPU contract check ran. Table rows were checked
  on the CPU against the kernel's static assertions (warp/tile divisibility, vector-load register counts, pipeline
  depth against k-steps, LDS size, split-K counter capacity for every `m` a row serves); hand-derived rows have
  not been compiled or timed.
- **Workloads:** m = 1 to 245760. Timed cases are m = 2400 to 245760 (102x). AITER measures about 10.3 us at m = 2048 and 8 us at m = 1024, and the estimate (bytes / 4.5 TB/s plus a 4 us floor) is under 10 us below m of about 3000, so m = 2400 is the case most likely to measure under the floor on the baseline. If the GPU audit moves it to correctness-only, the 100x span cannot be kept (the 2^31-byte limit of the kernel's 32-bit buffer offsets caps `a` at m = 262143, 2.0 GB at m = 245760) and the task should be deferred. Decode batches and sizes up to 2048 are correctness-only.

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
