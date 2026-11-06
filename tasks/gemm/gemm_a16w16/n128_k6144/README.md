# bfloat16 GEMM (NT), N=128, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[128, 6144]` and `out[m, 128]`, all bfloat16 row-major: a narrow
projection (128 outputs) of a 6144-wide hidden state, such as an expert-routing or small head projection.
With so few output columns the problem is bound by streaming `a` (about 12 KB per row), so row tiles and
split-K over `k` decide the speed for small `m`.

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
- **Dispatch table:** AITER's default tuning file has only assembly/OPUS rows for this shape, which cannot be
  vendored. The model tuning files shipped with AITER (GLM-5 and MiniMax-M3-Eagle, bfloat16 output, gfx950)
  contain FlyDSL rows for N=128, K=6144; `CONFIGS` reproduces them (m = 2 to 128 from the first, 196 to
  1024 and 4096 from the second, 16384 from the first). Sizes AITER sends to another library (m = 1 skinny
  kernel, 2048 assembly, 8192 and 32768 torch) use the FlyDSL row of the next tuned size at or above them.
- **Scenario:** replaces the earlier gfx950 and gfx942 tasks of the same shape, which called the installed
  `aiter.tuned_gemm.gemm_a16w16` (hipBLASLt, torch, assembly or FlyDSL depending on `m`). The gfx942 variant
  is not part of this task: `platform_support` is gfx950; the HGEMM kernel also has a gfx942 path, so a
  gfx942 entry can be added after it is verified there.
- **Workloads:** decode batches and the tuned sizes (1 to 1024) are correctness-only; the timed cases are
  tuned prefill sizes 2048 to 32768 plus 1600, 65536 and 163840 tokens (`m` spans 102x; the largest case keeps
  `a` below 2^31 bytes, which the kernel's 32-bit buffer offsets need). The smallest timed cases are estimated
  at roughly 8 to 10 us from the bytes moved plus launch overhead (AITER's tuning measured 10.6 us at
  m = 1024), so m = 1600 and 2048 may fall under the 10 us floor and are the first to move to
  correctness-only if the GPU audit says so; the 100x span then needs a larger `m` range.
- **Not verified here:** the kernel needs FlyDSL and gfx950. The table rows were checked against the
  kernel's tiling, LDS and split-K counter constraints on the CPU only.

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
