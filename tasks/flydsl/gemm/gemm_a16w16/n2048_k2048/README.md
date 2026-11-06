# bfloat16 GEMM (NT), N=2048, K=2048

`out = a @ b.T` with `a[m, 2048]`, `b[2048, 2048]` and `out[m, 2048]`, all bfloat16 row-major: a square
2048x2048 projection (for example an attention output projection of a model with hidden size 2048). The
8 MB weight fits easily in cache, so small `m` is launch bound (below the 10 us timing floor) and large `m`
is compute bound.

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
- **Dispatch table:** AITER has no tuned row for this shape with bfloat16 output (it falls back to the
  torch/hipBLASLt default), so `CONFIGS` applies the FlyDSL HGEMM configurations AITER tunes for neighbouring
  shapes: split-K 8 with 16-row, 64-deep tiles (a 128-deep tile would leave fewer k iterations than pipeline stages) for `m <= 64`, split-K 4 with 32-row tiles up to 256, then single-pass
  64x128x64, 128x128x64 and 256x256x64 tiles. The rows are not tuned on hardware.
- **Scenario:** replaces the earlier gfx950 and gfx942 tasks of the same shape, whose baseline was the
  installed `aiter.tuned_gemm.gemm_a16w16`. The gfx942 variant is not part of this task (`platform_support` is
  gfx950); the HGEMM kernel has a gfx942 path, so that entry can be added after verification.
- **Workloads:** decode batches and short prefills (1 to 512, plus 300 as the unaligned case) are
  correctness-only because they take under 10 us; the timed cases are prefill chunks and batched prefills of
  1024 to 131072 tokens (128x span). The smallest timed case (1024 tokens, about 8.6 GFLOP) is estimated near
  13 us and must be confirmed on MI355X.
- **Not verified here:** the kernel needs FlyDSL and gfx950. Rows were checked against the kernel's tiling,
  LDS and split-K counter constraints on the CPU only.

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
