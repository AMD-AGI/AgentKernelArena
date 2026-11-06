# bfloat16 GEMM (NT), N=6144, K=4096

`out = a @ b.T` with `a[m, 4096]`, `b[6144, 4096]` and `out[m, 6144]`, all bfloat16 row-major: a wide
projection (for example a fused QKV or gate/up style slice) of a model whose hidden size is 4096. The 50 MB
weight is read once per launch, so small `m` is weight-bandwidth and launch bound and large `m` is compute bound.

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
- **Dispatch table:** the earlier task's baseline was `aiter.tuned_gemm.gemm_a16w16`, which for this shape
  chooses among assembly, Opus and FlyDSL kernels (and torch/hipBLASLt for large `m`); the tuned rows in
  `aiter/configs/bf16_tuned_gemm.csv` are for float32 output and those kernels cannot be vendored as Python source.
  `CONFIGS` therefore applies hand-selected configurations of the FlyDSL HGEMM kernel: split-K with
  16-row tiles for `m <= 64`, split-K 2 with 32-row tiles up to 128, then single-pass 64x128x64, 128x128x64 and
  256x256x64 tiles. Rows were chosen to satisfy the kernel's tiling, LDS and stage constraints and the
  256-counter limit of the split-K reduction (with N=6144 that restricts split-K to at most 192 output tiles);
  they are not tuned on hardware.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output. The earlier task used a float32 reference.
- **Workloads:** decode batches and short prefills are correctness-only (plus 300 and 1000 as unaligned
  cases); the timed cases are 64 to 131072 tokens (2048x span). The smallest timed case is estimated at
  about 15 us and must be confirmed on MI355X; the earlier task covered `m` 1 to 4096 and this one extends the timed
  range to long batched prefills.
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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 8 case(s) failed the default bound, up to 0.47% of the elements over it, peak 2.11x at `k` = 2, about 0.53x at `k` = 8.
