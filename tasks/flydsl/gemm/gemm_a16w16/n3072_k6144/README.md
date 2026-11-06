# bfloat16 GEMM (NT), N=3072, K=6144

`out = a @ b.T` with `a[m, 6144]`, `b[3072, 6144]` and `out[m, 3072]`, all bfloat16 row-major. A 3072-wide output projection of a 6144-wide hidden state (the bfloat16 FFN/attention-output style slice): token counts of decode batches, prefill chunks and long batched prefills.
b is 37.8 MB; row tiles and split-K over `k` decide the speed for small `m`.

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
- **Dispatch table:** AITER's default tuning file routes N=3072, K=6144 only to assembly and OPUS kernels (m <= 256, 4096 to 16384) or OPUS (512 to 2048), none of which can be vendored, and has no FlyDSL row for this shape; the table is therefore a hand-derived selection of the FlyDSL HGEMM kernel's tiles, modelled on the FlyDSL rows AITER tunes for neighbouring shapes: split-K for the weight-read-bound m <= 128 (the in-kernel counters limit split-K to 256 tiles), single-pass tiles that grow with m above, ending in the 256x256 tile of the kernel's eight-wave pipeline. The rows were checked on the CPU against the kernel's static
  assertions (tile and warp divisibility, vector-load register counts, pipeline depth against k-steps,
  LDS size, split-K counter capacity for every `m` a row serves); they have not been compiled or timed.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged
  against the bfloat16 output.
- **Scenario:** replaces the earlier task `gemm_a16w16_nt_n3072_k6144`, which called the installed `aiter.tuned_gemm.gemm_a16w16`
  (hipBLASLt, torch, assembly or FlyDSL depending on `m`) and gated every element at 1e-2 against an
  FP32 reference. `platform_support` is gfx950; the HGEMM kernel also has a gfx942 path, which is not
  verified for this table.
- **Workloads:** 64 to 131072 tokens (2048x); the largest case keeps `a` (1.6 GB) below 2^31 bytes, which the kernel's 32-bit buffer offsets need. Decode batches and sizes below the timing floor are correctness-only. Each timed case reads at least the 37.8 MB weight (about 8.4 us at 4.5 TB/s plus launch), so even m = 64 is estimated above 10 us (AITER's tuned assembly kernel measures 13.6 to 15.4 us at m = 1 to 64).
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
