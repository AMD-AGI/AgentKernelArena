# bfloat16 GEMM (NT), N=4096, K=2048

`out = a @ b.T` with `a[m, 2048]`, the row-major weight `b[4096, 2048]` (the `nn.Linear` layout) and `out[m, 4096]`, all
bfloat16, accumulated in float32 or better and rounded once. No bias. The var axis is `m`. Both inputs are standard
normal. The scenario comes from the earlier `gemm_a16w16_nt_n4096_k2048` task, whose baseline was
`aiter.tuned_gemm.gemm_a16w16`.

- **Baseline:** AITER's FlyDSL split-K HGEMM kernel (`aiter/ops/flydsl/kernels/splitk_hgemm.py`, ROCm/aiter at the
  recorded revision, MIT) with its `tensor_shim`, `buffer_ops` and `vector` helpers, launched through the per-`m` table
  `CONFIGS` in `baseline/kernel.py`. For `m <= 256` the table holds the FlyDSL rows of AITER's tuned file for this
  (N, K) on gfx950 (`model_configs/glm5_bf16_tuned_gemm.csv`): split-K 1 to 4 with 16-row tiles for decode, a 64x64x128
  tile at `m = 256`. AITER pads `m` up to the next tuned row, which the bucket bounds reproduce. The same file sends
  `m = 16384` to an assembly kernel, and there is no tuned row for the other large sizes; both cannot be vendored,
  so `m > 256` uses the single-pass FlyDSL rows AITER tuned at the same `m` for the 5120x2880 projection
  (`gptoss_bf16_tuned_gemm.csv`, rows 512, 1024, 2048 and 8192, whose tiles divide N=4096 and K=2048); the 8192 row also
  serves every larger `m`. Those rows are not tuned for this shape, so the baseline is probably slower than
  production at large `m`; it is still this task's baseline.
- **Baseline changes:** the kernel was a builder returning a launcher; `run()` selects the table row for `m`, compiles each configuration once (`functools.lru_cache`) and launches on the current stream. Absolute `aiter` imports became relative; environment switches, `numpy` and star imports were removed from the helpers so the sources pass the FlyDSL candidate rules. `buffer_ops.py` and `vector.py` are AITER's copies of ROCm/FlyDSL code (Apache-2.0, headers kept); `baseline/LICENSE` is AITER's MIT license. `splitk_hgemm.py` differs from AITER only in its import line.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged against the bfloat16 output, with the `baseline_relative` bound at `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative against an FP32 reference, which AITER's own split-K dispatches could not meet; here each element is judged against the baseline's own error. Split-K configurations reduce inside the kernel (zero-fill, then atomic bfloat16 adds guarded by a counter pair the kernel resets), so their rounding error is larger than a single-pass kernel's; that is part of the baseline error budget.
- **Workloads:** `m` = 1 to 128 (decode and small batches; the earlier sweep was 1..4096 in powers of two) plus the unaligned
  sizes 33, 100, 200, 300 and 1000 are correctness-only: the tuned kernel needs 6 to 10 us there (the tuned file
  lists 6.2 us at `m = 1` and 9.5 us at `m = 128`), below the 10 us timing floor. Performance cases are `m = 256 .. 65536`
  (256x span, the tuned file lists 11.6 us at 256): prefill chunks and batched prefill. The largest output is 512 MiB.
  Boundary correctness cases: `m = 1`, `m = 33` and `m = 100`.
- **Not verified here:** the kernel needs FlyDSL and gfx950. The table was checked on the CPU against the kernel's tiling,
  LDS and split-K counter constraints only. Risks for the GPU audit are the `m = 256` case sitting close to the 10 us
  floor, the untuned large-`m` rows, and the behaviour of the spin-waiting split-K counters under CUDA-graph replay.

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
