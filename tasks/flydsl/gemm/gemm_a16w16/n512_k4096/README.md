# BF16 GEMM, N=512, K=4096 (NT)

`out = a @ b.T` with `a` of shape `[m, 4096]` and the row-major `[512, 4096]` weight `b` (the `nn.Linear`
layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias. The var axis is `m`.
Both inputs are standard normal.

## Production origin

A narrow 4096-to-512 projection listed in AITER's tuned bf16 GEMM catalogue for the DeepSeek-V4 (`dsv4`) and
Qwen3.5-397B model configurations. Both weight and output are small, so the problem is bound by streaming `a`
(8 KiB per row) and, at large `m`, by the 4.2 MFLOP per row.

## Workloads

Performance cases are `m = 512 .. 131072` (256x span): prefill chunks and batched prefill, where AITER's tuning
measured 12 us (m=512) to 68 us (m=16384) for its own kernels. `m = 1 .. 300` are decode and small-batch sizes
(the old task's 1..4096 sweep, plus unaligned sizes) and are correctness-only: the tuned kernels need 5 to 10 us
there, below the 10 us timing floor. The boundary correctness cases are `m = 1`, `m = 33` and `m = 100`. The
largest case keeps `a` at 1 GiB.

## Baseline

AITER's `gemm_a16w16` dispatches this shape (gfx950) per M bucket to a skinny kernel (M=1), its FlyDSL split-K
HGEMM kernel, assembly kernels (M=480, 544..800, 864, 1088..1216), opus kernels and hipBLASLt/torch (M >= 8192).
Only the FlyDSL kernel can be vendored as source, so `baseline/` contains:

- `splitk_hgemm.py`, `tensor_shim.py`, `buffer_ops.py`, `vector.py`: AITER's FlyDSL HGEMM kernel and the
  helper modules it imports (ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d, MIT,
  `baseline/LICENSE`), with relative imports. Environment-variable switches were replaced by their defaults,
  the numpy stride computation by a Python one, and the star import of the vector dialect by the single op the
  kernel uses (`broadcast`), so that the package meets the candidate rules. The kernel code itself is unchanged.
- `kernel.py`: the `run()` wrapper. It replaces AITER's launcher (`aiter/ops/flydsl/gemm_kernels.py`): the kernel
  is compiled once per tile configuration, the split-K semaphores are allocated once, and the launch uses the
  current stream. `run()` takes the first table row whose bound covers M (rows above the last use the last row).
- The table lists the 47 FlyDSL HGEMM rows of AITER's model tuning files for N=512, K=4096 (M = 2 .. 6144;
  where both files list an M the faster measurement is kept). M values AITER sends to a library kernel use the
  next FlyDSL row, and every M above 6144 uses the 96x128x64 single-pass row of M=6144. The baseline is therefore
  slower than production for the assembly buckets and for M above 8192 (where AITER uses hipBLASLt or opus);
  it is still this task's baseline.

## Comparison

`baseline_relative` with `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative
against an FP32 reference, which AITER's own split-K dispatches fail for some elements (the old README records
failures on 11 of 13 sizes for N=512, K=4096); here each element is judged against the baseline's error. The
reference is float64 `a.double() @ b.double().T`. Split-K kernels add partial sums into the bf16 output with
atomic adds, so the baseline carries more rounding error than a single-pass kernel; that error sets the scale.

## Risks for the GPU audit

- Timing is tight at `m = 512`: AITER measured 12.4 us for its FlyDSL row and the 10 us floor leaves little room.
  If the audit measures under 10 us, drop `m = 512` (and `m = 1024` if needed) to correctness-only and keep the
  span at 100x with the larger cases.
- No FlyDSL toolchain or GPU was available when this package was written: only the CPU checks ran. Every
  table row was checked on the CPU against the kernel's own assertions (tiling, warp layout, stage count versus
  K loops, shared-memory size, tile-counter limit of the split-K semaphore), but compilation and numerics are
  unverified on gfx950.
- `compile_hgemm_kernel` import paths follow AITER's current FlyDSL API (`ptrtoint`, `fx.Stream`); the runtime
  image's FlyDSL version must provide them.
- The split-K kernel zeroes its output and uses spin-waiting semaphores; behavior under CUDA-graph replay with
  the cold-cache flush should be confirmed.
- The kernel addresses `a` and the output with 32-bit byte offsets, so `a` and `out` stay below 2 GiB.

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
