# BF16 GEMM, N=1024, K=4096 (NT)

`out = a @ b.T` with `a` of shape `[m, 4096]` and the row-major `[1024, 4096]` weight `b` (the `nn.Linear`
layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias. The var axis is `m`.
Both inputs are standard normal.

## Workloads

`m = 1 .. 128` (the decode and small-batch sizes of the earlier task's 1..4096 sweep), plus `m = 33`, `100`,
`200` and `300` (unaligned sizes, `300` falls between tuned rows), are correctness-only: the tuned
production kernel needs 6 to 10 us there, below the 10 us timing floor. Performance cases are
`m = 256 .. 65536` (256x span), prefill chunks and batched prefill, where the tuned baseline needs 12 us and
more. The boundary correctness cases are `m = 1`, `m = 33` and `m = 100`.

## Baseline

AITER's `gemm_a16w16` dispatches this shape (on gfx950) per M bucket to its FlyDSL split-K HGEMM kernel, to
assembly kernels, to opus kernels or to hipBLASLt/torch. Only the FlyDSL kernel can be vendored as source, so
`baseline/` contains:

- `splitk_hgemm.py`, `tensor_shim.py`, `buffer_ops.py`, `vector.py`: AITER's FlyDSL HGEMM kernel and the
  helper modules it imports (MIT, `baseline/LICENSE`), with relative imports. Environment-variable switches
  were replaced by their defaults, the numpy stride computation by a Python one, and the star import of the
  vector dialect by the single op the kernel uses (`broadcast`), so that the package meets the candidate
  rules. The kernel code itself is unchanged.
- `kernel.py`: the `run()` wrapper. It replaces AITER's launcher: the kernel is compiled once per tile
  configuration, the split-K semaphores are allocated once, and the launch uses the current stream.
  The tile table is the FlyDSL rows that AITER's merged tuned configuration lists for gfx950, N=1024, K=4096.
  AITER pads M up to a tuned row. For M values whose tuned row is an assembly kernel (288 to 384, 576) the
  next FlyDSL row is used, and every M above 2048 (where AITER uses opus, assembly or hipBLASLt) uses the
  M=2048 configuration (128x64x128 tiles, no split-K). The baseline is therefore slower than production for
  M above 2048 and for the assembly buckets; it is still this task's baseline.

## Comparison

`baseline_relative` with `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative
against an FP32 reference, which AITER's own split-K dispatches fail for some elements; here each element is
judged against the baseline's error. The reference is float64 `a.double() @ b.double().T`.

## Risks for the GPU audit

- No FlyDSL toolchain or GPU was available when this package was written: only the CPU checks ran. The
  vendored kernel is identical to the one already used in the repository's FlyDSL GEMM tasks, and every
  tile configuration in the table was checked on the CPU against the kernel's tiling, shared-memory and
  split-K semaphore constraints, but compilation and numerics are unverified on gfx950.
- `compile_hgemm_kernel` import paths follow AITER's current FlyDSL API (`ptrtoint`, `fx.Stream`); the runtime
  image's FlyDSL version must provide them.
- The split-K kernel zeroes its output and uses spin-waiting semaphores; behavior under CUDA-graph replay
  with the cold-cache flush should be confirmed.

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

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 10 case(s) failed the default bound, up to 0.59% of the elements over it, peak 1.86x at `k` = 2, about 0.47x at `k` = 8.
