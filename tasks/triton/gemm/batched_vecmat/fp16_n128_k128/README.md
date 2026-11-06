# Batched vector-matrix product, float16, n=128 k=128

Operator: `out[m, n] = sum_k a[m, k] * b[m, n, k]` for `a` [m, 128] and `b` [m, 128, 128], all float16.
Each row of `a` is contracted with its own 128 x 128 matrix (stored n-major with k contiguous), so the
operator streams `b` once (32 KB per row) and is memory-bound. The 128-term reduction accumulates in float32
and the result is rounded to float16.

**Origin.** The rocmbench `test_batched_vecmat` task (the former
`triton2triton/rocmbench/easy/test_batched_vecmat`), whose kernel comes from the `test_vecmat` unit test of
the Triton project. That task checked one float32 case (M = N = K = 128, small integers) and benchmarked a
float32/float16 sweep of five (M, N, K) shapes with M <= 1024 and three block configurations. This scenario
fixes float16 and N = K = 128 (the shape of the original test) and varies M. Of the two benchmarked dtypes it
takes float16, the case the old task's repair and its reference were written for (float16 products with a
float32 reduction); float32 inputs, and other N and K, would be separate scenarios. Partial n and k tiles of
the old 17 x 19 x 35 control cannot occur with N = K = 128 fixed; partial m tiles are covered by the M = 3,
100 and 1000 cases.

**Workloads.** The original N = K = 128 shape (M = 128) moves 4 MB and runs below the 10 us timing floor,
so the performance cases scale the batch of products: M = 1280 to 131072 (42 MB to 4.3 GB of `b`, 102x span;
an estimated 13 us to 1 ms at the HBM bandwidth). M = 1, 3, 100, 128 (the original test shape) and 1000 are
correctness-only. The declared maximum M = 131072 is where the baseline's int32 offsets reach 2^31 - 1.
Inputs are normal(0, 1), as in the original benchmark.

**Reference.** float64 product and sum, in row chunks to bound device memory, compared with the default
`baseline_relative` bound. A CPU simulation (M = 8192) shows the bound accepts float32 products, other
reduction orders and the correctly rounded result, and rejects a float16 accumulator (the kernel before its
repair, 0.3% of elements out of bound), bfloat16 products or output, the 16-ulp shift and the fp8 output
rounding.

**Baseline.** The rocmbench `batched_vecmat` kernel, copied unchanged from the old task: one program per
16-row x 32-column output tile loops over k in blocks of 64, multiplies the broadcast `a` tile with the
[32, 16, 64] `b` tile in float16 and reduces over k in float32. It includes this repository's later repair of
the rocmbench kernel (a float32 instead of float16 accumulator, a ceil-divided k loop and masks for partial
tiles); the provenance revision is the commit of that repair, the version copied.
`run()` is the old timed wrapper (block 16 x 32 x 64, Triton's default warps and stages) writing into the
preallocated output instead of a zero-filled one. `baseline/LICENSE` is the MIT license of the Triton
project and the rocmbench suite.

**GPU risks (not yet run).** The [32, 16, 64] three-dimensional tile is register-heavy (128 float16 values
per thread before the float32 reduction) and the grid has one program per 16 rows, so the baseline may stay
well below the HBM bandwidth; the smallest performance case (M = 1280, about 13 us at full bandwidth) is
then safely above the floor, but if the kernel turns out near the bandwidth limit it is the case closest to
10 us.
