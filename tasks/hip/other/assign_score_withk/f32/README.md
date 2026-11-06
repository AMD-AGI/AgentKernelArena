# assign_score_withk, float32, K=16, M=8, O=64

The memory-efficient PAConv score assignment as shipped in MMCV (`mmcv.ops.assign_score_withk`, forward,
sum aggregation, "neighbor" kernel input): the point features were already transformed by every matrix of
the weight bank, and each output element is the score-weighted sum over the M matrices of the neighbor
feature minus the center feature, `out[b,o,n,k] = sum_m s[b,n,k,m] * (P[b,i,m,o] - C[b,c,m,o])` with
`i = knn_idx[b,n,k]` and `c = knn_idx[b,n,0]`. Neighbor indices outside `[0, n_points)` contribute zero.
This task is the forward pass in sum mode; the earlier task's average/max modes were never implemented in
its kernel and its backward pass is a different operator.

- **Comparison.** `baseline_relative` on the float32 output with `k: 4` and `max_violation_fraction: 0.002`
  (default is `k: 2`, zero violations). Each output is a sum of 8 terms that cancel (`P - C` of two
  independent normal draws, weighted by a normal score), so the baseline, which rounds `s*P` and `s*C`
  separately, and a correct candidate that forms `(P - C) * s` or sums the terms in another order, each
  carry an independent float32 rounding error of a few ulp of the largest term, while an element whose
  terms cancel has a tiny float64 value and a baseline error that can happen to be much smaller than the
  candidate's. Emulating on CPU (batches 7x250, 2x513 and 16x1024 centers, float32 sequential
  accumulation of `(P - C) * s` forward and reversed, and a float32 einsum, against the float64 reference
  and a float32 emulation of the baseline), the default bound rejects 0.8% to 1.1% of elements at peak
  3.4x to 5.0x of the bound, with no bias (bias z 0.0002 to 0.001). At `k: 4` the violation fraction is at
  most 0.07% and the peak at most 2.8x of the bound; `max_violation_fraction: 0.002` leaves a factor of
  about 3 margin and every admitted element stays within the 16x peak limit. Float32 accumulation errors
  are not correlated across elements, so no bias test is needed beyond the default. Skipped or zeroed
  neighbors and shifted outputs (the controls) are many bounds off and still rejected.
- **Inputs.** `scores`, `point_features`, `center_features` are standard normal; `knn_idx` has the center
  as entry 0, random neighbors elsewhere and about 3% out-of-range entries (negative or beyond the cloud).
  `n_points = 4 * n_centers`, like a PointNet++ set-abstraction layer that keeps a quarter of the points.
- **Reference.** float64, gathered in chunks of centers (independent of the kernel).
- **Baseline.** The forward HIP kernel of the earlier hip2hip task (one thread per output element, a loop
  over M), vendored from this repository at the recorded revision. Change: the original `atomicAdd` of each
  of the M terms into a zero-initialised output became a register accumulation and a single store (same
  terms, same order), since the output is poisoned rather than zeroed. The pybind wrapper became the
  one-op torch library registration and the unused `aggregate` argument was dropped.
- **Workloads.** The old task used tiny shapes (up to 4x256x128 centers, O up to 32); these are replaced by
  PAConv-scale shapes (1024 to 4096 centers per cloud, batches of 1 to 64). Timed cases span 2 x 1024 to
  64 x 4096 centers (128x); smaller ones, including the single 1024-center cloud (estimated 10 to 20 us, too close to the floor), are correctness-only. Memory: the largest case holds about 4.3 GB of
  features. Timing risk: the baseline gathers features with a stride of K between consecutive threads, so it
  is slower than the bandwidth bound; the smallest timed case (2 x 1024 centers, 2 Mi threads) is estimated
  at 20 to 40 us.
