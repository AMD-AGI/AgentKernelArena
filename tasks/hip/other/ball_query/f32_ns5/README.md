# ball_query, float32 points, 5 neighbors

PointNet++-style grouping as shipped in MMCV (`mmcv.ops.ball_query`, originally from Pointnet2.PyTorch):
each center collects the first `nsample = 5` points, in index order, inside the ball
`min_radius^2 <= d2 < max_radius^2` (or at distance exactly 0). Unfilled slots repeat the first match, and
a center with no match yields index 0. The task is the fixed-radius forward pass (`min_radius = 0`,
`max_radius = 0.25`); dilated shells (`min_radius > 0`) and a wide radius appear as scalar variants of
some cases.

- **Comparison: `exact`.** The output is an integer index tensor; any difference is a wrong neighbor.
  This is an intentional departure from the default bound.
- **Inputs.** Coordinates are a clipped standard-normal cloud snapped to odd multiples of 1/4096 (finer than bfloat16 resolves, so the inputs are not bf16-exact) and the centers are a
  random subset of the points (what farthest-point sampling yields; every center has a zero-distance match
  and duplicates occur). On that grid every squared distance that can lie inside a ball of radius <= 1 is exactly representable in float32 (component differences are m/4096 with |m| <= 4096, squares need <= 24 bits, partial sums stay below 1; larger distances may round but remain far above 1), so the
  float32 kernel and the float64 reference agree on every in/out-of-ball decision regardless of evaluation
  order or FMA contraction. Without this, near-boundary points would make an exact index comparison depend
  on rounding. `n_centers = ceil(n_points / 4)`.
- **Reference.** float64 distances, chunked over centers; `topk` on masked point positions gives the first
  matches in index order. `NSAMPLE` is fixed at 5 in `reference.py` and `task.yaml`.
- **Baseline.** The HIP kernel of the earlier hip2hip task (one thread per center, scan until
  `nsample` matches), vendored from this repository at the recorded revision. Changes: the pybind wrapper
  became the one-op torch library registration; because the output is poisoned rather than pre-zeroed,
  a center without any match now writes its zeros in the kernel (same result as the old zero-initialised
  buffer).
- **Workloads.** Batches of 1 to 16 clouds with 2048 to 65536 points (the old task used 256 to 16384
  points with 5 to 16 neighbors; 16384 matches KITTI-scale clouds). The timed span is 128x in
  `batch * n_points` (2 x 4096 to 16 x 65536). Smaller clouds, including the single 2048-point cloud, are
  correctness-only. Timing risk: with radius 0.25 in a unit-normal
  cloud a thread scans on the order of thousands of points before finding 5 matches; the smallest timed
  case (2 x 4096, 2 x 1024 threads) rests on an estimate of a scan latency of a few tens of microseconds and
  may still fall near 10 us on MI355X.
