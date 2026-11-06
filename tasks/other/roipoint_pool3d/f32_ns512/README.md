# roipoint_pool3d, float32, 512 sampled points per box

RoI point pooling of the PointRCNN second stage (`roipoint_pool3d` of OpenPCDet, shipped in MMCV as
`mmcv.ops.RoIPointPool3d`): every box gathers the first 512 points of its cloud that lie inside it, with
their 128 features, as the input of the RoI head. A box with fewer points repeats them cyclically; an empty
box is flagged and zero-filled. The earlier task covered 4 to 32 samples with 3 or 6 features; this task fixes
the operator's production setting of 512 samples (`num_sampled_points=512`) and 128 point features (the width of the
PointRCNN backbone output; the head may append a few extra channels, so 128 is representative, not exact).

- **Scenario.** float32 points `(B, N, 3)`, features `(B, N, 128)` and boxes `(B, M, 7)` = `(cx, cy, cz, dx,
  dy, dz, heading)` with `cz` at the bottom. A point is inside when `cz <= z <= cz + dz` and, rotated by
  `-heading` about the box center, `|local_x| < dx/2` and `|local_y| < dy/2`. Outputs: pooled `(B, M, 512,
  3 + 128)` and an int32 empty flag `(B, M)`.
- **Comparison: `exact`** for both outputs, an intentional departure from the default bound. The outputs are
  gathered float32 values and a flag; no arithmetic is performed on them, so any difference is a wrong
  point, order or padding.
- **Inputs.** `inputs.py` draws vehicle-like boxes; half of the points lie inside boxes drawn from the first
  three quarters of the box list and half are uniform background, so boxes hold more than 512 points (the
  first 512 by index are kept), a few points (repeated) or none (flag 1). The kernel decides membership in
  float32 and the reference in float64, so the generator redraws every point within 1e-3 of a face of a
  box it could be inside of; no membership decision depends on rounding or on a candidate's evaluation order.
- **Reference.** float64 geometry, membership per box chunk, `topk` of the point positions for the first
  512 inside points, cyclic repetition and zero fill; independent of the kernel.
- **Baseline.** The three HIP kernels of the earlier hip2hip task (point-to-box assignment, one thread per
  box collecting the first 512 inside points, gather), vendored from this repository at the recorded
  revision. Changes: the pybind wrapper became the one-op torch library registration, which allocates the
  scratch buffers of the original wrapper (`pts_assign` of `B * N * M` ints and the sampled indices); and,
  because outputs are poisoned instead of pre-zeroed, the kernels write the empty flag and the zeros of an
  empty box themselves. Scratch allocation inside `run` is a plain allocation and is capture-safe.
- **Workloads.** 1 to 8 clouds with 2048 to 16384 points and 16 to 512 boxes (PointRCNN samples 16384 points
  per cloud and keeps 128 to 512 RoIs); the timed span is more than 100x in `batch * n_points * n_boxes`.
  Tiny scenes are correctness-only. Timing risk: the smallest timed case (1 x 2048 x 16) is dominated by the
  serial per-box scan of up to 2048 assignment flags and is estimated at tens of microseconds, but may be
  close to the 10 us floor. The largest case writes about 134 MB of pooled output (8 x 128 x 512 x 131
  floats).
