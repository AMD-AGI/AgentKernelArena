# roiaware_pool3d, float32, 4x4x4 voxels, max pooling

RoI-aware 3D pooling of point features (`roiaware_pool3d` of OpenPCDet, shipped in MMCV as
`mmcv.ops.RoIAwarePool3d`), used by Part-A2 style second stages: each box is cut into a voxel grid in its
own rotated frame and every voxel keeps the channel-wise maximum of the features of the points inside it.
This task is the forward pass in `max` mode with `out_size = 4` (the configuration of MMCV's own unit test and
of the earlier task; Part-A2 itself uses 12 to 14 voxels per side) and 4 channels (the 3 part offsets and
the segmentation score of Part-A2). The earlier task also timed the `avg` mode (now the sibling task `f32_out4_avg`) and the backward pass; the
backward pass, `C = 3` or `6` and `out_size = 2` of its toy shapes are not carried over.

- **Scenario.** float32 `rois (R, 7) = (cx, cy, cz, dx, dy, dz, heading)` with `cz` at the bottom, `pts (P, 3)`
  and `pts_feature (P, 4)` of a single scene; output `(R, 4, 4, 4, 4)`. A point is inside a box when
  `cz <= z <= cz + dz` and, rotated by `-heading` about the box center, `|local_x| < dx/2` and
  `|local_y| < dy/2`; its voxel is the truncated quotient of the local offset by the voxel size, clamped.
  Operator semantics kept by the task: a voxel keeps only its first 127 points in index order
  (`max_pts_per_voxel = 128`, one slot being the counter); the remaining points of a crowded voxel do not
  contribute. An empty voxel is 0.
- **Comparison: `exact`.** The output is a maximum of input values (no arithmetic), so the result is
  bitwise determined; this is an intentional departure from the default bound.
- **Inputs.** `inputs.py` draws vehicle-like boxes; a quarter of the points lie inside randomly chosen boxes
  and the rest fill the scene uniformly. The kernel decides membership and voxel index in float32 and the
  reference in float64, so the generator redraws every point within 1e-3 of a box face or voxel boundary of a
  box it could be inside of; no decision depends on rounding or on a candidate's evaluation order.
- **Reference.** float64 geometry per box, a stable sort of the points by voxel gives the first 127 points of
  each voxel, a `scatter_reduce` maximum per channel; independent of the kernel. Features are the built-in
  standard normal distribution.
- **Baseline.** The HIP kernels of the earlier hip2hip task (point-to-voxel code, one thread per box
  collecting the per-voxel point lists, max pooling), vendored from this repository at the recorded revision.
  Changes: the pybind wrapper became the one-op torch library registration, which allocates the original
  scratch buffers (point mask, voxel point lists, argmax), a small kernel zeroes the voxel counters that the
  original zero-initialised buffer provided, and the max-pool kernel writes the zero of an empty voxel
  itself because outputs are poisoned instead of pre-zeroed. The average and backward kernels were dropped.
- **Workloads.** 1 to 512 boxes and 2048 to 65536 points of one scene (full LiDAR scans have about 120K
  points; second-stage RoI counts are 128 to 512); the timed span is more than 100x in `n_rois * n_points`.
  Tiny and dense (`r1_n40000`, voxels above the 127-point cap) scenes are correctness-only or limit-probing.
  Timing risk: the smallest timed case (8 x 2048) is dominated by the serial per-box scan of 2048 mask
  entries, estimated at tens of microseconds; the largest cases are slow (a single thread per box scans up
  to 65536 entries), so a timing round of them takes a few tens of milliseconds per sample.
