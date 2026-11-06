# points_in_boxes (part mode), float32

Which oriented 3D box contains each LiDAR point (MMCV `points_in_boxes_part`, from OpenPCDet): the
output is the index of the first containing box along the box axis, or -1. Siblings in this directory:
`f32_all` returns a flag for every point-box pair (`points_in_boxes_all`). The earlier task also
timed a "standard" and a "rotated" box variant; here the box heading is always random, which covers both.

- **Scenario.** float32 points and boxes; boxes are `(cx, cy, cz, dx, dy, dz, heading)` with `cz` at
  the bottom. A point is inside when `cz <= z <= cz + dz` and, rotated by `-heading` about the box center,
  `|local_x| < dx/2` and `|local_y| < dy/2` (x and y faces are outside). The task covers the
  rotated-box case with a heading uniform in [-pi, pi), which includes the axis-aligned case.
  Var axes are the batch size (1 to 8), points per cloud (up to 262144: a single scan has about 120K points, accumulated multi-sweep clouds
  reach 300K) and boxes per cloud (up to 256, padded ground-truth or proposal lists of detection pipelines).
- **Inputs.** `inputs.py` draws vehicle-like boxes and uniformly distributed points over the same volume
  (about 5% of the points fall inside some box). The kernel decides in float32 and the reference in float64, so
  the generator redraws every point within 1e-3 of a face of a box it could be inside of; no
  membership decision then depends on rounding or on the order of operations in a candidate.
- **Comparison: `exact`.** The output is an integer tensor and every flag or index is a hard decision;
  this is an intentional departure from the default bound.
- **Reference.** float64 geometry, chunked over points, independent of the kernel.
- **Baseline.** The HIP kernel of the earlier hip2hip task (one thread per point scanning the boxes),
  vendored from this repository at the recorded revision; it derives from the OpenPCDet
  `roiaware_pool3d` kernel as shipped in MMCV. Changes: the pybind wrapper became the one-op torch
  library registration, and, because outputs are poisoned instead of pre-filled, the kernel now writes
  the default value itself.
- **Workloads.** 65K to 262K points per cloud, 1 to 8 clouds, 64 to 256 boxes: the timed span is
  128x in `batch * n_points * n_boxes` (4.2M to 537M point-box pairs). The work is compute-bound (float32
  sin/cos and compares per point-box pair, one thread per point), so only the larger scenes clearly clear
  the 10 us floor; the smallest timed case (1 x 65536 x 64) is estimated at about 10 us on MI355X and may
  fall just under it, which is the main risk of the task. Smaller scenes are correctness-only. The
  largest case is 8 x 262144 x 256 pairs.
