# three_nn, float32 points

Three nearest neighbors of every target point among a source set, as used for feature interpolation in
PointNet++ (`mmcv.ops.three_nn`, originally from Pointnet2.PyTorch). For each of `n_target` points the
task returns the distances to and the indices of its 3 nearest source points of the same batch element,
nearest first. `n_source` is a variable axis (at least 3). Most workload cases use `n_source = ceil(n_target / 4)` (a
feature-propagation layer interpolates a sparse set onto one four times denser); the source-heavy and
source-poor shapes of the earlier task are kept as well. The earlier task wrapped the kernel and took `sqrt` of its
squared distances outside the kernel; here the returned distances are Euclidean, which is the operator's
public result.

- **Comparison.** `dist` uses the default `baseline_relative` bound. `idx` is `exact`, an intentional
  departure from the default: indices are hard decisions. Points at exactly the same distance are ordered
  by lowest index (the baseline's strict `<` insertion keeps the earlier index first); this tie rule is part
  of the task description.
- **Inputs.** Coordinates are uniform odd multiples of 1/1024 in (1, 2): each has 11 significant bits, so no
  value is exactly representable in bfloat16 or float16 (a coarse grid such as multiples of 1/256 around zero
  is, and fails the QA-INPUTS check). Coordinate differences are even multiples of 1/1024 below 1, so every
  squared distance is an integer multiple of 2^-20 below 2^22 units and exactly representable in float32. The
  float32 kernel and the float64 reference then agree on every ranking and tie regardless of evaluation order
  or FMA contraction. Without it, near-ties would make an exact index comparison depend on rounding. Distinct
  points at equal distance (ties) occur often on this grid.
- **Reference.** float64 squared distances per target chunk, stable sort for the tie rule, square root.
- **Baseline.** The HIP kernel of the earlier hip2hip task (one thread per target scanning all sources and
  keeping a sorted triple), vendored from this repository at the recorded revision; changes are the
  one-op torch library registration and the in-kernel square root.
- **Workloads.** 1 to 8 clouds with 2048 to 32768 targets (512 to 8192 sources), plus two timed source-heavy
  shapes of the earlier task (512 targets with 4096 sources, 1024 with 2048); the timed span is far above 100x in
  `batch * n_target * n_source`. Small clouds are correctness-only. Timing risk: the baseline cost is a serial scan of
  the sources per thread; the smallest timed case (1 x 2048 targets, 512 sources) is estimated at about
  10 to 20 us on MI355X and may fall below the 10 us floor.
