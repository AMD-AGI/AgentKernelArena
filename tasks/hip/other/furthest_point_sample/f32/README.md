# Furthest point sampling, float32 point clouds

Greedy furthest point sampling (FPS) of a batch of 3-D point clouds, the first stage of PointNet++
style point-cloud networks (mmcv `furthest_point_sample`, from Pointnet2.PyTorch): starting from
point 0, repeatedly select the point whose squared distance to the already selected set is largest.
Each cloud yields `n_samples = n_points // 4` int32 indices. The loop is sequential (every step
depends on the previous argmax), one thread block per cloud. Equal distances resolve to the
smallest point index.

**Scenario.** float32 `(batch, n_points, 3)` coordinates; `batch` and `n_points` vary. The old
task also measured a pairwise-distance-matrix input (`furthest_point_sampling_with_dist`); that is a
different operator interface and is not part of this scenario.

**Inputs.** Coordinates are odd integers / 128 in (-8, 8) (differences are even integers up to 2046
units, squared sums below 2^24), so every squared distance is exactly representable in float32 and independent of FMA contraction. Odd integers are used because a plain integer grid is about half exact in bfloat16 (the QA-INPUTS
check rejects more than 50%); with odd numerators only about a quarter of the values are. This keeps the float64 reference and
the float32 kernels bit-identical, which a chaotic argmax chain needs: with continuous coordinates a
last-bit difference would flip a selection and every later one. The reference breaks ties by the
smallest index, so tie order is part of the contract (exact comparison of the int32 indices, hence
the departure from the default bound).

**Reference.** Float64 greedy FPS over the whole batch with `argmax` (first maximum). The sample
count rule `max(1, n_points // 4)` is the task's `n_samples` derived axis.

**Baseline.** The in-task HIP kernel of the retired `hip2hip/furthest_point_sample` task (mmcv, from
Pointnet2.PyTorch's `sampling_gpu.cu`), registered as the arena `run` op. Changes: only the
coordinate kernel is kept; the running-minimum buffer is allocated and initialised inside the op;
ties between equal distances resolve to the smaller index (the original reduction tree kept
whichever tied thread index was lower, not the lower point index); a barrier is added after the
shared reduction slot is read, closing a race in which a fast wave could overwrite it before a slow
wave had read the winner; the thread count is computed with integer arithmetic; host-side printing
and `exit` are removed.

**Workloads.** Production-style clouds of 1K to 16K points (PointNet++ 1024/2048/4096, scene and
OpenPCDet-style 16384 inputs), batch 1 to 16; performance spans 256x in `batch * n_points`. Tiny and
odd shapes (16, 100, 1000 points; 32 clouds of 48) are correctness-only boundary cases.

**Risks for the GPU audit.** The reference runs `n_samples` sequential steps (4096 for the largest
case), a few seconds on the GPU. FPS time is dominated by the per-step block reduction, so
performance scales with `n_samples`, not with bytes. A candidate must reproduce the smallest-index
tie rule; the grid inputs make exact ties common enough that it is exercised.
