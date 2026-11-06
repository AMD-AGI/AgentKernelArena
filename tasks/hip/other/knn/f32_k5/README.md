# k-nearest neighbours (k = 5), float32 point clouds

For every query centre, the 5 nearest points of its cloud by squared Euclidean distance, nearest
first, returned as int32 indices and float32 squared distances (mmcv `knn`, from PAConv's
`knnquery_heap`). One thread per query scans the whole cloud with a size-k max-heap. The old task's
`transposed` layout and `self_query` variants only changed host-side tensor preparation and are not
separate scenarios; the old harness's `(B, k, M)` index transpose is dropped, and the op's native
`(B, M, k)` layout is used for both outputs.

**Scenario.** float32 `(batch, n_points, 3)` clouds with `n_query = n_points // 4` centres each,
`k = 5` (the mmcv/PointNet++ default). Centres are points of the cloud, as when FPS output feeds knn
(a centre's nearest neighbour is itself at distance 0).

**Inputs and comparison.** Coordinates are odd integers / 128 in (-8, 8) (odd, so only about a quarter of the values are exact in
bfloat16, which the QA-INPUTS check requires to be under half; differences are at most 2046/128 and
squared distances stay below 2^24 / 2^14), so every squared distance is
exactly representable in float32 and independent of FMA contraction; indices and distances are then
compared exactly (`exact` instead of the default bound). Equal distances resolve to the smaller
point index, which fixes the neighbour set and order even with ties; a neighbour-selection kernel
has no meaningful numeric tolerance, because a wrong neighbour is a wrong answer.

**Reference.** Float64 distances ordered by the exact integer key `round(d2 * 2^14) * n_points +
index` and `topk`, in chunks to bound memory. The key is exact only for the grid inputs; `k = 5`
repeats the task's constant axis (the reference sees no axes).

**Baseline.** The in-task HIP kernel of the retired `hip2hip/knn` task (mmcv, PAConv heap kernel),
registered as the arena `run` op. Changes: the heap and the final heap sort order by `(distance,
index)` so equal distances are deterministic (the original left their order to the heap shape);
host-side printing and `exit` are removed.

**Workloads.** Production-style clouds of 2K to 16K points, batch 1 to 16; performance spans 128x
in `batch * n_points`. Tiny and odd shapes (16, 100, 1000 points; 16 clouds of 64) are
correctness-only boundary cases.

**Risks for the GPU audit.** The smallest performance case (one 2048-point cloud, 512 query
threads) is an estimated tens of microseconds of serial scan per thread; demote it if the baseline
measures under 10 us. The largest case evaluates 1.1e9 point-pair distances in the float64
reference, done in chunks of about 16M pairs.
