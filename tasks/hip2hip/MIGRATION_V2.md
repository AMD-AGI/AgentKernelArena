# hip2hip v2 migration inventory

This inventory describes the 32 migrated tasks. The task's own `config.yaml`,
`README.md`, protected runner, and complete `workload.json` define its contract.
Each task has explicit task / baseline / candidate actions using `arena-eval-v1`.
No task imports the Arena framework or an agent. This is a CPU migration audit,
not evidence that the tasks passed GPU validation.

The migration preserved the original Python module and functional reference,
inputs, seeds, tolerances, warmups, repetitions, and state-reset callbacks.
Post-migration GPU validator repairs are explicitly listed below. The new
runner binds the declared model class and compiles the original nested source
path, so filename guessing and scratch basename copies cannot change the
contract. A missing final candidate fails; it never selects a baseline.

The 22 extension tasks use their separately provided HIP reference as the
performance baseline. The 10 native tasks freeze the initial HIP candidate.
For matrix multiplication and MLA, the original device implementation was
extracted verbatim into `source/kernel.hpp`, included by the protected host
program. The migration preserved host references, numerical gates and native replay
drivers; the later matrix coverage repair below supplements that initial state.

For the 8 native extension tasks, `scripts/reference_checks.py` adds analytical
CPU known answers and zero-output negative controls. Correctness also covers
backward gradients, annular radius queries, transposed/self KNN, rotated boxes,
and the direct native pooling calls used in timing. Existing comparisons are
retained, including the ROI max-pool sum tolerance and ROI-point sum tolerance;
these gates are intentionally not replaced with a new numerical policy.

Two inherited details should remain explicit during later task maintenance:

- Assign-score correctness uses `cpu_assign_score_withk_forward_vectorized`.
  The old unused scalar helper adds an extra point feature and is not the
  accepted reference. The active vectorized formula and its gradients are
  tested against independent known values.
- Ball-query's native implementation includes coincident points even for an
  annular query, while the existing CPU reference specifies `[min, max)`.
  The existing workload generates independent random points and centers.
  Adding exact-coincidence cases requires resolving this semantic difference;
  the migration does not weaken the existing reference to accommodate it.

Shared integration requirement: materialize the canonical Python benchmark
helper beside each importing harness (`eval_tools/_aka_benchmark.py` for the
extension tasks, `scripts/_aka_benchmark.py` for native Python harnesses), and
`hip_graph_benchmark.hpp` beside native benchmark drivers where present. Helpers
are supplied by the framework; task authors must not hand-edit generated files.
The new runner may call another protected module, so helper discovery cannot
inspect only the first command file or dispatch from a legacy `task_type`.

## Repairs found by real GPU validation

Job 139005 on MI355X produced finalized reports of FAIL for HIP GELU and matrix
multiplication. GELU aliased caller-owned input although its public reference is
out-of-place. Its provided HIP baseline and initial candidate now both use
separate contiguous output storage, and checks enforce input preservation and
non-aliasing. This intentionally adds the same copy/allocation work to both
roles. The timed graph output is checked against the protected reference before
and after output poisoning and exact replay, outside timed samples.

Matrix multiplication retains its constant-input host check and now also checks
all output cells for the existing nonuniform benchmark inputs in correctness
and exact replay. Its original replay tolerance, five shapes and timing settings
are preserved. CPU known answers and negative controls cover this added reference.
These changes require fresh finalized GPU reports; successful initial actions
or CPU regressions alone do not establish task-validator PASS.

| Task | Cases | Initial candidate | Baseline | Editable file |
|---|---:|---|---|---|
| [gpumode/CrossEntropyLossLabelSmoothing](gpumode/CrossEntropyLossLabelSmoothing/config.yaml) | 5 | implemented | provided / hip | `hip/hip_12501_CrossEntropyLossLabelSmoothing.hip` |
| [gpumode/Feedforward](gpumode/Feedforward/config.yaml) | 4 | implemented | provided / hip | `hip/hip_10024_Feedforward.hip` |
| [gpumode/FusedLeakyReLU](gpumode/FusedLeakyReLU/config.yaml) | 5 | implemented | provided / hip | `hip/hip_10190_FusedLeakyReLU.hip` |
| [gpumode/GELU](gpumode/GELU/config.yaml) | 11 | implemented | provided / hip | `hip/hip_14539_GELU.hip` |
| [gpumode/GateGRUSelectionLayer](gpumode/GateGRUSelectionLayer/config.yaml) | 5 | implemented | provided / hip | `hip/hip_5334_GateGRUSelectionLayer.hip` |
| [gpumode/InnerProd](gpumode/InnerProd/config.yaml) | 4 | implemented | provided / hip | `hip/hip_11709_InnerProd.hip` |
| [gpumode/ItemQueryAttention](gpumode/ItemQueryAttention/config.yaml) | 8 | implemented | provided / hip | `hip/hip_102_ItemQueryAttention.hip` |
| [gpumode/KDLoss](gpumode/KDLoss/config.yaml) | 4 | implemented | provided / hip | `hip/hip_14007_KDLoss.hip` |
| [gpumode/MLP_model](gpumode/MLP_model/config.yaml) | 5 | implemented | provided / hip | `hip/hip_1178_MLP_model.hip` |
| [gpumode/MaskedLanguageModel](gpumode/MaskedLanguageModel/config.yaml) | 4 | implemented | provided / hip | `hip/hip_8325_MaskedLanguageModel.hip` |
| [gpumode/MultiHeadAttention](gpumode/MultiHeadAttention/config.yaml) | 3 | implemented | provided / hip | `hip/hip_10456_MultiHeadAttention.hip` |
| [gpumode/NormalAttention_dot](gpumode/NormalAttention_dot/config.yaml) | 5 | implemented | provided / hip | `hip/hip_1001_NormalAttention_dot.hip` |
| [gpumode/NormalAttention_embedded_gaussian](gpumode/NormalAttention_embedded_gaussian/config.yaml) | 5 | implemented | provided / hip | `hip/hip_1003_NormalAttention_embedded_gaussian.hip` |
| [gpumode/PositionWiseFeedForward](gpumode/PositionWiseFeedForward/config.yaml) | 4 | implemented | provided / hip | `hip/hip_14044_PositionWiseFeedForward.hip` |
| [gpumode/SiLU](gpumode/SiLU/config.yaml) | 11 | implemented | provided / hip | `hip/hip_16636_SiLU.hip` |
| [gpumode/Sigmoid](gpumode/Sigmoid/config.yaml) | 11 | implemented | provided / hip | `hip/hip_11184_Sigmoid.hip` |
| [gpumode/SimpleMatmulModule](gpumode/SimpleMatmulModule/config.yaml) | 11 | implemented | provided / hip | `hip/hip_3267_SimpleMatmulModule.hip` |
| [gpumode/SoftmaxModule](gpumode/SoftmaxModule/config.yaml) | 5 | implemented | provided / hip | `hip/hip_10082_SoftmaxModule.hip` |
| [gpumode/TanH](gpumode/TanH/config.yaml) | 11 | implemented | provided / hip | `hip/hip_11178_TanH.hip` |
| [gpumode/TransformerFFNLayer](gpumode/TransformerFFNLayer/config.yaml) | 4 | implemented | provided / hip | `hip/hip_14069_TransformerFFNLayer.hip` |
| [gpumode/Transpose](gpumode/Transpose/config.yaml) | 5 | implemented | provided / hip | `hip/hip_1067_Transpose.hip` |
| [gpumode/layer_normalization](gpumode/layer_normalization/config.yaml) | 6 | implemented | provided / hip | `hip/hip_11754_layer_normalization.hip` |
| [others/assign_score_withk](others/assign_score_withk/config.yaml) | 10 | implemented | initial_candidate / hip | `src/assign_score_withk_cuda.hip` |
| [others/ball_query](others/ball_query/config.yaml) | 10 | implemented | initial_candidate / hip | `src/ball_query_cuda.hip` |
| [others/furthest_point_sample](others/furthest_point_sample/config.yaml) | 10 | implemented | initial_candidate / hip | `src/furthest_point_sample_cuda.hip` |
| [others/knn](others/knn/config.yaml) | 15 | implemented | initial_candidate / hip | `src/knn_cuda.hip` |
| [others/matrix_multiplication](others/matrix_multiplication/config.yaml) | 5 | implemented | initial_candidate / hip | `source/kernel.hpp` |
| [others/mla_decode](others/mla_decode/config.yaml) | 5 | implemented | initial_candidate / hip | `source/kernel.hpp` |
| [others/points_in_boxes](others/points_in_boxes/config.yaml) | 20 | implemented | initial_candidate / hip | `src/points_in_boxes_cuda.hip` |
| [others/roiaware_pool3d](others/roiaware_pool3d/config.yaml) | 10 | implemented | initial_candidate / hip | `src/roiaware_pool3d_kernel.hip` |
| [others/roipoint_pool3d](others/roipoint_pool3d/config.yaml) | 5 | implemented | initial_candidate / hip | `src/roipoint_pool3d_kernel.hip` |
| [others/three_nn](others/three_nn/config.yaml) | 5 | implemented | initial_candidate / hip | `src/three_nn_cuda.hip` |
