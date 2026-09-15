# torch2hip v2 migration inventory

This inventory describes the 57 migrated tasks. The task's own `config.yaml`,
`README.md`, protected runner, and complete `workload.json` define its contract.
Each task has explicit task / baseline / candidate actions using `arena-eval-v1`.
No task imports the Arena framework or an agent. This is a CPU migration audit,
not evidence that the tasks passed GPU validation.

The extension tasks preserve their original Python module and functional
reference, inputs, HIP files, correctness and performance implementations,
seeds, tolerances, warmups, repetitions, and state-reset callbacks. The new
runner binds the declared model class and compiles the original nested source
path, so filename guessing and scratch basename copies cannot change the
contract. A missing final candidate fails; it never selects a baseline.

All 57 HIP targets were empty in the original tasks. Their explicit initial
state is `unimplemented`, and the provided performance baseline is the PyTorch
module. Baseline correctness cross-checks the separately written functional
form. Final correctness always invokes the compiled HIP extension. MiniGPTBlock
explicitly names its full block class; its first class, NewGELU, is a helper.

Shared integration requirement: materialize the canonical Python benchmark
helper beside each importing harness (`eval_tools/_aka_benchmark.py` for the
extension tasks, `scripts/_aka_benchmark.py` for native Python harnesses), and
`hip_graph_benchmark.hpp` beside native benchmark drivers where present. Helpers
are supplied by the framework; task authors must not hand-edit generated files.
The new runner may call another protected module, so helper discovery cannot
inspect only the first command file or dispatch from a legacy `task_type`.

| Task | Cases | Initial candidate | Baseline | Editable file |
|---|---:|---|---|---|
| [gpumode/1001_NormalAttention_dot](gpumode/1001_NormalAttention_dot/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_1001_NormalAttention_dot.hip` |
| [gpumode/10024_Feedforward](gpumode/10024_Feedforward/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_10024_Feedforward.hip` |
| [gpumode/1003_NormalAttention_embedded_gaussian](gpumode/1003_NormalAttention_embedded_gaussian/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_1003_NormalAttention_embedded_gaussian.hip` |
| [gpumode/10082_SoftmaxModule](gpumode/10082_SoftmaxModule/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_10082_SoftmaxModule.hip` |
| [gpumode/10099_Gather](gpumode/10099_Gather/config.yaml) | 4 | unimplemented | provided / pytorch | `hip/hip_10099_Gather.hip` |
| [gpumode/10190_FusedLeakyReLU](gpumode/10190_FusedLeakyReLU/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_10190_FusedLeakyReLU.hip` |
| [gpumode/102_ItemQueryAttention](gpumode/102_ItemQueryAttention/config.yaml) | 8 | unimplemented | provided / pytorch | `hip/hip_102_ItemQueryAttention.hip` |
| [gpumode/10456_MultiHeadAttention](gpumode/10456_MultiHeadAttention/config.yaml) | 3 | unimplemented | provided / pytorch | `hip/hip_10456_MultiHeadAttention.hip` |
| [gpumode/1067_Transpose](gpumode/1067_Transpose/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_1067_Transpose.hip` |
| [gpumode/11122_PositionEmbedder](gpumode/11122_PositionEmbedder/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_11122_PositionEmbedder.hip` |
| [gpumode/11178_TanH](gpumode/11178_TanH/config.yaml) | 11 | unimplemented | provided / pytorch | `hip/hip_11178_TanH.hip` |
| [gpumode/11184_Sigmoid](gpumode/11184_Sigmoid/config.yaml) | 11 | unimplemented | provided / pytorch | `hip/hip_11184_Sigmoid.hip` |
| [gpumode/11709_InnerProd](gpumode/11709_InnerProd/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_11709_InnerProd.hip` |
| [gpumode/11754_layer_normalization](gpumode/11754_layer_normalization/config.yaml) | 6 | unimplemented | provided / pytorch | `hip/hip_11754_layer_normalization.hip` |
| [gpumode/1178_MLP_model](gpumode/1178_MLP_model/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_1178_MLP_model.hip` |
| [gpumode/12501_CrossEntropyLossLabelSmoothing](gpumode/12501_CrossEntropyLossLabelSmoothing/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_12501_CrossEntropyLossLabelSmoothing.hip` |
| [gpumode/14007_KDLoss](gpumode/14007_KDLoss/config.yaml) | 4 | unimplemented | provided / pytorch | `hip/hip_14007_KDLoss.hip` |
| [gpumode/14044_PositionWiseFeedForward](gpumode/14044_PositionWiseFeedForward/config.yaml) | 10 | unimplemented | provided / pytorch | `hip/hip_14044_PositionWiseFeedForward.hip` |
| [gpumode/14069_TransformerFFNLayer](gpumode/14069_TransformerFFNLayer/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_14069_TransformerFFNLayer.hip` |
| [gpumode/14539_GELU](gpumode/14539_GELU/config.yaml) | 11 | unimplemented | provided / pytorch | `hip/hip_14539_GELU.hip` |
| [gpumode/16636_SiLU](gpumode/16636_SiLU/config.yaml) | 11 | unimplemented | provided / pytorch | `hip/hip_16636_SiLU.hip` |
| [gpumode/3267_SimpleMatmulModule](gpumode/3267_SimpleMatmulModule/config.yaml) | 11 | unimplemented | provided / pytorch | `hip/hip_3267_SimpleMatmulModule.hip` |
| [gpumode/5334_GateGRUSelectionLayer](gpumode/5334_GateGRUSelectionLayer/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_5334_GateGRUSelectionLayer.hip` |
| [gpumode/8325_MaskedLanguageModel](gpumode/8325_MaskedLanguageModel/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_8325_MaskedLanguageModel.hip` |
| [kernelbench/level1/l1n1_Square_matrix_multiplication_](kernelbench/level1/l1n1_Square_matrix_multiplication_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n1_Square_matrix_multiplication_.hip` |
| [kernelbench/level1/l1n23_Softmax](kernelbench/level1/l1n23_Softmax/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n23_Softmax.hip` |
| [kernelbench/level1/l1n26_GELU_](kernelbench/level1/l1n26_GELU_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n26_GELU_.hip` |
| [kernelbench/level1/l1n2_Standard_matrix_multiplication_](kernelbench/level1/l1n2_Standard_matrix_multiplication_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n2_Standard_matrix_multiplication_.hip` |
| [kernelbench/level1/l1n36_RMSNorm_](kernelbench/level1/l1n36_RMSNorm_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n36_RMSNorm_.hip` |
| [kernelbench/level1/l1n3_Batched_matrix_multiplication](kernelbench/level1/l1n3_Batched_matrix_multiplication/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n3_Batched_matrix_multiplication.hip` |
| [kernelbench/level1/l1n40_LayerNorm](kernelbench/level1/l1n40_LayerNorm/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n40_LayerNorm.hip` |
| [kernelbench/level1/l1n42_Max_Pooling_2D](kernelbench/level1/l1n42_Max_Pooling_2D/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n42_Max_Pooling_2D.hip` |
| [kernelbench/level1/l1n47_Sum_reduction_over_a_dimension](kernelbench/level1/l1n47_Sum_reduction_over_a_dimension/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n47_Sum_reduction_over_a_dimension.hip` |
| [kernelbench/level1/l1n4_Matrix_vector_multiplication_](kernelbench/level1/l1n4_Matrix_vector_multiplication_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n4_Matrix_vector_multiplication_.hip` |
| [kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel](kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n63_conv_standard_2D__square_input__square_kernel.hip` |
| [kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel](kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n82_conv_depthwise_2D_square_input_square_kernel.hip` |
| [kernelbench/level1/l1n8_Matmul_with_irregular_shapes_](kernelbench/level1/l1n8_Matmul_with_irregular_shapes_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n8_Matmul_with_irregular_shapes_.hip` |
| [kernelbench/level1/l1n95_CrossEntropyLoss](kernelbench/level1/l1n95_CrossEntropyLoss/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n95_CrossEntropyLoss.hip` |
| [kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_](kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l1n9_Tall_skinny_matrix_multiplication_.hip` |
| [kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide](kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n17_Conv2d_InstanceNorm_Divide.hip` |
| [kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm](kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n37_Matmul_Swish_Sum_GroupNorm.hip` |
| [kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd](kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n40_Matmul_Scaling_ResidualAdd.hip` |
| [kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool](kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool.hip` |
| [kernelbench/level2/l2n52_Conv2d_Activation_BatchNorm](kernelbench/level2/l2n52_Conv2d_Activation_BatchNorm/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n52_Conv2d_Activation_BatchNorm.hip` |
| [kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale](kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n55_Matmul_MaxPool_Sum_Scale.hip` |
| [kernelbench/level2/l2n59_Matmul_Swish_Scaling](kernelbench/level2/l2n59_Matmul_Swish_Scaling/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n59_Matmul_Swish_Scaling.hip` |
| [kernelbench/level2/l2n66_Matmul_Dropout_Softmax](kernelbench/level2/l2n66_Matmul_Dropout_Softmax/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n66_Matmul_Dropout_Softmax.hip` |
| [kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool](kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n6_Conv3d_Softmax_MaxPool_MaxPool.hip` |
| [kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling](kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n73_Conv2d_BatchNorm_Scaling.hip` |
| [kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max](kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max.hip` |
| [kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp](kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.hip` |
| [kernelbench/level2/l2n86_Matmul_Divide_GELU](kernelbench/level2/l2n86_Matmul_Divide_GELU/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n86_Matmul_Divide_GELU.hip` |
| [kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max](kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n98_Matmul_AvgPool_GELU_Scale_Max.hip` |
| [kernelbench/level2/l2n99_Matmul_GELU_Softmax](kernelbench/level2/l2n99_Matmul_GELU_Softmax/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l2n99_Matmul_GELU_Softmax.hip` |
| [kernelbench/level3/l3n31_VisionAttention](kernelbench/level3/l3n31_VisionAttention/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l3n31_VisionAttention.hip` |
| [kernelbench/level3/l3n43_MinGPTCausalAttention](kernelbench/level3/l3n43_MinGPTCausalAttention/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l3n43_MinGPTCausalAttention.hip` |
| [kernelbench/level3/l3n44_MiniGPTBlock](kernelbench/level3/l3n44_MiniGPTBlock/config.yaml) | 5 | unimplemented | provided / pytorch | `hip/hip_l3n44_MiniGPTBlock.hip` |
