// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

// Grid-stride loop fused final kernel: out = (1 - update) * x1 + update * proposal
__global__ void fused_final_kernel(
    const float* __restrict__ update,  // [N*D]
    const float* __restrict__ x1,      // [N*D]
    const float* __restrict__ proposal,// [N*D]
    float* __restrict__ out,           // [N*D]
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel; idx += (int64_t)blockDim.x * gridDim.x) {
        float u = update[idx];
        float x = x1[idx];
        float p = proposal[idx];
        out[idx] = (1.0f - u) * x + u * p;
    }
}

static inline void check_inputs(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& reset_weight,
    const torch::Tensor& reset_bias,
    const torch::Tensor& update_weight,
    const torch::Tensor& update_bias,
    const torch::Tensor& proposal_weight,
    const torch::Tensor& proposal_bias)
{
    TORCH_CHECK(x1.is_cuda(), "x1 must be on GPU (HIP)");
    TORCH_CHECK(x2.is_cuda(), "x2 must be on GPU (HIP)");
    TORCH_CHECK(x1.scalar_type() == at::kFloat, "x1 must be float32");
    TORCH_CHECK(x2.scalar_type() == at::kFloat, "x2 must be float32");
    TORCH_CHECK(x1.sizes().size() == 4, "x1 must be 4D [B0,B1,B2,D]");
    TORCH_CHECK(x2.sizes().size() == 4, "x2 must be 4D [B0,B1,B2,D]");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");

    const auto D = x1.size(-1);

    auto check_gate = [&](const torch::Tensor& W, const torch::Tensor& b, const char* name){
        TORCH_CHECK(W.is_cuda() && b.is_cuda(), name, " params must be on GPU (HIP)");
        TORCH_CHECK(W.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, name, " params must be float32");
        TORCH_CHECK(W.dim() == 2, name, " weight must be 2D [D, 2D]");
        TORCH_CHECK(W.size(0) == D && W.size(1) == 2*D, name, " weight must be [D, 2D]");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == D, name, " bias must be [D]");
    };

    check_gate(reset_weight, reset_bias, "reset");
    check_gate(update_weight, update_bias, "update");
    check_gate(proposal_weight, proposal_bias, "proposal");
}

// Forward function matching the requested signature
torch::Tensor forward(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor reset_weight,
    torch::Tensor reset_bias,
    torch::Tensor update_weight,
    torch::Tensor update_bias,
    torch::Tensor proposal_weight,
    torch::Tensor proposal_bias)
{
    check_inputs(x1, x2, reset_weight, reset_bias, update_weight, update_bias, proposal_weight, proposal_bias);

    // Ensure contiguous tensors
    x1 = x1.contiguous();
    x2 = x2.contiguous();
    reset_weight = reset_weight.contiguous();
    reset_bias = reset_bias.contiguous();
    update_weight = update_weight.contiguous();
    update_bias = update_bias.contiguous();
    proposal_weight = proposal_weight.contiguous();
    proposal_bias = proposal_bias.contiguous();

    auto sizes = x1.sizes();
    const int64_t B0 = sizes[0];
    const int64_t B1 = sizes[1];
    const int64_t B2 = sizes[2];
    const int64_t D  = sizes[3];
    const int64_t N  = B0 * B1 * B2; // batch items

    auto options = x1.options();

    // Flatten to [N, D]
    auto x1_2d = x1.view({N, D});
    auto x2_2d = x2.view({N, D});

    // Helper to compute gate: y = bias + x1*W_l^T + x2*W_r^T, then activation
    auto compute_gate = [&](const torch::Tensor& W, const torch::Tensor& b, bool use_reset_x1, const torch::Tensor& reset_tensor, bool tanh_act) {
        // Split weights along input dimension: [D, 2D] -> ([D, D], [D, D]) columns
        auto W_l = W.narrow(/*dim=*/1, /*start=*/0, /*length=*/D);      // [D, D]
        auto W_r = W.narrow(/*dim=*/1, /*start=*/D, /*length=*/D);      // [D, D]

        // Prepare left input: either x1 or reset * x1
        torch::Tensor left_in;
        if (use_reset_x1) {
            left_in = reset_tensor.mul(x1_2d); // [N, D]
        } else {
            left_in = x1_2d;
        }

        // y = b + left_in * W_l^T
        auto y = at::addmm(b, left_in, W_l.t()); // [N, D]
        // y += x2 * W_r^T
        y.addmm_(x2_2d, W_r.t());

        // activation
        if (tanh_act) return at::tanh(y);
        return at::sigmoid(y);
    };

    // 1) reset = sigmoid(Linear([x1, x2]))
    auto reset = compute_gate(reset_weight, reset_bias, /*use_reset_x1=*/false, /*reset_tensor=*/x1_2d, /*tanh_act=*/false);

    // 2) update = sigmoid(Linear([x1, x2]))
    auto update = compute_gate(update_weight, update_bias, /*use_reset_x1=*/false, /*reset_tensor=*/x1_2d, /*tanh_act=*/false);

    // 3) proposal = tanh(Linear([reset * x1, x2]))
    auto proposal = compute_gate(proposal_weight, proposal_bias, /*use_reset_x1=*/true, /*reset_tensor=*/reset, /*tanh_act=*/true);

    // 4) out = (1 - update) * x1 + update * proposal  [pointwise HIP kernel]
    auto out = torch::empty_like(x1_2d);

    const int64_t numel = N * D;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);

    hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(
        fused_final_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        update.data_ptr<float>(),
        x1_2d.data_ptr<float>(),
        proposal.data_ptr<float>(),
        out.data_ptr<float>(),
        numel);

    // Return to original shape
    return out.view({B0, B1, B2, D});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GateGRUSelectionLayer HIP forward (optimized)");
}

