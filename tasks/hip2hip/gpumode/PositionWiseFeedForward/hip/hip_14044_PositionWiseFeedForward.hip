// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>

// The two dense layers dominate this operation. Dispatch them through ATen so
// ROCm can select its tuned rocBLAS implementation for the actual GPU instead
// of computing them with scalar-FMA, 32x32 HIP tiles. The pointwise operations
// remain in the same order as the PyTorch module. In particular, the residual
// add is in-place on dropout's result, just as `out += x` is in the module.
torch::Tensor positionwise_ffn_forward(
    torch::Tensor x,
    torch::Tensor w1_weight,
    torch::Tensor w1_bias,
    torch::Tensor w2_weight,
    torch::Tensor w2_bias,
    torch::Tensor ln_gamma,
    torch::Tensor ln_beta,
    float dropout_p = 0.5f,
    float eps = 1e-5f)
{
    TORCH_CHECK(x.is_cuda(), "x must be on HIP/CUDA device (ROCm)");
    TORCH_CHECK(
        w1_weight.is_cuda() && w1_bias.is_cuda() &&
        w2_weight.is_cuda() && w2_bias.is_cuda() &&
        ln_gamma.is_cuda() && ln_beta.is_cuda(),
        "All tensors must be on device");

    x = x.contiguous();
    w1_weight = w1_weight.contiguous();
    w1_bias = w1_bias.contiguous();
    w2_weight = w2_weight.contiguous();
    w2_bias = w2_bias.contiguous();
    ln_gamma = ln_gamma.contiguous();
    ln_beta = ln_beta.contiguous();

    const int64_t D = x.size(-1);
    TORCH_CHECK(w1_weight.size(1) == D,
                "w1_weight in_features must match D");
    const int64_t H = w1_weight.size(0);
    TORCH_CHECK(w2_weight.size(0) == D && w2_weight.size(1) == H,
                "w2_weight must be [D, H]");
    TORCH_CHECK(
        w1_bias.numel() == H && w2_bias.numel() == D &&
        ln_gamma.numel() == D && ln_beta.numel() == D,
        "bias/gamma/beta sizes must match");

    torch::Tensor hidden = at::linear(x, w1_weight, w1_bias);
    hidden.relu_();
    torch::Tensor projected = at::linear(hidden, w2_weight, w2_bias);
    torch::Tensor dropped = at::dropout(projected, dropout_p, /*train=*/true);
    dropped.add_(x);

    return at::layer_norm(
        dropped,
        at::IntArrayRef{D},
        ln_gamma,
        ln_beta,
        static_cast<double>(eps),
        /*cudnn_enable=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &positionwise_ffn_forward,
          py::arg("x"),
          py::arg("w1_weight"),
          py::arg("w1_bias"),
          py::arg("w2_weight"),
          py::arg("w2_bias"),
          py::arg("ln_gamma"),
          py::arg("ln_beta"),
          py::arg("dropout_p") = 0.5f,
          py::arg("eps") = 1e-5f,
          "Position-wise FeedForward forward (HIP/ROCm)");
}
