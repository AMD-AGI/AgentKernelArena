// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/full_like.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <string>

namespace py = pybind11;

// =============================================
// HIP Kernels
// =============================================

template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* __restrict__ x, // (N, C)
    const scalar_t* __restrict__ b, // (C)
    int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && c < C) {
        x[n * C + c] = x[n * C + c] + b[c];
    }
}

template <typename scalar_t>
__global__ void gelu_erf_kernel(scalar_t* __restrict__ x, int64_t numel)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        float v = static_cast<float>(x[i]);
        float r = 0.5f * v * (1.0f + erff(v * static_cast<float>(M_SQRT1_2)));
        x[i] = static_cast<scalar_t>(r);
    }
}

template <typename scalar_t>
__global__ void dropout_apply_kernel(
    scalar_t* __restrict__ x,
    const uint8_t* __restrict__ mask,
    float scale,
    int64_t numel)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        x[i] = mask[i] ? static_cast<scalar_t>(static_cast<float>(x[i]) * scale)
                       : static_cast<scalar_t>(0.0f);
    }
}

// =============================================
// Incremental state helpers (mirror Python logic)
// =============================================

static std::mutex g_mutex_inst_id;
static std::unordered_map<std::string, int64_t> g_inc_state_instance_counter;

static std::string get_class_name(const py::handle& module_instance) {
    py::object cls = module_instance.attr("__class__");
    py::object name = cls.attr("__name__");
    return std::string(py::str(name));
}

static std::string get_full_incremental_state_key(const py::handle& module_instance, const std::string& key) {
    std::string module_name = get_class_name(module_instance);
    bool has_id = py::hasattr(module_instance, "_instance_id");
    int64_t instance_id = 0;
    if (!has_id) {
        std::lock_guard<std::mutex> guard(g_mutex_inst_id);
        int64_t& counter = g_inc_state_instance_counter[module_name];
        counter += 1;
        instance_id = counter;
        module_instance.attr("_instance_id") = py::int_(instance_id);
    } else {
        instance_id = py::cast<int64_t>(module_instance.attr("_instance_id"));
    }
    return module_name + "." + std::to_string(instance_id) + "." + key;
}

static py::dict get_input_buffer(const py::handle& module_instance, py::dict incremental_state) {
    std::string full_key = get_full_incremental_state_key(module_instance, "f");
    if (!incremental_state.is_none() && incremental_state.contains(py::str(full_key))) {
        py::object val = incremental_state[py::str(full_key)];
        if (!val.is_none()) {
            try {
                return val.cast<py::dict>();
            } catch (...) {
                // If not a dict, return empty
            }
        }
    }
    return py::dict();
}

static void set_input_buffer(const py::handle& module_instance, py::dict incremental_state, const py::dict& buffer) {
    if (!incremental_state.is_none()) {
        std::string full_key = get_full_incremental_state_key(module_instance, "f");
        incremental_state[py::str(full_key)] = buffer;
    }
}

// =============================================
// Forward implementation matching transformer_ffn_layer_fn
// Signature must match Python call in the harness
// =============================================

torch::Tensor forward(
    torch::Tensor x,                // (T, B, C)
    torch::Tensor ffn1_weight,      // (F, C, K)
    torch::Tensor ffn1_bias,        // (F)
    torch::Tensor ffn2_weight,      // (C_out, F)
    torch::Tensor ffn2_bias,        // (C_out)
    int kernel_size,
    double dropout_p,               // float
    std::string act,                // 'gelu' or 'relu'
    std::string padding,            // 'SAME' or 'LEFT'
    bool training,
    py::object incremental_state_obj, // dict or None
    py::object module_instance        // module or None
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor (ROCm)");
    TORCH_CHECK(ffn1_weight.is_cuda() && ffn1_bias.is_cuda(), "conv params must be CUDA/HIP tensors");
    TORCH_CHECK(ffn2_weight.is_cuda() && ffn2_bias.is_cuda(), "linear params must be CUDA/HIP tensors");

    TORCH_CHECK(x.dim() == 3, "x must be (T, B, C)");
    TORCH_CHECK(ffn1_weight.dim() == 3, "ffn1_weight must be (F, C, K)");
    TORCH_CHECK(ffn1_bias.dim() == 1, "ffn1_bias must be (F)");
    TORCH_CHECK(ffn2_weight.dim() == 2, "ffn2_weight must be (C_out, F)");
    TORCH_CHECK(ffn2_bias.dim() == 1, "ffn2_bias must be (C_out)");

    const int64_t T_in = x.size(0);
    const int64_t B = x.size(1);
    const int64_t C = x.size(2);
    const int64_t F = ffn1_weight.size(0);
    const int64_t Cw = ffn1_weight.size(1);
    const int64_t K = ffn1_weight.size(2);

    TORCH_CHECK(Cw == C, "ffn1_weight in_channels must equal input C");
    TORCH_CHECK(K == kernel_size, "kernel_size must match weight's K");
    TORCH_CHECK(ffn2_weight.size(1) == F, "ffn2_weight in_features must equal F");

    // Handle incremental state behavior
    bool have_incremental = !incremental_state_obj.is_none() && !module_instance.is_none();
    py::dict inc_state;
    if (have_incremental) {
        inc_state = incremental_state_obj.cast<py::dict>();
        py::dict saved = get_input_buffer(module_instance, inc_state);
        if (saved.contains("prev_input")) {
            torch::Tensor prev_input = saved[py::str("prev_input")].cast<torch::Tensor>();
            // concat along time dim 0
            x = at::cat({prev_input, x}, 0);
        }
        // keep last kernel_size frames along time
        if (x.size(0) > kernel_size) {
            x = x.index({at::indexing::Slice(x.size(0) - kernel_size, at::indexing::None)});
        }
        // save current as prev_input
        saved[py::str("prev_input")] = x;
        set_input_buffer(module_instance, inc_state, saved);
    }

    // x: (T, B, C) -> (B, C, T)
    auto x_bct = x.permute({1, 2, 0}).contiguous();

    // Conv1d
    torch::Tensor conv_out_bft;
    if (padding == std::string("SAME")) {
        conv_out_bft = at::conv1d(x_bct, ffn1_weight, ffn1_bias, /*stride=*/1, /*padding=*/kernel_size / 2);
    } else if (padding == std::string("LEFT")) {
        // pad left by (kernel_size - 1) along time dimension
        int64_t pad_left = std::max<int64_t>(0, kernel_size - 1);
        // constant_pad_nd expects pads in reverse order of dimensions: (T_right, T_left)
        conv_out_bft = at::conv1d(at::constant_pad_nd(x_bct, {0, pad_left}, 0.0), ffn1_weight, ffn1_bias);
    } else {
        TORCH_CHECK(false, "Unsupported padding mode: ", padding);
    }

    // (B, F, T) -> (T, B, F)
    auto x_tbf = conv_out_bft.permute({2, 0, 1}).contiguous();

    // Scale by kernel_size ** -0.5
    const float scale = std::pow(static_cast<float>(kernel_size), -0.5f);
    x_tbf.mul_(scale);

    // If incremental_state provided, slice last step
    if (have_incremental) {
        // x = x[-1:]
        int64_t Tnow = x_tbf.size(0);
        x_tbf = x_tbf.index({at::indexing::Slice(Tnow - 1, at::indexing::None)});
    }

    // Activation
    if (act == std::string("gelu")) {
        auto x_contig = x_tbf.contiguous();
        const int64_t numel = x_contig.numel();
        const int threads = 256;
        const int blocks = static_cast<int>((numel + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "gelu_erf_kernel", ([&] {
            hipLaunchKernelGGL(gelu_erf_kernel<scalar_t>, dim3(blocks), dim3(threads), 0, 0,
                               x_contig.data_ptr<scalar_t>(), numel);
        }));
        x_tbf = x_contig;
    } else if (act == std::string("relu")) {
        x_tbf.relu_();
    } else {
        // no-op
    }

    // Dropout
    if (dropout_p > 0.0 && training) {
        const double keep_p = 1.0 - dropout_p;
        auto mask = at::bernoulli(at::full_like(x_tbf, keep_p)).to(torch::kUInt8);
        const float inv_keep = keep_p > 0.0 ? static_cast<float>(1.0 / keep_p) : 0.0f;
        auto x_contig = x_tbf.contiguous();
        const int64_t numel = x_contig.numel();
        const int threads = 256;
        const int blocks = static_cast<int>((numel + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "dropout_apply_kernel", ([&] {
            hipLaunchKernelGGL(dropout_apply_kernel<scalar_t>, dim3(blocks), dim3(threads), 0, 0,
                               x_contig.data_ptr<scalar_t>(), mask.data_ptr<uint8_t>(), inv_keep, numel);
        }));
        x_tbf = x_contig;
    }

    // Linear: x (T, B, F) => (T*B, F)
    int64_t Tnow = x_tbf.size(0);
    auto x_tb_f = x_tbf.view({Tnow * B, F});

    // y = x @ W^T + b
    auto y = at::matmul(x_tb_f, ffn2_weight.t()); // (T*B, C_out)

    // add bias via HIP kernel
    {
        auto y_contig = y.contiguous();
        const int64_t N = Tnow * B;
        const int64_t Cout = ffn2_bias.size(0);
        dim3 threads(16, 16);
        dim3 blocks((static_cast<unsigned int>((N + threads.x - 1) / threads.x)),
                    (static_cast<unsigned int>((Cout + threads.y - 1) / threads.y)));
        AT_DISPATCH_FLOATING_TYPES(y_contig.scalar_type(), "add_bias_kernel", ([&] {
            hipLaunchKernelGGL(add_bias_kernel<scalar_t>, blocks, threads, 0, 0,
                               y_contig.data_ptr<scalar_t>(), ffn2_bias.data_ptr<scalar_t>(),
                               static_cast<int>(N), static_cast<int>(Cout));
        }));
        y = y_contig;
    }

    auto out = y.view({Tnow, B, ffn2_weight.size(0)});
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TransformerFFNLayer functional (HIP/ROCm) with incremental state and padding");
}

