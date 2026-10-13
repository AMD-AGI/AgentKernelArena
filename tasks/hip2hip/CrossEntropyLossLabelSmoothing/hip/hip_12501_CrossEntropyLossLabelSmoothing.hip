// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <vector>
#include <string>
#include <cmath>

// One thread computes loss for one row (all classes) to avoid inter-thread comms.
// Supports:
// - from_logits: compute log_softmax on the fly, else treat input as log-probabilities
// - target as Long (indices) or Float (distribution)
// - class weights (per-class scaling of log-probs)
// - ignore_index for Long targets
// - label smoothing epsilon: if smooth_dist provided and Long target -> mix onehot with smooth_dist;
//   else if no smooth_dist -> use eps_nll/eps_sum formulation. For Float targets smoothing is handled on host.
extern "C" __global__ void cross_entropy_label_smoothing_kernel(
    const float* __restrict__ input,      // [outer, C]
    const float* __restrict__ target_dist,// [outer, C] or nullptr
    const int64_t* __restrict__ target_idx, // [outer] or nullptr
    const float* __restrict__ weight,     // [C] or nullptr
    const float* __restrict__ smooth_dist,// [C] or [outer,C] or nullptr (Long targets only)
    float* __restrict__ out_loss,         // [outer]
    int32_t* __restrict__ out_valid,      // [outer]
    int64_t outer,
    int64_t C,
    int64_t ignore_index,
    int from_logits,
    float smooth_eps,
    int has_weight,
    int tgt_is_long,
    int tgt_is_float,
    int has_smooth,
    int smooth_per_row
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;

    const float* x = input + row * C;

    // Compute logsumexp if needed
    float lse = 0.0f;
    if (from_logits) {
        float m = -INFINITY;
        // max
        for (int64_t c = 0; c < C; ++c) {
            float v = x[c];
            m = v > m ? v : m;
        }
        // sumexp
        float s = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            s += expf(x[c] - m);
        }
        lse = m + logf(s);
    }

    int32_t valid = 1;
    int64_t t_idx = -1;
    if (tgt_is_long) {
        t_idx = target_idx[row];
        if (ignore_index >= 0 && t_idx == ignore_index) {
            valid = 0;
            out_loss[row] = 0.0f;
            out_valid[row] = 0;
            return;
        }
    }

    const float* smooth_row = nullptr;
    if (has_smooth && tgt_is_long) {
        smooth_row = smooth_per_row ? (smooth_dist + row * C) : smooth_dist;
    }

    float loss_val = 0.0f;

    if (tgt_is_long) {
        // Integer targets
        if (smooth_eps > 0.0f && has_smooth) {
            // Mix onehot with provided smooth distribution
            float accum = 0.0f;
            float one_minus_eps = 1.0f - smooth_eps;
            for (int64_t c = 0; c < C; ++c) {
                float lsm = from_logits ? (x[c] - lse) : x[c];
                if (has_weight) lsm *= weight[c];
                float oh = (c == t_idx) ? 1.0f : 0.0f;
                float t = one_minus_eps * oh + smooth_eps * smooth_row[c];
                accum += t * lsm;
            }
            loss_val = -accum;
        } else {
            // eps_nll/eps_sum formulation (covers smooth_eps==0 or smooth_dist not provided)
            float sum_lsm = 0.0f;
            float like = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                float lsm = from_logits ? (x[c] - lse) : x[c];
                if (has_weight) lsm *= weight[c];
                sum_lsm += lsm;
                if (c == t_idx) like = lsm;
            }
            float eps_sum = (smooth_eps > 0.0f) ? (smooth_eps / (float)C) : 0.0f;
            float eps_nll = 1.0f - eps_sum - smooth_eps;
            loss_val = -(eps_nll * like + eps_sum * sum_lsm);
        }
    } else if (tgt_is_float) {
        // Distribution targets; optional pre-applied smoothing on host
        const float* trow = target_dist + row * C;
        float accum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            float lsm = from_logits ? (x[c] - lse) : x[c];
            if (has_weight) lsm *= weight[c];
            accum += trow[c] * lsm;
        }
        loss_val = -accum;
    } else {
        // Shouldn't happen
        valid = 0;
        loss_val = 0.0f;
    }

    out_loss[row] = loss_val;
    out_valid[row] = valid;
}

static inline std::tuple<int64_t,int64_t> flatten_outer(const at::Tensor& t) {
    int64_t C = t.size(-1);
    int64_t outer = 1;
    for (int i = 0; i < t.dim() - 1; ++i) outer *= t.size(i);
    return {outer, C};
}

static inline at::Tensor make_scalar_like(const at::Tensor& ref, float v) {
    return at::full({}, v, ref.options());
}

// Full signature expected by the Python module
at::Tensor forward(
    at::Tensor input,
    at::Tensor target,
    c10::optional<at::Tensor> weight_opt,
    int64_t ignore_index,
    std::string reduction,
    c10::optional<double> smooth_eps_opt,
    c10::optional<at::Tensor> smooth_dist_opt,
    bool from_logits
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA/HIP tensor");
    TORCH_CHECK(target.is_cuda(), "target must be CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dim");

    const float smooth_eps = smooth_eps_opt.has_value() ? static_cast<float>(*smooth_eps_opt) : 0.0f;

    // Shapes
    auto [outer, C] = flatten_outer(input);
    auto input_2d = input.contiguous().view({outer, C});

    bool tgt_is_long = (target.scalar_type() == at::kLong) && (target.dim() == input.dim() - 1);
    bool tgt_is_float = (target.scalar_type() == at::kFloat) && (target.dim() == input.dim());
    TORCH_CHECK(tgt_is_long || tgt_is_float, "target must be Long indices with shape input.shape[:-1] or Float distribution with shape equal to input");

    at::Tensor target_idx;
    at::Tensor target_dist;

    if (tgt_is_long) {
        TORCH_CHECK(target.is_contiguous(), "Long target must be contiguous");
        target_idx = target.view({outer});
    } else {
        TORCH_CHECK(target.sizes() == input.sizes(), "float target must match input shape");
        target_dist = target.contiguous().view({outer, C});
    }

    // Class weights
    const float* weight_ptr = nullptr;
    at::Tensor weight;
    int has_weight = 0;
    if (weight_opt.has_value() && weight_opt->defined()) {
        weight = weight_opt->contiguous();
        TORCH_CHECK(weight.is_cuda(), "weight must be CUDA/HIP tensor");
        TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must be 1-D with size == num_classes");
        weight_ptr = weight.data_ptr<float>();
        has_weight = 1;
    }

    // smooth_dist handling:
    // - If float target and smooth_dist provided with smooth_eps>0: precompute mixed target on host (GPU) to simplify kernel.
    // - If long target and smooth_dist provided: pass pointer to kernel to mix with onehot.
    at::Tensor smooth_dist;
    const float* smooth_ptr = nullptr;
    int has_smooth = 0;
    int smooth_per_row = 0;

    if (smooth_dist_opt.has_value() && smooth_dist_opt->defined() && smooth_eps > 0.0f) {
        at::Tensor sd = smooth_dist_opt.value();
        TORCH_CHECK(sd.is_cuda(), "smooth_dist must be CUDA/HIP tensor");
        TORCH_CHECK(sd.scalar_type() == at::kFloat, "smooth_dist must be float32");
        if (tgt_is_float) {
            // Pre-mix on device: target = lerp(target, smooth_dist, smooth_eps)
            if (sd.dim() < target.dim()) sd = sd.unsqueeze(0); // broadcast like PyTorch func
            TORCH_CHECK(sd.sizes() == target.sizes(), "smooth_dist must broadcast to target shape");
            target_dist = at::lerp(target_dist, sd.contiguous().view({outer, C}), smooth_eps);
        } else {
            // Long target: pass to kernel (allow [C] or [outer, C])
            if (sd.dim() == 1) {
                TORCH_CHECK(sd.size(0) == C, "smooth_dist 1-D must have size C");
                smooth_dist = sd.contiguous();
                smooth_ptr = smooth_dist.data_ptr<float>();
                has_smooth = 1;
                smooth_per_row = 0;
            } else {
                TORCH_CHECK(sd.dim() == input.dim() && sd.sizes() == input.sizes(), "smooth_dist must be shape [..., C]");
                smooth_dist = sd.contiguous().view({outer, C});
                smooth_ptr = smooth_dist.data_ptr<float>();
                has_smooth = 1;
                smooth_per_row = 1;
            }
        }
    }

    // Allocate outputs
    at::Tensor loss_per_row = at::empty({outer}, input.options());
    at::Tensor valid_mask = at::empty({outer}, input.options().dtype(at::kInt));

    // Launch
    int threads = 256;
    int64_t blocks = (outer + threads - 1) / threads;

    hipLaunchKernelGGL(
        cross_entropy_label_smoothing_kernel,
        dim3(blocks), dim3(threads), 0, c10::cuda::getCurrentCUDAStream(),
        input_2d.data_ptr<float>(),
        tgt_is_float ? target_dist.data_ptr<float>() : nullptr,
        tgt_is_long ? target_idx.data_ptr<int64_t>() : nullptr,
        weight_ptr,
        (tgt_is_long ? smooth_ptr : nullptr),
        loss_per_row.data_ptr<float>(),
        valid_mask.data_ptr<int32_t>(),
        outer,
        C,
        ignore_index,
        from_logits ? 1 : 0,
        smooth_eps,
        has_weight,
        tgt_is_long ? 1 : 0,
        tgt_is_float ? 1 : 0,
        (tgt_is_long ? has_smooth : 0),
        (tgt_is_long ? smooth_per_row : 0)
    );

    // Reduction handling
    if (reduction == "none") {
        // reshape to outer dims (input.shape[:-1])
        std::vector<int64_t> out_shape;
        out_shape.reserve(input.dim() - 1);
        for (int i = 0; i < input.dim() - 1; ++i) out_shape.push_back(input.size(i));
        return loss_per_row.view(out_shape);
    } else if (reduction == "sum") {
        at::Tensor total = loss_per_row.sum();
        return total; // 0-D scalar
    } else {
        // mean over valid rows
        at::Tensor total = loss_per_row.sum();
        at::Tensor valid = valid_mask.toType(at::kLong).sum();
        // If Float targets or no ignore_index, valid == outer
        // Avoid div by zero
        int64_t valid_count = valid.item<int64_t>();
        if (valid_count == 0) {
            return make_scalar_like(input, 0.0f);
        }
        at::Tensor denom = at::full({}, static_cast<double>(valid_count), total.options());
        return total / denom;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = c10::optional<at::Tensor>{},
        py::arg("ignore_index") = -100,
        py::arg("reduction") = std::string("mean"),
        py::arg("smooth_eps") = c10::optional<double>{},
        py::arg("smooth_dist") = c10::optional<at::Tensor>{},
        py::arg("from_logits") = true
    );
}

