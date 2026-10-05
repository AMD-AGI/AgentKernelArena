// SPDX-License-Identifier: MIT
// Observed native export copied from AITER's QUANT_PYBIND macro.
// No enum/type registration and no unrelated MXFP4/MXFP6 exports.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("dynamic_per_token_scaled_quant",
          &aiter::dynamic_per_token_scaled_quant,
          py::arg("out"),
          py::arg("input"),
          py::arg("scales"),
          py::arg("scale_ub") = std::nullopt,
          py::arg("shuffle_scale") = false,
          py::arg("num_rows") = std::nullopt,
          py::arg("num_rows_factor") = 1);
}
