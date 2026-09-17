// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace matrix_reference {
// Independent CPU product. Reuse dot products only for rows/columns proven
// identical by their actual values; no assumption about the input generator.
inline std::vector<double> product(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   std::size_t rows, std::size_t inner,
                                   std::size_t cols) {
    if (!rows || !inner || !cols || a.size() != rows * inner || b.size() != inner * cols)
        throw std::invalid_argument("Invalid matrix reference dimensions");
    std::map<std::vector<float>, std::size_t> unique_rows, unique_cols;
    std::vector<std::vector<float>> row_values, col_values;
    std::vector<std::size_t> row_ids, col_ids;
    auto intern = [](const std::vector<float>& values, auto& known, auto& storage) {
        auto inserted = known.emplace(values, storage.size());
        if (inserted.second) storage.push_back(values);
        return inserted.first->second;
    };
    for (std::size_t row = 0; row < rows; ++row) {
        std::vector<float> values(a.begin() + row * inner, a.begin() + (row + 1) * inner);
        row_ids.push_back(intern(values, unique_rows, row_values));
    }
    for (std::size_t col = 0; col < cols; ++col) {
        std::vector<float> values(inner);
        for (std::size_t k = 0; k < inner; ++k) values[k] = b[k * cols + col];
        col_ids.push_back(intern(values, unique_cols, col_values));
    }
    std::vector<double> dots(row_values.size() * col_values.size());
    for (std::size_t row = 0; row < row_values.size(); ++row)
        for (std::size_t col = 0; col < col_values.size(); ++col)
            for (std::size_t k = 0; k < inner; ++k)
                dots[row * col_values.size() + col] +=
                    static_cast<double>(row_values[row][k]) * static_cast<double>(col_values[col][k]);
    std::vector<double> expected(rows * cols);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t col = 0; col < cols; ++col)
            expected[row * cols + col] = dots[row_ids[row] * col_values.size() + col_ids[col]];
    return expected;
}

inline std::string validate(const std::vector<float>& actual,
                            const std::vector<double>& expected, std::size_t cols) {
    if (!cols || actual.size() != expected.size()) return "Matrix output size differs";
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double tolerance = 2.0e-3 * std::max(1.0, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            std::fabs(static_cast<double>(actual[i]) - expected[i]) > tolerance) {
            std::ostringstream message;
            message << "C[" << i / cols << "," << i % cols << "]=" << actual[i]
                    << ", expected=" << expected[i];
            return message.str();
        }
    }
    return {};
}
} // namespace matrix_reference
