// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace matrix_inputs {
// SplitMix64 is used only as a reproducible host input generator. Values are
// independently drawn per element; no row/column repetition is constructed.
inline std::uint64_t next(std::uint64_t& state) {
    std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
inline std::uint64_t fresh_seed() {
    std::random_device entropy;
    return (static_cast<std::uint64_t>(entropy()) << 32)
           ^ static_cast<std::uint64_t>(entropy());
}
inline float value(std::uint64_t& state, float scale) {
    const int raw = static_cast<int>(next(state) % 30);
    const int nonzero = raw < 15 ? raw - 15 : raw - 14;
    return static_cast<float>(nonzero) * scale;
}
inline void fill(std::vector<float>& a, std::vector<float>& b, std::uint64_t seed) {
    std::uint64_t state = seed;
    for (float& x : a) x = value(state, 0.03125f);
    for (float& x : b) x = value(state, 0.015625f);
}
// FNV-1a64 over canonical little-endian FP32 bit patterns is a noncryptographic
// reproducibility fingerprint, not a security hash or provenance signature.
inline std::string fingerprint(const std::vector<float>& values) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "FP32 inputs required");
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (float value : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (byte * 8)) & 255U;
            hash *= UINT64_C(1099511628211);
        }
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}
inline bool unchanged(const std::vector<float>& values, const std::vector<float>& original) {
    return values.size() == original.size()
           && std::memcmp(values.data(), original.data(), values.size() * sizeof(float)) == 0;
}
inline std::string record(std::uint64_t initial_seed, std::uint64_t replay_seed,
                          const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& replay_a, const std::vector<float>& replay_b) {
    const auto a_hash = fingerprint(a), b_hash = fingerprint(b);
    const auto replay_a_hash = fingerprint(replay_a), replay_b_hash = fingerprint(replay_b);
    if (initial_seed == replay_seed || a_hash == replay_a_hash || b_hash == replay_b_hash)
        throw std::invalid_argument("Independent replay must change both supplied matrices");
    std::ostringstream out;
    out << "{\"schema_version\":1,\"generator\":\"splitmix64_dyadic_dense_v1\","
        << "\"fingerprint_kind\":\"fnv1a64_le_fp32_noncryptographic\","
        << "\"initial_seed\":\"" << initial_seed << "\",\"replay_seed\":\"" << replay_seed
        << "\",\"initial_A\":\"" << a_hash << "\",\"initial_B\":\"" << b_hash
        << "\",\"replay_A\":\"" << replay_a_hash << "\",\"replay_B\":\"" << replay_b_hash << "\"}";
    return out.str();
}
} // namespace matrix_inputs
