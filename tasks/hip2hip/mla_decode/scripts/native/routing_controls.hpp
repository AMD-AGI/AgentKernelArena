// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Protected input policy used by correctness and both timed roles. Reverse the
// physical cache including its 1024 spare slots; valid request tokens are unique.
inline void mla_routing_inputs(std::vector<int32_t>& mapping,
                               std::vector<int32_t>& lengths,
                               int batch, int ctx, std::size_t cache_tokens)
{
    if(batch <= 0 || ctx <= 0 || cache_tokens < std::size_t(batch) * ctx)
        throw std::invalid_argument("invalid MLA routing dimensions");
    mapping.resize(std::size_t(batch) * ctx);
    lengths.resize(batch);
    for(int b = 0; b < batch; ++b)
    {
        const int choices[] = {ctx / 2 + 1, ctx, std::max(1, ctx - 1), 1};
        lengths[b] = choices[b % 4];
        for(int t = 0; t < ctx; ++t)
            mapping[std::size_t(b) * ctx + t] = int32_t(cache_tokens - 1 - (std::size_t(b) * ctx + t));
    }
}

// Q=0 makes attention uniform over the VALID mapped tokens. Each token has a
// constant value across all features, so its exact mean has a closed form.
// Invalid/padded tokens contain -4, making ignored lengths observable; the
// reversed map and spare cache region make identity addressing incorrect too.
template<class Encode>
inline std::vector<float> mla_routing_known_answer(
    std::vector<uint8_t>& kv, const std::vector<int32_t>& mapping,
    const std::vector<int32_t>& lengths, int batch, int ctx, int width, Encode encode)
{
    if(mapping.size() != std::size_t(batch) * ctx || lengths.size() != std::size_t(batch)
       || width <= 0 || kv.size() % width != 0)
        throw std::invalid_argument("invalid MLA known-answer storage");
    std::fill(kv.begin(), kv.end(), encode(-4.0f));
    std::vector<float> means(batch);
    for(int b = 0; b < batch; ++b)
    {
        const int length = lengths[b];
        if(length < 1 || length > ctx) throw std::invalid_argument("invalid MLA valid length");
        const float low = 0.5f + 0.25f * (b % 4), high = 2.0f + 0.5f * (b % 4);
        const int first = length / 2;
        means[b] = float((double(first) * low + double(length - first) * high) / length);
        for(int t = 0; t < length; ++t)
        {
            const int32_t physical = mapping[std::size_t(b) * ctx + t];
            if(physical < 0 || std::size_t(physical) >= kv.size() / width)
                throw std::invalid_argument("MLA mapped token out of bounds");
            std::fill_n(kv.begin() + std::size_t(physical) * width, width, encode(t < first ? low : high));
        }
    }
    return means;
}
