// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// Supplementary decoded-float check. Raw BF16 storage must be checked before
// promotion; fast-math assumptions can invalidate checks on a promoted value.
inline bool mla_finite_float(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

// Inspect encoded storage before decoding BF16. Under -ffast-math a BF16
// promotion may carry nnan/ninf assumptions, so checking the promoted float's
// bits can be optimized away. The float branch retains CPU reference fixtures.
template<class Value>
inline bool mla_finite_storage(const Value& value)
{
    static_assert(sizeof(Value) == 2 || sizeof(Value) == 4,
                  "MLA output storage must be BF16 or float");
    if constexpr(sizeof(Value) == 2)
    {
        uint16_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f80U) != 0x7f80U;
    }
    else
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f800000U) != 0x7f800000U;
    }
}

template<class Values, class Decode>
std::string validate_mla_outputs(const Values& actual, const Values& eager,
                                 const Values& reference, Decode decode)
{
    if(actual.size() != eager.size() || actual.size() != reference.size())
        return "MLA output sizes differ";
    double max_abs = 0.0, max_rel = 0.0;
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        if(!mla_finite_storage(actual[i]) || !mla_finite_storage(eager[i])
           || !mla_finite_storage(reference[i]))
            return "MLA output contains NaN/Inf at " + std::to_string(i);
        const float value = decode(actual[i]);
        const float old = decode(eager[i]);
        const float expected = decode(reference[i]);
        if(!mla_finite_float(value) || !mla_finite_float(old) || !mla_finite_float(expected))
            return "MLA output contains NaN/Inf at " + std::to_string(i);
        // Preserve the existing eager/replay consistency gate as well.
        if(std::fabs(value - old) > 1.0e-2f)
            return "MLA eager/replay mismatch at " + std::to_string(i);
        const double x = value, y = expected;
        const double difference = std::fabs(x - y);
        max_abs = std::max(max_abs, difference);
        max_rel = std::max(max_rel, difference / std::max(1e-8, std::max(std::fabs(x), std::fabs(y))));
    }
    // Original independent host-reference rule; neither threshold is loosened.
    if(!(max_abs <= 5e-2 || max_rel <= 1e-1))
        return "MLA output disagrees with independent host reference";
    return {};
}

// Preserve original buffers even if a validation callback throws while building
// diagnostics. Restoration failure cannot turn a failed check into success.
template<class Validate, class Restore>
std::string mla_with_restored_inputs(Validate validate, Restore restore)
{
    std::string result;
    try { result = validate(); }
    catch(...) { restore(); throw; }
    if(!restore()) return "MLA failed to restore original input buffers";
    return result;
}

template<class Values>
bool mla_same_buffer(const Values& left, const Values& right)
{
    return left.size() == right.size() &&
        std::memcmp(left.data(), right.data(), left.size() * sizeof(typename Values::value_type)) == 0;
}
