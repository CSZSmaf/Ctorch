#pragma once

#include <bit>
#include <cstdint>

namespace dl {
namespace detail {

/// 二进制右移采用 round-to-nearest-even，处理尾数舍入和次正规数。
constexpr std::uint32_t rounded_shift(std::uint32_t value, unsigned shift) {
    return (value + ((1u << (shift - 1)) - 1) + ((value >> shift) & 1)) >> shift;
}

constexpr std::uint16_t encode_float16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = (bits >> 16) & 0x8000u;
    const auto fraction = bits & 0x7fffffu;
    const auto exponent = (bits >> 23) & 0xffu;
    if (exponent == 255) return static_cast<std::uint16_t>(sign | 0x7c00u | (fraction ? (fraction >> 13) | 0x200u : 0u));
    const auto adjusted = static_cast<int>(exponent) - 112;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (adjusted <= 0) {
        const auto magnitude = adjusted < -10 ? 0u : rounded_shift(fraction | 0x800000u, 14 - adjusted);
        return static_cast<std::uint16_t>(sign | magnitude);
    }
    return static_cast<std::uint16_t>(sign | ((adjusted << 10) + rounded_shift(fraction, 13)));
}

constexpr float decode_float16(std::uint16_t value) {
    const auto sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const auto exponent = (value >> 10) & 31u;
    auto fraction = static_cast<std::uint32_t>(value & 0x3ffu);
    if (exponent == 0) {
        if (fraction == 0) return std::bit_cast<float>(sign);
        unsigned normalized = 113;
        while ((fraction & 0x400u) == 0) { fraction <<= 1; --normalized; }
        return std::bit_cast<float>(sign | (normalized << 23) | ((fraction & 0x3ffu) << 13));
    }
    const auto wide_exponent = exponent == 31 ? 255u : exponent + 112u;
    return std::bit_cast<float>(sign | (wide_exponent << 23) | (fraction << 13));
}

constexpr std::uint16_t encode_bfloat16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1)) >> 16);
}

} // namespace detail

/// 16 位存储值；宿主计算通过 float 转换，原始 bits 可用于文件与 C 接口交换。
struct Float16 {
    std::uint16_t bits = 0;
    constexpr Float16() = default;
    constexpr Float16(float value) : bits(detail::encode_float16(value)) {}
    constexpr operator float() const { return detail::decode_float16(bits); }
    static constexpr Float16 from_bits(std::uint16_t value) { Float16 result; result.bits = value; return result; }
};

struct BFloat16 {
    std::uint16_t bits = 0;
    constexpr BFloat16() = default;
    constexpr BFloat16(float value) : bits(detail::encode_bfloat16(value)) {}
    constexpr operator float() const { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }
    static constexpr BFloat16 from_bits(std::uint16_t value) { BFloat16 result; result.bits = value; return result; }
};

} // namespace dl
