#pragma once

#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#define DL_PHILOX_HD __host__ __device__
#else
#define DL_PHILOX_HD
#endif

namespace dl::detail::random_bits {

struct Block { std::uint32_t x, y, z, w; };

DL_PHILOX_HD inline std::uint32_t multiply_high(std::uint32_t a, std::uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(a) * b) >> 32);
#endif
}

/// Philox4x32-10：counter 的前两个字为元素编号，后两个字为显式 stream。
/// 轮函数和常量遵循 https://github.com/DEShawResearch/random123 的算法定义。
DL_PHILOX_HD inline Block philox(std::uint64_t seed, std::uint64_t index, std::uint64_t stream = 0) {
    Block value{static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(index >> 32),
        static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32)};
    auto key0 = static_cast<std::uint32_t>(seed), key1 = static_cast<std::uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        const auto lo0 = value.x * 0xd2511f53u, hi0 = multiply_high(value.x, 0xd2511f53u);
        const auto lo1 = value.z * 0xcd9e8d57u, hi1 = multiply_high(value.z, 0xcd9e8d57u);
        value = {hi1 ^ value.y ^ key0, lo1, hi0 ^ value.w ^ key1, lo0};
        key0 += 0x9e3779b9u;
        key1 += 0xbb67ae85u;
    }
    return value;
}

template <class T> DL_PHILOX_HD T unit(std::uint32_t first, std::uint32_t second, bool open_left = false) {
    if constexpr (std::is_same_v<T, float>) {
        return (static_cast<float>(first >> 8) + static_cast<float>(open_left)) * 0x1p-24f;
    } else {
        const auto bits = (static_cast<std::uint64_t>(first >> 5) << 26) | (second >> 6);
        return (static_cast<double>(bits) + static_cast<double>(open_left)) * 0x1p-53;
    }
}

DL_PHILOX_HD inline bool selected(Block bits, std::uint64_t threshold) {
    const auto value = (static_cast<std::uint64_t>(bits.x >> 5) << 26) | (bits.y >> 6);
    return value < threshold;
}

} // namespace dl::detail::random_bits

#undef DL_PHILOX_HD
