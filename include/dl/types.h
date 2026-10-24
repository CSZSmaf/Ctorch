#pragma once

#include "dl/export.h"
#include "dl/half.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dl {

using Index = std::int64_t;
using Shape = std::vector<Index>;

enum class DType : std::uint8_t { f32, f64, i64, boolean, f16, bf16 };
enum class DeviceType : std::uint8_t { cpu, cuda };
enum class KeepDim { no, yes };

/// 设备值描述存储的位置，CPU 后端的设备编号为零。
struct Device {
    DeviceType type = DeviceType::cpu;
    int index = 0;

    static constexpr Device cpu() { return {}; }
    static constexpr Device cuda(int index = 0) { return {DeviceType::cuda, index}; }
    bool operator==(const Device&) const = default;
};

struct TensorOptions {
    DType dtype = DType::f32;
    Device device = Device::cpu();
    /// 浮点创建器可将结果登记为可求导叶张量。
    bool requires_grad = false;
};

/// 等待所选设备已经提交的操作；CPU 操作在返回前完成。
DL_API void synchronize(Device device);

constexpr std::size_t element_size(DType dtype) {
    constexpr std::size_t sizes[] = {sizeof(float), sizeof(double), sizeof(Index), sizeof(bool), sizeof(Float16), sizeof(BFloat16)};
    return sizes[static_cast<std::size_t>(dtype)];
}

constexpr bool is_low_precision(DType dtype) { return dtype == DType::f16 || dtype == DType::bf16; }
constexpr bool is_floating(DType dtype) { return dtype == DType::f32 || dtype == DType::f64 || is_low_precision(dtype); }
constexpr DType accumulation_dtype(DType dtype) { return is_low_precision(dtype) ? DType::f32 : dtype; }

template <class T> struct DTypeOf;
template <> struct DTypeOf<float> { static constexpr auto value = DType::f32; };
template <> struct DTypeOf<double> { static constexpr auto value = DType::f64; };
template <> struct DTypeOf<Index> { static constexpr auto value = DType::i64; };
template <> struct DTypeOf<bool> { static constexpr auto value = DType::boolean; };
template <> struct DTypeOf<Float16> { static constexpr auto value = DType::f16; };
template <> struct DTypeOf<BFloat16> { static constexpr auto value = DType::bf16; };
template <class T> inline constexpr DType dtype_of = DTypeOf<T>::value;

} // namespace dl
